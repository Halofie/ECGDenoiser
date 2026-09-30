"""1D ECG denoising ResUNet, implemented as a 2D graph for DPU deployment.

The Vitis AI PyTorch quantizer/compiler targets the DPUCZDX8G IP, which is a
2D image-processing core. nn.Conv1d is not a reliably compilable op through
vai_q_pytorch / vai_c_xir. The standard workaround is to represent the 1D
signal as a (N, C, 1, L) tensor and build the network entirely from
nn.Conv2d / nn.ConvTranspose2d with kernel shape (1, k) -- this keeps every
layer expressible as an op the DPU compiler recognizes.

Input contract: (N, 1, 1, 512) float32, already reshaped by the caller.
Reshape/view ops are intentionally kept OUT of forward() so the traced graph
handed to the quantizer contains only Conv2d/BatchNorm2d/ReLU/add ops.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

INPUT_LENGTH = 512


def to_dpu_input(x: torch.Tensor) -> torch.Tensor:
    """Convert a (N, 1, L) or (N, L) signal batch to the (N, 1, 1, L) model input."""
    if x.dim() == 2:
        x = x.unsqueeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(2)
    if x.shape[-1] != INPUT_LENGTH:
        raise ValueError(f"Expected length {INPUT_LENGTH}, got {x.shape[-1]}")
    return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int = 1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, (1, kernel), stride=(1, stride), padding=(0, pad))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    """Two 1x-kernel convs with a residual add. Requires stride 1 / same channels."""

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(channels, channels, (1, kernel), padding=(0, pad))
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, (1, kernel), padding=(0, pad))
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu(out)


class DownBlock(nn.Module):
    """Stride-2 downsample (length L -> L/2) followed by a residual block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.down = ConvBNReLU(in_ch, out_ch, kernel=5, stride=2)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """Stride-2 transposed-conv upsample (length L -> 2L), residual-added with the skip."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn(self.up(x)))
        out = out + skip
        return self.res(out)


class ECGResUNet1D(nn.Module):
    """Lightweight 1D ResUNet for ECG denoising, expressed as 2D DPU-friendly ops.

    512 -> 256 -> 128 -> 64 (bottleneck) -> 128 -> 256 -> 512
    Skip connections use elementwise add (channel counts match at each level),
    which avoids concat-related channel-layout issues on the DPU compiler.
    """

    def __init__(self, base_channels: int = 16):
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.stem = ConvBNReLU(1, c1, kernel=7)

        self.down1 = DownBlock(c1, c2)  # 512 -> 256
        self.down2 = DownBlock(c2, c3)  # 256 -> 128
        self.down3 = DownBlock(c3, c4)  # 128 -> 64

        self.bottleneck = ResBlock(c4)

        self.up3 = UpBlock(c4, c3)  # 64 -> 128
        self.up2 = UpBlock(c3, c2)  # 128 -> 256
        self.up1 = UpBlock(c2, c1)  # 256 -> 512

        self.head = nn.Conv2d(c1, 1, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s1 = self.stem(x)        # (N, c1, 1, 512)
        s2 = self.down1(s1)      # (N, c2, 1, 256)
        s3 = self.down2(s2)      # (N, c3, 1, 128)
        b = self.down3(s3)       # (N, c4, 1, 64)

        b = self.bottleneck(b)

        u3 = self.up3(b, s3)     # (N, c3, 1, 128)
        u2 = self.up2(u3, s2)    # (N, c2, 1, 256)
        u1 = self.up1(u2, s1)    # (N, c1, 1, 512)

        return self.head(u1)     # (N, 1, 1, 512)


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, path)


def load_checkpoint(model: nn.Module, path: str | Path, map_location: str = "cpu") -> nn.Module:
    # weights_only= was only added to torch.load in later torch releases;
    # the older torch bundled in some Vitis AI Docker envs (e.g. the 3.0
    # image's vitis-ai-wego-torch env, torch 1.x/py3.7) raises TypeError on
    # an unrecognized kwarg rather than ignoring it. Checkpoints saved by
    # this repo only ever contain a plain state_dict, so falling back to a
    # full unpickle on old torch is not a meaningful security regression
    # for our own trusted checkpoints -- just an unavailable safety check.
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(model: nn.Module, path: str | Path, opset: int = 13) -> None:
    """Export a static-shape ONNX graph. Reference/portability only -- the DPU
    deployment path below goes through the PyTorch quantizer directly, not ONNX.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    dummy = torch.zeros(1, 1, 1, INPUT_LENGTH)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        do_constant_folding=True,
    )


if __name__ == "__main__":
    model = ECGResUNet1D()
    dummy = torch.zeros(2, 1, 1, INPUT_LENGTH)
    out = model(dummy)
    assert tuple(out.shape) == (2, 1, 1, INPUT_LENGTH), out.shape
    print(f"OK: output shape {tuple(out.shape)}")
    save_checkpoint(model, "checkpoints/ecg_resunet1d_init.pt")
    export_onnx(model, "artifacts/ecg_resunet1d.onnx")
    print("Saved checkpoints/ecg_resunet1d_init.pt and artifacts/ecg_resunet1d.onnx")
