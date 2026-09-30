from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.blocks import QuantConv1D, QuantReLU
from src.models.quant_autoencoder import ECGDenoiseAutoencoder


class StaticInferenceWrapper(nn.Module):
    """Expose the fixed deployment graph without runtime shape assertions."""

    def __init__(self, model: ECGDenoiseAutoencoder):
        super().__init__()
        self.layers = model.layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


def load_checkpoint(path: Path) -> ECGDenoiseAutoencoder:
    """Load a trained checkpoint and verify it produces the required shape."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Train a model before HLS export."
        )
    model = ECGDenoiseAutoencoder()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 256))
    if tuple(output.shape) != (1, 1, 256):
        raise RuntimeError(f"Unexpected model output shape: {tuple(output.shape)}")
    return model


def _strip_quantizers(module: nn.Module) -> None:
    """Replace Brevitas wrappers with standard inference operators for ONNX."""
    for name, child in list(module.named_children()):
        if isinstance(child, QuantConv1D):
            replacement = nn.Conv1d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                bias=child.bias is not None,
            )
            replacement.weight.data.copy_(child.weight.detach().float())
            if child.bias is not None:
                replacement.bias.data.copy_(child.bias.detach().float())
            setattr(module, name, replacement)
        elif isinstance(child, QuantReLU):
            setattr(module, name, nn.ReLU())
        else:
            _strip_quantizers(child)


def export_onnx(model: ECGDenoiseAutoencoder, path: Path) -> None:
    """Export a static-shape ONNX intermediate for hls4ml."""
    export_model = model
    _strip_quantizers(export_model)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            model,
            torch.zeros(1, 1, 256),
            str(path),
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ONNX export requires the optional 'onnx' package. "
            "Install it with: python -m pip install onnx"
        ) from error


def convert_pytorch_to_hls(
    model: ECGDenoiseAutoencoder,
    output_dir: Path,
    precision: str,
    io_type: str,
) -> None:
    """Convert the stripped PyTorch inference graph into Vitis HLS.

    Vivado HLS was retired after the 2019.x line; Vivado/Vitis 2023.2 only
    ships Vitis HLS, so hls4ml must target the "Vitis" backend rather than
    the legacy "Vivado" backend.
    """
    try:
        import hls4ml
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HLS export requires optional package 'hls4ml'. "
            "Install the pinned FPGA toolchain before exporting."
        ) from error

    hls_config = hls4ml.utils.config_from_pytorch_model(
        model,
        granularity="layer",
        backend="Vitis",
        default_precision=precision,
        inputs_channel_last=True,
        transpose_outputs=False,
    )
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=(None, 1, 256),
        output_dir=str(output_dir),
        project_name="ecg_denoiser",
        backend="Vitis",
        hls_config=hls_config,
        io_type=io_type,
        part="xczu7ev-ffvc1156-2-e",
        clock_period=5,
    )
    hls_model.write()
    print(f"Generated HLS project at {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the trained ECG denoiser through ONNX to Vivado HLS."
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", type=Path)
    parser.add_argument("--onnx", default="artifacts/ecg_denoiser.onnx", type=Path)
    parser.add_argument("--output-dir", default="artifacts/ecg_denoiser_hls", type=Path)
    parser.add_argument("--precision", default="ap_fixed<16,6>")
    parser.add_argument("--io-type", choices=("io_stream", "io_parallel"), default="io_stream")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint)
    export_onnx(model, args.onnx)
    print(f"Exported ONNX intermediate to {args.onnx}")
    _strip_quantizers(model)
    convert_pytorch_to_hls(
        StaticInferenceWrapper(model), args.output_dir, args.precision, args.io_type
    )


if __name__ == "__main__":
    main()
