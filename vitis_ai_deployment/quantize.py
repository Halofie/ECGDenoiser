"""Post-Training Quantization (PTQ) of ECGResUNet1D using the Vitis AI 3.5
PyTorch quantizer (pytorch_nndct), producing a DPU-deployable xmodel.

Must run in two passes, per the vai_q_pytorch calibration flow:

  1. calib -- runs the model in fake-quantized mode over a calibration
     dataset to collect per-tensor activation statistics, then writes
     quantize_result/quant_info.json.

  2. test  -- reloads those stats and runs ONE forward pass to trace and
     export the deployable graph as an xmodel.

This must run inside the Vitis AI 3.5 PyTorch Docker image
(vitis-ai-pytorch-cpu or vitis-ai-pytorch-gpu), which pins the matching
torch/pytorch_nndct versions. It has not been executed in this workspace --
no Vitis AI install is present here -- so the exact torch_quantizer call
signature should be checked against `python -c "import pytorch_nndct;
help(pytorch_nndct.apis.torch_quantizer)"` in your container before relying
on it; Xilinx has changed argument names across 2.5/3.0/3.5 releases.

Usage:
    python quantize.py --mode calib --checkpoint checkpoints/ecg_resunet1d.pt
    python quantize.py --mode test   --checkpoint checkpoints/ecg_resunet1d.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from model import INPUT_LENGTH, ECGResUNet1D, load_checkpoint
from train import build_datasets


def run_calibration_forward_passes(
    quant_model: torch.nn.Module, dataset: Dataset, num_batches: int, device: str
) -> None:
    """Feed real (or representative) data through the quantized model.

    Calibrating on the synthetic placeholder dataset (used only when
    --clean-dir/--noise-dir are omitted) will produce activation ranges that
    do not match real ECG signals and will silently degrade INT8 accuracy on
    the board -- always pass --clean-dir/--noise-dir pointing at real,
    representative data for calibration.

    shuffle=True (fixed seed, for a reproducible calibration set) matters
    here: ECG512Dataset's windows are appended record-by-record, so with
    shuffle=False the first `num_batches` batches come from only the first
    one or two records in the training list -- calibration would silently
    see one patient's amplitude/noise distribution and miss the rest,
    degrading INT8 accuracy for everyone else without erroring.
    """
    generator = torch.Generator().manual_seed(0)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, generator=generator)
    quant_model.eval()
    with torch.no_grad():
        for i, (noisy, _clean) in enumerate(loader):
            if i >= num_batches:
                break
            noisy = noisy.unsqueeze(2).to(device)
            quant_model(noisy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vitis AI PTQ for ECGResUNet1D.")
    parser.add_argument("--mode", choices=("calib", "test"), required=True)
    parser.add_argument("--checkpoint", default="checkpoints/ecg_resunet1d.pt", type=Path)
    parser.add_argument("--output-dir", default="artifacts/quantize_result", type=Path)
    parser.add_argument("--calib-batches", type=int, default=100)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean-dir", default=None, help="Local MIT-BIH clean record directory (e.g. data/clean).")
    parser.add_argument("--noise-dir", default=None, help="Local nstdb noise record directory.")
    parser.add_argument("--records", default=None, help="Comma-separated MIT-BIH record names (default: full list).")
    parser.add_argument("--val-records", default=None, help="Comma-separated record names held out for validation.")
    parser.add_argument("--download-missing", action="store_true", default=True)
    args = parser.parse_args()

    try:
        from pytorch_nndct.apis import torch_quantizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Quantization requires the Vitis AI PyTorch quantizer "
            "(pytorch_nndct). Run inside the vitis-ai-pytorch-cpu/gpu "
            "Docker image, not the project's plain training venv."
        ) from error

    model = ECGResUNet1D()
    load_checkpoint(model, args.checkpoint, map_location=args.device)
    model.to(args.device)

    dummy_input = torch.randn(1, 1, 1, INPUT_LENGTH).to(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    quantizer = torch_quantizer(
        quant_mode=args.mode,
        module=model,
        input_args=(dummy_input,),
        output_dir=str(args.output_dir),
    )
    quant_model = quantizer.quant_model

    if args.mode == "calib":
        dataset, _ = build_datasets(args)
        run_calibration_forward_passes(quant_model, dataset, args.calib_batches, args.device)
        quantizer.export_quant_config()
        print(f"Calibration complete. Quant config written under {args.output_dir}")
        print("Next: re-run this script with --mode test to export the xmodel.")
    else:
        quant_model.eval()
        with torch.no_grad():
            quant_model(dummy_input)
        quantizer.export_xmodel(output_dir=str(args.output_dir), deploy_check=True)
        xmodel_files = sorted(args.output_dir.glob("*.xmodel"))
        print(f"Exported xmodel(s): {[str(p) for p in xmodel_files]}")
        print(
            "Rename/copy the ECGResUNet1D_int.xmodel above to quantized.xmodel "
            "before compiling, or pass its exact path directly to vai_c_xir."
        )


if __name__ == "__main__":
    main()
