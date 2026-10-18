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
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from model import INPUT_LENGTH, ECGResUNet1D, load_checkpoint, require_normalization
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


def evaluate_quantized(args, float_model: torch.nn.Module, quant_model: torch.nn.Module) -> dict:
    """Float vs INT8 (fake-quant) quality on held-out windows, scored like the board runs.

    Batch size 1 throughout: vai_q_pytorch test mode expects batch 1 before
    export_xmodel(). If INT8 is much worse than float here, fix it (QAT,
    more/better calibration) before compiling -- the board cannot do better.
    """
    from dataset import ECG512Dataset, default_train_val_split
    from evaluation import EvalSet, torch_model_fn

    if args.normalization == "clean_oracle":
        print("Skipping --evaluate: clean_oracle normalization is not deployable.")
        return {}
    _, val_records = default_train_val_split()
    dataset = ECG512Dataset(args.clean_dir, args.noise_dir, records=val_records, deterministic=True,
                            seed=1000, download_missing=args.download_missing,
                            normalization=args.normalization)
    count = min(args.evaluate, len(dataset))
    eval_set = EvalSet(dataset, np.linspace(0, len(dataset) - 1, count, dtype=int))
    results = {
        "num_windows": count,
        "normalization": args.normalization,
        "noisy_input": eval_set.noisy_baseline(),
        "float": eval_set.evaluate(torch_model_fn(float_model, args.device, batch_size=1), args.normalization),
        "int8_fake_quant": eval_set.evaluate(torch_model_fn(quant_model, args.device, batch_size=1),
                                             args.normalization),
    }
    for name in ("noisy_input", "float", "int8_fake_quant"):
        r = results[name]
        print(f"[evaluate] {name:<16} PRD {r['prd_percent']:6.2f}%  SNR {r['snr_db']:5.2f} dB  MSE {r['mse']:.4f}")
    results["int8_minus_float_prd_points"] = results["int8_fake_quant"]["prd_percent"] - results["float"]["prd_percent"]
    with open(args.output_dir / "quant_eval.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Vitis AI PTQ for ECGResUNet1D.")
    parser.add_argument("--mode", choices=("calib", "test"), required=True)
    parser.add_argument("--checkpoint", default="checkpoints/ecg_resunet1d.pt", type=Path)
    parser.add_argument("--output-dir", default="artifacts/quantize_result", type=Path)
    parser.add_argument("--calib-batches", type=int, default=100)
    parser.add_argument(
        "--evaluate",
        type=int,
        default=500,
        help="test mode: score float vs fake-quantized INT8 on this many held-out windows through the "
        "board's exact pre/post-processing (0 to skip). Writes quant_eval.json into --output-dir.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean-dir", default=None, help="Local MIT-BIH clean record directory (e.g. data/clean).")
    parser.add_argument("--noise-dir", default=None, help="Local nstdb noise record directory.")
    parser.add_argument("--records", default=None, help="Comma-separated MIT-BIH record names (default: full list).")
    parser.add_argument("--val-records", default=None, help="Comma-separated record names held out for validation.")
    parser.add_argument("--download-missing", action="store_true", default=True)
    parser.add_argument(
        "--normalization",
        default="noisy_minmax",
        choices=("clean_oracle", "noisy_minmax", "noisy_robust"),
        help="Window scaling. Must match what the board runtime does (ecg_board.preprocess); "
        "clean_oracle uses the clean signal and is NOT deployable.",
    )
    args = parser.parse_args()

    try:
        from pytorch_nndct.apis import torch_quantizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Quantization requires the Vitis AI PyTorch quantizer "
            "(pytorch_nndct). Run inside the vitis-ai-pytorch-cpu/gpu "
            "Docker image, not the project's plain training venv."
        ) from error

    require_normalization(args.checkpoint, args.normalization)
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
        if args.evaluate > 0:
            evaluate_quantized(args, model, quant_model)
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
