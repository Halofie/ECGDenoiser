"""Evaluate ECGResUNet1D checkpoints under board-deployable input normalization.

The published validation numbers (PRD 36.4%) were computed with the
"clean_oracle" normalization, which scales the noisy input using the CLEAN
window's min/max -- information the board never has. This script runs each
checkpoint with each normalization method, maps the output back to physical
units with the method's own affine transform, and then scores every method in
one common domain (the clean window's [-1, 1] scale, identical to the
published reports), so the numbers are directly comparable.

Usage:
    python scripts/eval_deployable_normalization.py \
        --checkpoint oracle=checkpoints/ecg_resunet1d.pt \
        --checkpoint deploy=checkpoints/ecg_resunet1d_noisy_minmax.pt \
        --methods clean_oracle noisy_minmax noisy_robust
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VITIS_DIR = ROOT / "vitis_ai_deployment"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VITIS_DIR))

from dataset import NORMALIZATIONS, ECG512Dataset, default_train_val_split  # noqa: E402
from evaluation import EvalSet, torch_model_fn  # noqa: E402
from model import ECGResUNet1D, load_checkpoint  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", action="append", required=True,
                        help="name=path; may be repeated to compare checkpoints.")
    parser.add_argument("--methods", nargs="+", default=list(NORMALIZATIONS), choices=NORMALIZATIONS)
    parser.add_argument("--clean-dir", default=str(ROOT / "data" / "clean"))
    parser.add_argument("--noise-dir", default=str(ROOT / "mit-bih-noise-stress-test-database-1.0.0"))
    parser.add_argument("--val-seed", type=int, default=1000)
    parser.add_argument("--max-windows", type=int, default=None, help="Subsample for a quick look.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "normalization_study.json")
    args = parser.parse_args()

    _, val_records = default_train_val_split()
    dataset = ECG512Dataset(args.clean_dir, args.noise_dir, records=val_records,
                            deterministic=True, seed=args.val_seed, download_missing=False)
    count = len(dataset) if args.max_windows is None else min(args.max_windows, len(dataset))
    eval_set = EvalSet(dataset, np.linspace(0, len(dataset) - 1, count, dtype=int))

    results = {"noisy_input": eval_set.noisy_baseline()}
    print(f"{count} held-out windows from records {val_records}")
    print(f"{'noisy input':<40} PRD {results['noisy_input']['prd_percent']:6.2f}%")

    for spec in args.checkpoint:
        name, _, path = spec.partition("=")
        if not path:
            name, path = Path(spec).stem, spec
        model = load_checkpoint(ECGResUNet1D(), path, map_location=args.device).to(args.device)
        model_fn = torch_model_fn(model, args.device)
        for method in args.methods:
            summary = eval_set.evaluate(model_fn, method)
            summary.update({"checkpoint": str(path), "normalization": method})
            results[f"{name}/{method}"] = summary
            bins = "  ".join(f"{k}:{v:5.1f}" for k, v in summary["prd_by_input_snr_db"].items())
            print(f"{name + '/' + method:<40} PRD {summary['prd_percent']:6.2f}%  "
                  f"SNR {summary['snr_db']:5.2f} dB  MSE {summary['mse']:.4f}  | by input SNR: {bins}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"num_windows": count, "val_records": val_records,
                                       "val_seed": args.val_seed, "results": results}, indent=2) + "\n",
                           encoding="utf-8", newline="\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
