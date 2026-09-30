"""Compare DPU outputs (copied back from the board) against the float
model's expected outputs, and save a persistent, timestamped test report.

Run this on the laptop, after:
  1. generate_golden_vectors.py produced golden_vectors/input_i.npy +
     expected_i.npy here on the laptop.
  2. Those input_i.npy files (only the inputs, not expected_i.npy) were
     copied to the board and run through deploy_zcu104.py --golden-dir,
     which saved output_i.npy for each into a golden_outputs/ directory
     on the board.
  3. That golden_outputs/ directory was copied back to the laptop.

This script never touches the board or Vitis AI tools -- just numpy, so it
runs in the normal project Python environment.

Usage:
    python compare_golden_results.py \
        --golden-dir ../artifacts/golden_vectors \
        --dpu-output-dir ../artifacts/golden_outputs_from_board \
        --report-dir ../reports/dpu_validation
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare board-produced DPU outputs against expected float model outputs."
    )
    parser.add_argument("--golden-dir", default="../artifacts/golden_vectors", type=Path,
                         help="Directory with input_i.npy/expected_i.npy (from generate_golden_vectors.py).")
    parser.add_argument("--dpu-output-dir", required=True, type=Path,
                         help="Directory with output_i.npy files copied back from the board.")
    parser.add_argument("--report-dir", default="../reports/dpu_validation", type=Path,
                         help="Where to save the timestamped comparison report.")
    parser.add_argument("--mse-warn-threshold", type=float, default=0.05,
                         help="Flag a vector if its MSE against the float model exceeds this.")
    args = parser.parse_args()

    expected_files = sorted(args.golden_dir.glob("expected_*.npy"))
    if not expected_files:
        raise FileNotFoundError(f"No expected_*.npy files found under {args.golden_dir}")

    results = []
    for expected_file in expected_files:
        vector_id = expected_file.stem.split("_", 1)[1]
        output_file = args.dpu_output_dir / f"output_{vector_id}.npy"
        if not output_file.is_file():
            print(f"WARNING: missing {output_file} for vector {vector_id}, skipping", file=sys.stderr)
            continue

        expected = np.load(expected_file).astype(np.float32)
        dpu_output = np.load(output_file).astype(np.float32)

        if dpu_output.shape != expected.shape:
            results.append({
                "vector_id": vector_id,
                "status": "SHAPE_MISMATCH",
                "dpu_shape": list(dpu_output.shape),
                "expected_shape": list(expected.shape),
            })
            print(f"[{vector_id}] SHAPE MISMATCH: dpu={dpu_output.shape} expected={expected.shape}")
            continue

        if not np.isfinite(dpu_output).all():
            results.append({"vector_id": vector_id, "status": "NON_FINITE_OUTPUT"})
            print(f"[{vector_id}] NON-FINITE VALUES in DPU output -- real bug, not quantization noise")
            continue

        error = dpu_output - expected
        mse = float(np.mean(np.square(error)))
        max_abs = float(np.max(np.abs(error)))
        status = "OK" if mse <= args.mse_warn_threshold else "HIGH_ERROR"

        results.append({
            "vector_id": vector_id,
            "status": status,
            "mse": mse,
            "max_abs_diff": max_abs,
            "dpu_range": [float(dpu_output.min()), float(dpu_output.max())],
            "expected_range": [float(expected.min()), float(expected.max())],
        })
        print(f"[{vector_id}] {status} mse={mse:.6f} max_abs_diff={max_abs:.4f} "
              f"dpu_range=[{dpu_output.min():.4f}, {dpu_output.max():.4f}] "
              f"expected_range=[{expected.min():.4f}, {expected.max():.4f}]")

    ok_mse = [r["mse"] for r in results if r.get("status") in ("OK", "HIGH_ERROR")]
    summary = {
        "num_vectors_compared": len(results),
        "num_ok": sum(1 for r in results if r["status"] == "OK"),
        "num_high_error": sum(1 for r in results if r["status"] == "HIGH_ERROR"),
        "num_shape_mismatch": sum(1 for r in results if r["status"] == "SHAPE_MISMATCH"),
        "num_non_finite": sum(1 for r in results if r["status"] == "NON_FINITE_OUTPUT"),
        "mean_mse": float(np.mean(ok_mse)) if ok_mse else None,
    }
    print(f"\nSummary: {summary}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_path = args.report_dir / f"dpu_validation_{timestamp}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "timestamp_utc": timestamp,
                "golden_dir": str(args.golden_dir),
                "dpu_output_dir": str(args.dpu_output_dir),
                "mse_warn_threshold": args.mse_warn_threshold,
                "summary": summary,
                "vectors": results,
            },
            handle,
            indent=2,
        )
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
