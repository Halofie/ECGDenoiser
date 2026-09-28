from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fpga.deployment import (
    load_calibration,
    quantize_int16,
    run_reference_inference,
    save_golden_vector,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a PyTorch FPGA golden vector.")
    parser.add_argument("--input", required=True, type=Path, help="Raw 256-sample .npy signal")
    parser.add_argument("--calibration", default="artifacts/input_calibration.json")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--output", default="artifacts/golden_vector.npz")
    args = parser.parse_args()

    raw = np.asarray(np.load(args.input), dtype=np.float32)
    calibration = load_calibration(args.calibration)
    input_samples = quantize_int16(raw, calibration)
    model_input, output_samples = run_reference_inference(input_samples, args.checkpoint)
    save_golden_vector(input_samples, model_input, output_samples, args.output)
    print(f"Wrote golden vector to {args.output}")


if __name__ == "__main__":
    main()
