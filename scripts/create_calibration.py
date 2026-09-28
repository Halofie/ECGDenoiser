from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fpga.deployment import calibrate_signals, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create fixed FPGA input calibration from one or more .npy signals."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="1D NumPy .npy signal files")
    parser.add_argument("--output", default="artifacts/input_calibration.json")
    args = parser.parse_args()

    calibration = calibrate_signals(np.load(path) for path in args.inputs)
    save_calibration(calibration, args.output)
    print(f"Wrote calibration to {args.output}")
    print(f"offset={calibration.offset:.9g}, scale={calibration.scale:.9g}")


if __name__ == "__main__":
    main()
