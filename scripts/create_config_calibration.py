from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, validate_config
from src.data.fetch_dataset import load_clean_record
from src.fpga.deployment import calibrate_signals, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create FPGA calibration from configured clean WFDB records."
    )
    parser.add_argument("--config", default="configs/defaults.yaml", type=Path)
    parser.add_argument("--output", default="artifacts/input_calibration.json", type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    validate_config(config)
    root = args.config.resolve().parents[1]
    clean = config["data"]["clean"]
    signals = (
        load_clean_record(
            root / clean["path"] / name,
            channel=clean["channel"],
        )["signal"]
        for name in clean["records"]
    )
    calibration = calibrate_signals(signals)
    save_calibration(calibration, args.output)
    print(f"Wrote calibration to {args.output}")
    print(f"offset={calibration.offset:.9g}, scale={calibration.scale:.9g}")


if __name__ == "__main__":
    main()
