from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def check_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Check ECG FPGA deployment prerequisites.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt", type=Path)
    parser.add_argument("--vivado-project", type=Path)
    args = parser.parse_args()

    requirements = {
        "checkpoint": args.checkpoint.is_file(),
        "onnx": check_module("onnx"),
        "qonnx": check_module("qonnx"),
        "hls4ml": check_module("hls4ml"),
        "pynq": check_module("pynq"),
        "pyserial": check_module("serial"),
    }
    if args.vivado_project is not None:
        requirements["vivado_project"] = args.vivado_project.is_dir()

    print("FPGA deployment prerequisites:")
    for name, present in requirements.items():
        print(f"  [{'OK' if present else 'MISSING'}] {name}")
    return 0 if all(requirements.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
