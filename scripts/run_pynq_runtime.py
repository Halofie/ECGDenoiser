from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.fpga.runtime import serve_uart


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ECG denoiser PYNQ UART service.")
    parser.add_argument("--bitstream", required=True, type=Path)
    parser.add_argument("--serial-device", required=True)
    parser.add_argument("--baud-rate", type=int, default=115200)
    parser.add_argument("--dma-name", default="axi_dma")
    args = parser.parse_args()

    try:
        from pynq import Overlay
        import serial
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The board runtime requires PYNQ and pyserial. "
            "Run this command on the ZCU104 PYNQ image."
        ) from error

    overlay = Overlay(str(args.bitstream))
    if not hasattr(overlay, args.dma_name):
        raise RuntimeError(f"Overlay does not expose DMA object {args.dma_name!r}")
    dma = getattr(overlay, args.dma_name)
    with serial.Serial(args.serial_device, args.baud_rate, timeout=2.0) as port:
        serve_uart(dma, port.read, port.write)


if __name__ == "__main__":
    main()
