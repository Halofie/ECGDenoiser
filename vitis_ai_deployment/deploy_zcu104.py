"""Quick board-side smoke test for the compiled ecg_denoiser.xmodel.

Thin wrapper over board/ecg_board (the same DpuBackend the UART service and
board_session.py use), kept so the commands in README.md still work:

    python3 deploy_zcu104.py --xmodel ecg_denoiser.xmodel --test-input --loop 5
    python3 deploy_zcu104.py --xmodel ecg_denoiser.xmodel \
        --golden-dir golden_vectors --golden-output-dir golden_outputs

For the full measurement flow use board/board_session.py and
board/uart_server.py instead (see BOARD_DAY_RUNBOOK.md).
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for candidate in (os.path.join(HERE, "board"), HERE):
    if os.path.isdir(os.path.join(candidate, "ecg_board")):
        sys.path.insert(0, candidate)
        break

import numpy as np  # noqa: E402

from ecg_board.backends import DpuBackend  # noqa: E402
from board_session import run_golden  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ECG denoiser DPU model on ZCU104.")
    parser.add_argument("--xmodel", default="./compiled_model/ecg_denoiser.xmodel")
    parser.add_argument("--test-input", action="store_true", help="Run synthetic windows.")
    parser.add_argument("--golden-dir", default=None, help="Directory of input_i.npy golden inputs.")
    parser.add_argument("--golden-output-dir", default="./golden_outputs")
    parser.add_argument("--loop", type=int, default=1)
    args = parser.parse_args()

    backend = DpuBackend(args.xmodel)
    info = backend.info()
    print("DPU subgraphs: %d" % info["dpu_subgraphs"])
    print("Input tensor:  %s" % info["input"])
    print("Output tensor: %s" % info["output"])

    if args.golden_dir:
        parent = os.path.dirname(os.path.abspath(args.golden_output_dir))
        summary = run_golden(backend, args.golden_dir, parent)
        produced = os.path.join(parent, "golden_outputs")
        if os.path.abspath(produced) != os.path.abspath(args.golden_output_dir):
            os.rename(produced, args.golden_output_dir)
        print("Wrote %d outputs to %s" % (summary.get("count", 0), args.golden_output_dir))
        return

    if not args.test_input:
        parser.error("Use --test-input or --golden-dir (the UART path is board/uart_server.py).")
    rng = np.random.RandomState(0)
    for i in range(args.loop):
        window = np.clip(rng.randn(512).astype(np.float32) * 0.3, -1, 1)
        clean = backend.infer(window)
        print("[%d] inference %.3f ms, output range [%.4f, %.4f]"
              % (i, backend.last_infer_s * 1e3, clean.min(), clean.max()))


if __name__ == "__main__":
    main()
