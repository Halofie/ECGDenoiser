"""Dry-run the whole board-day flow on the laptop, end to end.

Same scripts, same protocol, same report -- only the DPU is replaced by the
float PyTorch model (--backend torch), the UART by TCP on localhost, and the
board's hwmon power sensors by a fake sysfs tree. If this passes, board day
is only about the board.

    python scripts/rehearse_board_day.py --checkpoint checkpoints/ecg_resunet1d_noisy_minmax.pt

Output: results/rehearsal_<timestamp>/ with every artifact the real run
produces, plus hardware_results_SIMULATED.md.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOARD_DIR = ROOT / "vitis_ai_deployment" / "board"
sys.path.insert(0, str(BOARD_DIR))
sys.path.insert(0, str(ROOT / "scripts"))

import board_session  # noqa: E402
import uart_server  # noqa: E402
import uart_host_client  # noqa: E402


def fake_hwmon(root: Path) -> Path:
    """Two rails with plausible ZCU104-scale numbers. Static: the rehearsal checks plumbing, not physics."""
    for index, (driver, watts) in enumerate((("ina226", 3.2), ("ina226", 1.1))):
        node = root / f"hwmon{index}"
        node.mkdir(parents=True, exist_ok=True)
        (node / "name").write_text(driver + "\n")
        (node / "power1_input").write_text(f"{int(watts * 1e6)}\n")
    return root


def serve_in_thread(argv: list[str]) -> tuple[threading.Thread, int]:
    ready = threading.Event()
    port = {}

    def on_ready(bound_port):
        port["value"] = bound_port
        ready.set()

    thread = threading.Thread(target=uart_server.main, args=(argv, on_ready), daemon=True)
    thread.start()
    if not ready.wait(120):
        raise RuntimeError("uart_server did not start")
    return thread, port["value"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", default=str(ROOT / "checkpoints" / "ecg_resunet1d_noisy_minmax.pt"))
    parser.add_argument("--windows", type=int, default=200)
    parser.add_argument("--seconds", type=float, default=30.0, help="continuous-mode stream length")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = (args.out or ROOT / "results" / time.strftime("rehearsal_%Y%m%d_%H%M%S")).resolve()
    args.checkpoint = str(Path(args.checkpoint).resolve())  # subprocesses run from other directories
    out.mkdir(parents=True, exist_ok=True)
    hwmon = fake_hwmon(out / "fake_hwmon")
    backend = ["--backend", "torch", "--checkpoint", args.checkpoint]
    golden_dir = str(out / "golden_vectors")

    print("\n######## 0. golden vectors from the checkpoint under test")
    subprocess.check_call([sys.executable, "generate_golden_vectors.py", "--checkpoint", args.checkpoint,
                           "--output-dir", golden_dir, "--clean-dir", str(ROOT / "data" / "clean"),
                           "--noise-dir", str(ROOT / "mit-bih-noise-stress-test-database-1.0.0")],
                          cwd=ROOT / "vitis_ai_deployment")

    print("\n######## 1. offline board session")
    board_session.main(backend + ["--golden-dir", golden_dir, "--out-dir", str(out / "session"),
                                  "--hwmon-root", str(hwmon), "--idle-seconds", "3", "--settle-seconds", "0.5",
                                  "--latency-iterations", "300", "--throughput-seconds", "3", "--threads", "1,2"])

    print("\n######## 2. golden comparison (laptop side)")
    subprocess.check_call([sys.executable, str(ROOT / "vitis_ai_deployment" / "compare_golden_results.py"),
                           "--golden-dir", golden_dir, "--dpu-output-dir", str(out / "session" / "golden_outputs"),
                           "--report-dir", str(out / "dpu_validation")])
    golden_report = sorted((out / "dpu_validation").glob("*.json"))[-1]

    print("\n######## 3. UART service + host client, held-out windows")
    server_args = backend + ["--tcp-port", "0", "--tcp-host", "127.0.0.1", "--power", "--hwmon-root", str(hwmon),
                             "--warmup", "5"]
    thread, port = serve_in_thread(server_args + ["--out-dir", str(out / "server_windows")])
    uart_host_client.main(["--port", f"tcp://127.0.0.1:{port}", "--num-windows", str(args.windows),
                           "--reference-checkpoint", args.checkpoint, "--out-dir", str(out / "host_windows")])
    thread.join(30)

    print("\n######## 4. UART service + host client, continuous stream")
    thread, port = serve_in_thread(server_args + ["--out-dir", str(out / "server_continuous")])
    uart_host_client.main(["--port", f"tcp://127.0.0.1:{port}", "--mode", "continuous", "--record", "231",
                           "--seconds", str(args.seconds), "--reference-checkpoint", args.checkpoint,
                           "--out-dir", str(out / "host_continuous")])
    thread.join(30)

    print("\n######## 5. report")
    subprocess.check_call([sys.executable, str(ROOT / "scripts" / "build_hardware_report.py"),
                           "--session", str(out / "session"), "--server", str(out / "server_windows"),
                           "--host", str(out / "host_windows"), "--host-continuous", str(out / "host_continuous"),
                           "--golden-report", str(golden_report), "--out", str(out / "hardware_results_SIMULATED.md"),
                           "--title", "REHEARSAL (laptop, simulated board)"])
    print(f"\nRehearsal complete: {out}")


if __name__ == "__main__":
    main()
