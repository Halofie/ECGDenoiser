"""Shared CLI options, latency statistics and environment capture for the board scripts."""

from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time

import numpy as np

from . import __version__, preprocess
from .backends import create_backend


def add_backend_args(parser) -> None:
    group = parser.add_argument_group("model backend")
    group.add_argument("--backend", choices=("dpu", "torch", "identity"), default="dpu",
                       help="dpu on the ZCU104; torch/identity to rehearse on a laptop.")
    group.add_argument("--xmodel", default="ecg_denoiser.xmodel", help="Compiled xmodel (dpu backend).")
    group.add_argument("--checkpoint", default=None, help="PyTorch checkpoint (torch backend).")
    group.add_argument("--device", default="cpu", help="torch backend device.")
    group.add_argument("--io-fix-points", default=None,
                       help="torch backend only: 'IN,OUT' fix points to mimic DPU int8 I/O rounding.")
    group.add_argument("--normalization", choices=preprocess.METHODS, default=preprocess.DEFAULT_METHOD,
                       help="Must match the normalization the deployed model was trained with.")


def backend_from_args(args):
    fix_points = None
    if args.io_fix_points:
        fix_points = tuple(int(v) for v in args.io_fix_points.split(","))
    return create_backend(args.backend, xmodel=args.xmodel, checkpoint=args.checkpoint,
                          device=args.device, io_fix_points=fix_points)


def latency_stats(seconds) -> dict:
    """Summary in milliseconds."""
    values = np.asarray(list(seconds), dtype=np.float64) * 1e3
    if values.size == 0:
        return {"count": 0}
    p50, p90, p95, p99 = np.percentile(values, [50, 90, 95, 99])
    return {
        "count": int(values.size),
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "min_ms": float(values.min()),
        "p50_ms": float(p50),
        "p90_ms": float(p90),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "max_ms": float(values.max()),
    }


def _run(cmd, timeout: float = 20.0) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)
        return out.decode("utf-8", "replace")
    except Exception as error:  # tool missing / not permitted -- record why
        return "UNAVAILABLE: %r" % (error,)


def environment_report(backend=None, pipeline=None, include_tools: bool = True) -> dict:
    report = {
        "ecg_board_version": __version__,
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
    }
    try:
        import vart  # noqa: F401

        report["vart_version"] = getattr(vart, "__version__", "unknown")
    except ImportError:
        report["vart_version"] = None
    try:
        import serial

        report["pyserial"] = getattr(serial, "__version__", "unknown")
    except ImportError:
        report["pyserial"] = None
    report["serial_devices"] = sorted(
        os.path.join("/dev", d) for d in (os.listdir("/dev") if os.path.isdir("/dev") else [])
        if d.startswith(("ttyPS", "ttyUL", "ttyUSB", "ttyACM"))
    )
    if include_tools and os.path.exists("/dev"):
        report["xdputil_query"] = _run(["xdputil", "query"])
        report["uname"] = _run(["uname", "-a"])
        report["meminfo_head"] = _run(["head", "-3", "/proc/meminfo"])
        report["cpufreq"] = _run(["sh", "-c", "cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"])
    if backend is not None:
        report["backend"] = backend.info()
    if pipeline is not None:
        report["pipeline"] = pipeline.info()
    return report


def write_json(path: str, payload) -> None:
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def ensure_dir(path: str) -> str:
    if not os.path.isdir(path):
        os.makedirs(path)
    return path
