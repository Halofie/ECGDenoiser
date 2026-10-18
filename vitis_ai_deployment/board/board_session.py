#!/usr/bin/env python3
"""One-shot on-board measurement session (no host PC needed).

Runs, in order, and writes everything into one directory you copy back:
  1. env          board image / VART / DPU fingerprint (xdputil query) / model I/O tensors
  2. golden       DPU output for golden_vectors/input_*.npy -> golden_outputs/output_*.npy
  3. idle power   model loaded, nothing running
  4. latency      single-thread DPU-only and full pipeline (int16 -> int16), N windows
  5. throughput   K threads x one runner each, DPU saturated, power logged per K
  6. summary      session.json with latency, throughput, power and energy per window

On the ZCU104:
    python3 board_session.py --xmodel ecg_denoiser.xmodel --golden-dir golden_vectors

Rehearsal on a laptop:
    python vitis_ai_deployment/board/board_session.py --backend torch \
        --checkpoint checkpoints/ecg_resunet1d_noisy_minmax.pt --golden-dir artifacts/golden_vectors \
        --idle-seconds 3 --latency-iterations 200 --throughput-seconds 3 --threads 1,2
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import tarfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ecg_board import common  # noqa: E402
from ecg_board.pipeline import DenoisePipeline  # noqa: E402
from ecg_board.power import DEFAULT_HWMON_ROOT, PowerLogger  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_backend_args(parser)
    parser.add_argument("--out-dir", default=None, help="Default: results/session_<timestamp>.")
    parser.add_argument("--golden-dir", default="golden_vectors")
    parser.add_argument("--idle-seconds", type=float, default=60.0)
    parser.add_argument("--latency-iterations", type=int, default=2000)
    parser.add_argument("--throughput-seconds", type=float, default=30.0)
    parser.add_argument("--threads", default="1,2,4", help="Comma-separated runner/thread counts.")
    parser.add_argument("--hwmon-root", default=DEFAULT_HWMON_ROOT)
    parser.add_argument("--rail-map", default=None)
    parser.add_argument("--power-interval", type=float, default=0.1)
    parser.add_argument("--settle-seconds", type=float, default=3.0,
                        help="Pause between phases so power readings do not smear across them.")
    parser.add_argument("--no-tar", action="store_true", help="Do not pack the session into a .tar.gz.")
    return parser.parse_args(argv)


def test_windows(golden_dir: str, count: int = 16):
    """Normalized float windows: golden inputs when present, else synthetic (DPU timing is data independent)."""
    files = sorted(glob.glob(os.path.join(golden_dir, "input_*.npy")))
    if files:
        return [np.load(f).astype(np.float32).reshape(-1) for f in files]
    rng = np.random.RandomState(0)
    return [np.clip(rng.randn(512).astype(np.float32) * 0.3, -1, 1) for _ in range(count)]


def run_golden(backend, golden_dir: str, out_dir: str) -> dict:
    files = sorted(glob.glob(os.path.join(golden_dir, "input_*.npy")))
    if not files:
        return {"ran": False, "reason": "no input_*.npy in %s" % golden_dir}
    target = common.ensure_dir(os.path.join(out_dir, "golden_outputs"))
    rows = []
    for path in files:
        vector_id = os.path.basename(path)[len("input_"):-len(".npy")]
        output = backend.infer(np.load(path).astype(np.float32).reshape(-1))
        np.save(os.path.join(target, "output_%s.npy" % vector_id), output.astype(np.float32))
        rows.append({"vector_id": vector_id, "infer_ms": backend.last_infer_s * 1e3,
                     "finite": bool(np.isfinite(output).all()),
                     "range": [float(output.min()), float(output.max())]})
        print("  golden %s: %.3f ms, range [%.3f, %.3f]" % (vector_id, rows[-1]["infer_ms"], *rows[-1]["range"]))
    return {"ran": True, "count": len(rows), "vectors": rows, "output_dir": "golden_outputs"}


def run_latency(backend, pipeline, windows, iterations: int) -> dict:
    infer, full, pre, post = [], [], [], []
    raw_windows = [np.round(w * 1000.0).astype(np.int16) for w in windows]
    for i in range(iterations):
        backend.infer(windows[i % len(windows)])
        infer.append(backend.last_infer_s)
        start = time.perf_counter()
        result = pipeline.process(raw_windows[i % len(raw_windows)])
        full.append(time.perf_counter() - start)
        pre.append(result.pre_s)
        post.append(result.post_s)
    return {
        "iterations": iterations,
        "infer": common.latency_stats(infer),
        "pipeline_total": common.latency_stats(full),
        "pipeline_pre": common.latency_stats(pre),
        "pipeline_post": common.latency_stats(post),
    }


def run_throughput(args, primary_backend, windows, threads: int, seconds: float) -> dict:
    """K threads, each with its own runner, hammering inference for ``seconds``."""
    backends = [primary_backend] + [common.backend_from_args(args) for _ in range(threads - 1)]
    counts = [0] * threads
    latencies = [[] for _ in range(threads)]
    errors = []
    stop = threading.Event()
    barrier = threading.Barrier(threads + 1)

    def worker(index: int) -> None:
        backend = backends[index]
        quantize = getattr(backend, "quantize", None)
        run_int8 = getattr(backend, "run_int8", None)
        prepared = [quantize(w) for w in windows] if run_int8 else windows
        call = run_int8 if run_int8 else backend.infer
        barrier.wait()
        i = 0
        try:
            while not stop.is_set():
                call(prepared[i % len(prepared)])
                latencies[index].append(backend.last_infer_s)
                counts[index] += 1
                i += 1
        except Exception as error:  # surface in the report instead of killing the session
            errors.append(repr(error))

    workers = [threading.Thread(target=worker, args=(k,)) for k in range(threads)]
    for t in workers:
        t.start()
    barrier.wait()
    start = time.perf_counter()
    time.sleep(seconds)
    stop.set()
    for t in workers:
        t.join()
    elapsed = time.perf_counter() - start
    for extra in backends[1:]:
        extra.close()
    total = sum(counts)
    return {
        "threads": threads,
        "seconds": elapsed,
        "windows": total,
        "windows_per_s": total / elapsed if elapsed else 0.0,
        "per_call": common.latency_stats([v for lane in latencies for v in lane]),
        "errors": errors,
    }


def energy_summary(power_summary: dict, throughput: list, idle_phase: str = "idle") -> dict:
    phases = power_summary.get("phases", {}) if power_summary else {}
    idle = (phases.get(idle_phase) or {}).get("total")
    if not idle:
        return {"available": False}
    out = {"available": True, "idle_total_w": idle["mean_w"], "per_threads": []}
    for entry in throughput:
        stats = (phases.get("saturated_t%d" % entry["threads"]) or {}).get("total")
        if not stats or not entry["windows_per_s"]:
            continue
        dynamic = stats["mean_w"] - idle["mean_w"]
        out["per_threads"].append({
            "threads": entry["threads"],
            "windows_per_s": entry["windows_per_s"],
            "active_total_w": stats["mean_w"],
            "dynamic_w": dynamic,
            "energy_per_window_total_mj": stats["mean_w"] / entry["windows_per_s"] * 1e3,
            "energy_per_window_dynamic_mj": dynamic / entry["windows_per_s"] * 1e3,
        })
    return out


def main(argv=None) -> dict:
    args = parse_args(argv)
    out_dir = common.ensure_dir(args.out_dir or os.path.join("results", time.strftime("session_%Y%m%d_%H%M%S")))
    print("Session output: %s" % out_dir)

    t0 = time.time()
    backend = common.backend_from_args(args)
    load_s = time.time() - t0
    pipeline = DenoisePipeline(backend, args.normalization)
    windows = test_windows(args.golden_dir)
    for w in windows[:5]:
        backend.infer(w)  # warm-up: first DPU runs include buffer/instruction setup

    print("[1/6] environment")
    env = common.environment_report(backend, pipeline)
    env["model_load_s"] = load_s
    common.write_json(os.path.join(out_dir, "env.json"), env)
    print("  backend: %s" % env["backend"])

    print("[2/6] golden vectors")
    golden = run_golden(backend, args.golden_dir, out_dir)
    if not golden["ran"]:
        print("  WARNING: golden vectors skipped (%s) -- DPU correctness will NOT be checked." % golden["reason"])

    power = PowerLogger(os.path.join(out_dir, "power_session.csv"), args.hwmon_root,
                        args.power_interval, args.rail_map)
    if power.available:
        print("  power: %d hwmon channels -> power_session.csv" % len(power.channels))
    else:
        print("  WARNING: no hwmon power channels under %s; power phases will be empty." % args.hwmon_root)
    power.start("settle")

    print("[3/6] idle power for %.0f s" % args.idle_seconds)
    time.sleep(args.settle_seconds)
    power.set_phase("idle")
    time.sleep(args.idle_seconds)

    print("[4/6] single-thread latency x%d" % args.latency_iterations)
    power.set_phase("latency_single")
    latency = run_latency(backend, pipeline, windows, args.latency_iterations)
    print("  DPU-only mean %.3f ms (p99 %.3f) | full pipeline mean %.3f ms (p99 %.3f)"
          % (latency["infer"]["mean_ms"], latency["infer"]["p99_ms"],
             latency["pipeline_total"]["mean_ms"], latency["pipeline_total"]["p99_ms"]))

    print("[5/6] throughput")
    throughput = []
    for threads in [int(v) for v in args.threads.split(",") if v.strip()]:
        power.set_phase("settle")
        time.sleep(args.settle_seconds)
        power.set_phase("saturated_t%d" % threads)
        entry = run_throughput(args, backend, windows, threads, args.throughput_seconds)
        throughput.append(entry)
        print("  threads=%d: %.1f windows/s, per-call mean %.3f ms%s"
              % (threads, entry["windows_per_s"], entry["per_call"].get("mean_ms", float("nan")),
                 " ERRORS: %s" % entry["errors"][:2] if entry["errors"] else ""))
    power.set_phase("settle")
    time.sleep(min(args.settle_seconds, 1.0))
    power.stop()

    print("[6/6] summary")
    power_summary = power.summary()
    session = {
        "env": {k: env[k] for k in ("hostname", "platform", "python", "numpy", "vart_version", "time_utc")},
        "backend": env["backend"],
        "normalization": args.normalization,
        "golden": golden,
        "latency": latency,
        "throughput": throughput,
        "power": power_summary,
        "energy": energy_summary(power_summary, throughput),
        "config": vars(args),
    }
    common.write_json(os.path.join(out_dir, "session.json"), session)
    for row in session["energy"].get("per_threads", []):
        print("  threads=%d: %.3f W total, %.3f W dynamic, %.3f mJ/window total, %.3f mJ/window dynamic"
              % (row["threads"], row["active_total_w"], row["dynamic_w"],
                 row["energy_per_window_total_mj"], row["energy_per_window_dynamic_mj"]))

    if not args.no_tar:
        archive = out_dir.rstrip("/\\") + ".tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(out_dir, arcname=os.path.basename(out_dir.rstrip("/\\")))
        print("Packed %s -- copy it back to the laptop." % archive)
    print("Done: %s" % os.path.join(out_dir, "session.json"))
    return session


if __name__ == "__main__":
    main()
