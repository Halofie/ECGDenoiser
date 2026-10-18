#!/usr/bin/env python3
"""ECG denoiser service: receive noisy windows over UART (or TCP), reply with denoised ones.

On the ZCU104:
    python3 uart_server.py --xmodel ecg_denoiser.xmodel --serial /dev/ttyPS1 --baud 115200 --power

Rehearsal on a laptop (no board), with the float model standing in for the DPU:
    python vitis_ai_deployment/board/uart_server.py --backend torch \
        --checkpoint checkpoints/ecg_resunet1d_noisy_minmax.pt --tcp-port 5555

Writes into --out-dir: env.json, server_timings.csv (one row per window),
server_summary.json (latency stats, link stats) and, with --power,
power_uart.csv plus per-phase power stats in server_summary.json.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from ecg_board import common, protocol, transport  # noqa: E402
from ecg_board.pipeline import DenoisePipeline  # noqa: E402
from ecg_board.power import DEFAULT_HWMON_ROOT, PowerLogger  # noqa: E402
from ecg_board.server import DenoiseServer  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    common.add_backend_args(parser)
    link = parser.add_argument_group("link")
    link.add_argument("--serial", default=None, help="Serial device, e.g. /dev/ttyPS1.")
    link.add_argument("--baud", type=int, default=115200)
    link.add_argument("--tcp-port", type=int, default=None, help="Listen on TCP instead of a serial port.")
    link.add_argument("--tcp-host", default="0.0.0.0")
    link.add_argument("--timeout", type=float, default=1.0, help="Read timeout per poll (s).")
    run = parser.add_argument_group("run control")
    run.add_argument("--out-dir", default=None, help="Default: results/uart_<timestamp>.")
    run.add_argument("--warmup", type=int, default=20, help="Inferences to run before serving.")
    run.add_argument("--max-requests", type=int, default=None)
    run.add_argument("--idle-timeout", type=float, default=None,
                     help="Stop after this many seconds without traffic (default: never).")
    pwr = parser.add_argument_group("power")
    pwr.add_argument("--power", action="store_true", help="Log hwmon power while serving.")
    pwr.add_argument("--hwmon-root", default=DEFAULT_HWMON_ROOT)
    pwr.add_argument("--rail-map", default=None)
    pwr.add_argument("--power-interval", type=float, default=0.1)
    args = parser.parse_args(argv)
    if (args.serial is None) == (args.tcp_port is None):
        parser.error("Choose exactly one of --serial or --tcp-port.")
    return args


def main(argv=None, ready=None) -> dict:
    args = parse_args(argv)
    out_dir = common.ensure_dir(args.out_dir or os.path.join("results", time.strftime("uart_%Y%m%d_%H%M%S")))

    backend = common.backend_from_args(args)
    pipeline = DenoisePipeline(backend, args.normalization)
    warm = [pipeline.process((np.random.RandomState(i).randn(512) * 500).astype(np.int16)).infer_s
            for i in range(args.warmup)]
    env = common.environment_report(backend, pipeline)
    env["warmup_infer"] = common.latency_stats(warm)
    common.write_json(os.path.join(out_dir, "env.json"), env)

    info = {
        "protocol_version": protocol.VERSION,
        "window_length": protocol.WINDOW_LENGTH,
        "backend": backend.info(),
        "normalization": args.normalization,
        "hostname": env["hostname"],
        "python": env["python"],
        "numpy": env["numpy"],
        "ecg_board_version": env["ecg_board_version"],
    }

    power = None
    if args.power:
        power = PowerLogger(os.path.join(out_dir, "power_uart.csv"), args.hwmon_root,
                            args.power_interval, args.rail_map)
        if not power.available:
            print("WARNING: no hwmon power channels under %s; power not logged." % args.hwmon_root)
        power.start("uart_session")

    if args.tcp_port is not None:
        print("Waiting for a TCP client on %s:%d ..." % (args.tcp_host, args.tcp_port))
        link = transport.listen_tcp(args.tcp_port, args.tcp_host, args.timeout, ready=ready)
    else:
        link = transport.open_serial(args.serial, args.baud, args.timeout)
        link.reset_input()
        if ready is not None:
            ready(None)
    info["link"] = link.description
    print("READY on %s | backend=%s | normalization=%s | out=%s"
          % (link.description, backend.name, args.normalization, out_dir))
    sys.stdout.flush()

    server = DenoiseServer(link, pipeline, info, timing_csv=os.path.join(out_dir, "server_timings.csv"))
    try:
        stats = server.serve(max_requests=args.max_requests, idle_timeout=args.idle_timeout)
    except KeyboardInterrupt:
        stats = server.stats
        print("Interrupted.")
    finally:
        link.close()
        if power is not None:
            power.stop()

    summary = {
        "link": link.description,
        "baud": args.baud if args.serial else None,
        "stats": stats,
        "latency": {key: common.latency_stats(values) for key, values in server.latencies.items()},
        "power": power.summary() if power is not None else None,
    }
    common.write_json(os.path.join(out_dir, "server_summary.json"), summary)
    lat = summary["latency"]
    if lat["infer_s"].get("count"):
        print("Served %d windows | infer mean %.3f ms p99 %.3f ms | board mean %.3f ms"
              % (stats["requests"], lat["infer_s"]["mean_ms"], lat["infer_s"]["p99_ms"], lat["board_s"]["mean_ms"]))
    print("CRC errors %d, discarded bytes %d, errors sent %d. Results in %s"
          % (stats["crc_errors"], stats["discarded_bytes"], stats["errors_sent"], out_dir))
    return summary


if __name__ == "__main__":
    main()
