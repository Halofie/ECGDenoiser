#!/usr/bin/env python3
"""Inspect and log ZCU104 power rails exposed through Linux hwmon.

    python3 power_monitor.py list                      # what sensors exist + live readings
    python3 power_monitor.py list --write-map rails.json   # template rail map to edit
    python3 power_monitor.py log --seconds 60 --phase idle --out power_idle.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecg_board import power  # noqa: E402


def cmd_list(args) -> int:
    channels = power.discover_channels(args.hwmon_root)
    if not channels:
        print("No power*_input files under %s." % args.hwmon_root)
        print("Check: ls %s/*/name ; dmesg | grep -i -e ina2 -e pmbus -e irps" % args.hwmon_root)
        print("If the image has no hwmon power drivers, use an external meter on the 12 V input.")
        return 1
    print("%-40s %-12s %-10s %10s" % ("key", "label", "driver", "watts"))
    for channel in channels:
        watts = channel.read_watts()
        print("%-40s %-12s %-10s %10s" % (channel.key, channel.label[:12], channel.driver,
                                         "?" if watts is None else "%.4f" % watts))
    if args.write_map:
        template = {
            "names": {c.key: (c.label or c.key) for c in channels},
            "total": [c.key for c in channels],
            "ignore": [],
            "_comment": "Rename rails (see the ZCU104 schematic/UG1267) and keep in 'total' only channels "
                        "that do not feed each other, so the total does not double count.",
        }
        with open(args.write_map, "w") as handle:
            json.dump(template, handle, indent=2, sort_keys=True)
        print("Wrote rail map template to %s" % args.write_map)
    return 0


def cmd_log(args) -> int:
    logger = power.PowerLogger(args.out, args.hwmon_root, args.interval, args.rail_map)
    if not logger.available:
        print("No power channels found; nothing to log.")
        return 1
    logger.start(args.phase)
    try:
        time.sleep(args.seconds)
    except KeyboardInterrupt:
        pass
    logger.stop()
    stats = logger.summary()["phases"].get(args.phase, {})
    for name, values in sorted(stats.items()):
        if name.startswith("_") or not values:
            continue
        print("%-40s mean %.4f W  std %.4f  n=%d" % (name, values["mean_w"], values["std_w"], values["samples"]))
    print("CSV: %s" % args.out)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hwmon-root", default=power.DEFAULT_HWMON_ROOT)
    parser.add_argument("--rail-map", default=None)
    sub = parser.add_subparsers(dest="command")
    p_list = sub.add_parser("list")
    p_list.add_argument("--write-map", default=None)
    p_log = sub.add_parser("log")
    p_log.add_argument("--seconds", type=float, default=60.0)
    p_log.add_argument("--phase", default="manual")
    p_log.add_argument("--interval", type=float, default=0.1)
    p_log.add_argument("--out", default="power_log.csv")
    args = parser.parse_args(argv)
    if args.command == "log":
        return cmd_log(args)
    if args.command == "list":
        return cmd_list(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
