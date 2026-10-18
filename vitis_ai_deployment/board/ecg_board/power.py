"""Board power sampling through Linux hwmon (INA226 / PMBus regulators).

The ZCU104's on-board current monitors and PMBus regulators appear on the
PetaLinux image as /sys/class/hwmon/hwmon*/ devices when their drivers are
enabled (ina2xx exposes power1_input; pmbus drivers expose power*_input with
power*_label). Which rails exist depends on the board image's device tree,
so nothing here is hard-coded: every ``power*_input`` file is discovered and
logged, and an optional rail map JSON gives them readable names and chooses
which ones are summed into the "total". Values in sysfs are microwatts.

Rail map JSON (optional, create on board day from `power_monitor.py list`):
    {
      "names":   {"ina226@3-0040/power1": "VCCINT", ...},
      "total":   ["ina226@3-0040/power1", ...],      # channels summed as total
      "ignore":  ["..."]                              # channels never logged
    }
Without a map, the total is the sum of all channels. That can double count
when one monitor measures a rail that feeds another, so check the topology
before quoting a single "board power" number.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import threading
import time

DEFAULT_HWMON_ROOT = "/sys/class/hwmon"
_POWER_INPUT = re.compile(r"^power(\d+)_input$")


def _read_text(path: str):
    try:
        with open(path) as handle:
            return handle.read().strip()
    except (IOError, OSError):
        return None


class PowerChannel(object):
    def __init__(self, key: str, path: str, label: str, hwmon: str, driver: str, device: str):
        self.key = key
        self.path = path
        self.label = label
        self.hwmon = hwmon
        self.driver = driver
        self.device = device
        self.name = key

    def read_watts(self):
        raw = _read_text(self.path)
        try:
            return int(raw) / 1e6 if raw is not None else None
        except ValueError:
            return None

    def describe(self) -> dict:
        return {"key": self.key, "name": self.name, "label": self.label, "hwmon": self.hwmon,
                "driver": self.driver, "device": self.device, "path": self.path}


def discover_channels(root: str = DEFAULT_HWMON_ROOT):
    channels = []
    for hwmon_dir in sorted(glob.glob(os.path.join(root, "hwmon*")),
                            key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0)):
        hwmon = os.path.basename(hwmon_dir)
        driver = _read_text(os.path.join(hwmon_dir, "name")) or "unknown"
        device_link = os.path.join(hwmon_dir, "device")
        device = os.path.basename(os.path.realpath(device_link)) if os.path.exists(device_link) else hwmon
        for entry in sorted(os.listdir(hwmon_dir)):
            match = _POWER_INPUT.match(entry)
            if not match:
                continue
            index = match.group(1)
            label = _read_text(os.path.join(hwmon_dir, "power%s_label" % index)) or ""
            key = "%s@%s/power%s" % (driver, device, index)
            channels.append(PowerChannel(key, os.path.join(hwmon_dir, entry), label, hwmon, driver, device))
    return channels


def load_rail_map(path: str = None) -> dict:
    if not path:
        return {"names": {}, "total": None, "ignore": []}
    with open(path) as handle:
        data = json.load(handle)
    return {"names": data.get("names", {}), "total": data.get("total"), "ignore": data.get("ignore", [])}


def apply_rail_map(channels, rail_map: dict):
    kept = [c for c in channels if c.key not in rail_map["ignore"]]
    for channel in kept:
        channel.name = rail_map["names"].get(channel.key) or (
            "%s:%s" % (channel.key, channel.label) if channel.label else channel.key)
    total_keys = rail_map["total"] if rail_map["total"] is not None else [c.key for c in kept]
    return kept, set(total_keys)


def _stats(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {"mean_w": mean, "std_w": var ** 0.5, "min_w": values[0], "max_w": values[-1],
            "median_w": (values[(n - 1) // 2] + values[n // 2]) / 2.0, "samples": n}


class PowerLogger(object):
    """Background sampler. ``set_phase(label)`` tags subsequent samples; ``summary()`` aggregates per phase."""

    def __init__(self, csv_path: str, root: str = DEFAULT_HWMON_ROOT, interval_s: float = 0.1,
                 rail_map_path: str = None):
        channels = discover_channels(root)
        self.channels, self.total_keys = apply_rail_map(channels, load_rail_map(rail_map_path))
        self.interval_s = interval_s
        self.csv_path = csv_path
        self.phase = "start"
        self._samples = {}  # phase -> {channel name or "total": [watts]}
        self._phase_times = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    @property
    def available(self) -> bool:
        return bool(self.channels)

    def set_phase(self, label: str) -> None:
        with self._lock:
            self.phase = label
            self._phase_times.setdefault(label, [time.time(), None])

    def _sample_once(self, writer) -> None:
        now = time.time()
        readings = [(c, c.read_watts()) for c in self.channels]
        total = sum(w for c, w in readings if w is not None and c.key in self.total_keys)
        with self._lock:
            phase = self.phase
            bucket = self._samples.setdefault(phase, {})
            for channel, watts in readings:
                if watts is not None:
                    bucket.setdefault(channel.name, []).append(watts)
            bucket.setdefault("total", []).append(total)
            if phase in self._phase_times:
                self._phase_times[phase][1] = now
        writer.writerow(["%.4f" % now, phase] + ["" if w is None else "%.6f" % w for _, w in readings]
                        + ["%.6f" % total])

    def _run(self) -> None:
        with open(self.csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t_wall", "phase"] + [c.name for c in self.channels] + ["total_w"])
            next_tick = time.monotonic()
            while not self._stop.is_set():
                self._sample_once(writer)
                next_tick += self.interval_s
                delay = next_tick - time.monotonic()
                if delay > 0:
                    self._stop.wait(delay)
                else:
                    next_tick = time.monotonic()
                handle.flush()

    def start(self, phase: str = "start"):
        self.set_phase(phase)
        if self.available:
            self._thread = threading.Thread(target=self._run, name="power-logger")
            self._thread.daemon = True
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def summary(self) -> dict:
        with self._lock:
            phases = {}
            for phase, bucket in self._samples.items():
                phases[phase] = {name: _stats(values) for name, values in bucket.items()}
                times = self._phase_times.get(phase)
                if times:
                    phases[phase]["_window"] = {"start": times[0], "end": times[1]}
        return {
            "available": self.available,
            "interval_s": self.interval_s,
            "channels": [c.describe() for c in self.channels],
            "total_channels": sorted(self.total_keys),
            "phases": phases,
        }

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def snapshot(root: str = DEFAULT_HWMON_ROOT, rail_map_path: str = None):
    """One reading of every channel: list of (channel, watts)."""
    channels, _total = apply_rail_map(discover_channels(root), load_rail_map(rail_map_path))
    return [(c, c.read_watts()) for c in channels]
