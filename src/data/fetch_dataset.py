from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import wfdb


DEFAULT_CLEAN_RECORDS = [
    "100", "101", "103", "105", "106", "112", "113", "115", "117",
    "121", "122", "123", "200", "201", "202", "210", "212", "213",
    "214", "219", "220", "222", "230", "231", "233", "234",
]


def discover_records(dataset_dir: str | Path) -> list[dict[str, Any]]:
    """Discover WFDB-compatible records in a local dataset directory.

    Phase 2 only: provide a local-record discovery API for the project dataset
    already stored in the workspace. This is the first implementation step for
    dataset acquisition and validation.
    """
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records: list[dict[str, Any]] = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix == ".hea":
            name = path.stem
            records.append({"name": name, "path": str(path.parent / name), "header": str(path)})

    return records

def acquire_clean_records(
    records: list[str] | None,
    destination: str | Path,
    download_missing: bool = True,
    database: str = "mitdb",
) -> list[dict[str, Any]]:
    """Ensure selected MIT-BIH records are available locally.

    Existing local records are reused. Missing records are downloaded from
    PhysioNet when ``download_missing`` is true.
    """
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    requested = records or DEFAULT_CLEAN_RECORDS
    missing = [
        name for name in requested
        if not (destination_path / f"{name}.hea").exists()
        or not (destination_path / f"{name}.dat").exists()
    ]

    if missing and not download_missing:
        raise FileNotFoundError(f"Missing clean records: {missing}")
    failures: list[tuple[str, str]] = []
    for name in missing:
        last_error = ""
        for _attempt in range(3):
            if (destination_path / f"{name}.hea").exists() and (destination_path / f"{name}.dat").exists():
                break
            try:
                wfdb.dl_database(database, str(destination_path), records=[name])
            except Exception as exc:
                last_error = str(exc)
        if not (
            (destination_path / f"{name}.hea").exists()
            and (destination_path / f"{name}.dat").exists()
        ):
            failures.append((name, last_error))

    if failures:
        names = ", ".join(name for name, _error in failures)
        raise RuntimeError(f"Failed to download clean records: {names}") from RuntimeError(
            failures[0][1]
        )

    incomplete = [
        name for name in requested
        if not (destination_path / f"{name}.hea").exists()
        or not (destination_path / f"{name}.dat").exists()
    ]
    if incomplete:
        raise FileNotFoundError(f"Clean records remain incomplete: {incomplete}")

    return [{"name": name, "path": str(destination_path / name)} for name in requested]


def load_clean_record(record_path: str | Path, channel: str = "MLII") -> dict[str, Any]:
    """Load one clean WFDB record and return its selected physical channel."""
    base = Path(record_path)
    record = wfdb.rdrecord(str(base), physical=True)
    if record.fs != 360:
        raise ValueError(f"Expected sampling rate 360 Hz, got {record.fs}")
    try:
        channel_index = record.sig_name.index(channel)
    except ValueError as exc:
        raise ValueError(f"Channel {channel!r} not found in {base}: {record.sig_name}") from exc

    signal = np.asarray(record.p_signal[:, channel_index], dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Invalid signal data in record {base}")
    return {
        "name": base.name,
        "path": str(base),
        "signal": signal,
        "sample_rate_hz": float(record.fs),
        "channel": channel,
        "length": int(signal.size),
    }
