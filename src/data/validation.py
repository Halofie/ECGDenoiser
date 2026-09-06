from __future__ import annotations

from pathlib import Path

import numpy as np
import wfdb


def validate_record(
    record_path: str | Path,
    expected_sample_rate_hz: float = 360.0,
    channel: str | None = None,
) -> dict:
    """Validate a record base path by checking the required WFDB files exist.

    This Phase 2 implementation is intentionally conservative and only validates
    that a record has the expected .hea and .dat files present.
    """
    base = Path(record_path)
    hea_path = base.with_suffix(".hea")
    dat_path = base.with_suffix(".dat")

    if not hea_path.exists():
        raise FileNotFoundError(f"Missing header file for record: {hea_path}")
    if not dat_path.exists():
        raise FileNotFoundError(f"Missing data file for record: {dat_path}")

    header = wfdb.rdheader(str(base))
    if header.fs != expected_sample_rate_hz:
        raise ValueError(f"Expected sampling rate {expected_sample_rate_hz} Hz, got {header.fs}")
    channel_index = 0 if channel is None else header.sig_name.index(channel)

    record = wfdb.rdrecord(str(base), physical=True, channels=[channel_index])
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float32)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Invalid signal values in record: {base}")

    return {
        "base_path": str(base),
        "header_path": str(hea_path),
        "data_path": str(dat_path),
        "sample_rate_hz": float(header.fs),
        "channel": header.sig_name[channel_index],
        "length": int(signal.size),
        "is_valid": True,
    }


def validate_dataset_dir(dataset_dir: str | Path) -> dict:
    """Validate a dataset directory by checking the discovered record files."""
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    records = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix == ".hea":
            records.append(validate_record(path.with_suffix(""), channel=None))

    return {
        "dataset_dir": str(root),
        "record_count": len(records),
        "records": records,
        "is_valid": len(records) > 0,
    }
