from __future__ import annotations

from pathlib import Path


def validate_record(record_path: str | Path) -> dict:
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

    return {
        "base_path": str(base),
        "header_path": str(hea_path),
        "data_path": str(dat_path),
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
            records.append(validate_record(path.with_suffix("")))

    return {
        "dataset_dir": str(root),
        "record_count": len(records),
        "records": records,
        "is_valid": len(records) > 0,
    }
