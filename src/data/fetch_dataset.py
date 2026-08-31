from __future__ import annotations

from pathlib import Path
from typing import Any


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
