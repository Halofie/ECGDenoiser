from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.split.patient_split import split_by_patient


def create_split_manifest(
    records: Iterable[str],
    output_path: str | Path,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Create and persist a reproducible source-record split manifest."""
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    if any(value < 0.0 for value in fractions.values()):
        raise ValueError("Split fractions must be non-negative")
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("Split fractions must sum to 1.0")

    split = split_by_patient(
        records,
        val_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    manifest = {
        "version": 1,
        "unit": "source_record",
        "seed": seed,
        "fractions": fractions,
        "splits": {
            "train": split["train"],
            "validation": split["val"],
            "test": split["test"],
        },
    }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_split_manifest(path: str | Path) -> dict:
    """Load a persisted split manifest."""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if set(manifest.get("splits", {})) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation, and test splits")
    return manifest
