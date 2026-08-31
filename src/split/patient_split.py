from __future__ import annotations

import random
from typing import Iterable


def _patient_id(record_name: str) -> str:
    """Infer a patient-level ID from a physical record name.

    Example: 118e00 -> 118, 119e06 -> 119
    """
    for token in ("118", "119"):
        if record_name.startswith(token):
            return token
    return record_name.split("e")[0] if "e" in record_name else record_name


def split_by_patient(
    records: Iterable[str],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int | None = None,
) -> dict[str, list[str]]:
    """Split records by patient-level groups while preserving disjoint partitions.

    Phase 5 requirement: no patient should overlap between train/validation/test
    partitions. This is a foundation-only implementation that does not yet build
    the final dataset tensors; it makes the split contract explicit and testable.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1)")

    unique_records = sorted(set(records))
    patient_groups: dict[str, list[str]] = {}
    for record in unique_records:
        patient_groups.setdefault(_patient_id(record), []).append(record)

    patient_names = sorted(patient_groups)
    rng = random.Random(seed)
    rng.shuffle(patient_names)

    total_patients = len(patient_names)
    val_count = max(1, round(total_patients * val_fraction)) if total_patients > 1 else 0
    test_count = max(1, round(total_patients * test_fraction)) if total_patients > 1 else 0

    # Keep a conservative guard so the splits remain non-overlapping and complete.
    val_count = min(val_count, max(0, total_patients - 1))
    test_count = min(test_count, max(0, total_patients - val_count - 1))

    remaining = total_patients - val_count - test_count
    train_names = patient_names[:remaining] if remaining > 0 else []
    val_names = patient_names[remaining : remaining + val_count] if val_count > 0 else []
    test_names = patient_names[remaining + val_count : remaining + val_count + test_count] if test_count > 0 else []

    train_records = sorted(record for patient in train_names for record in patient_groups[patient])
    val_records = sorted(record for patient in val_names for record in patient_groups[patient])
    test_records = sorted(record for patient in test_names for record in patient_groups[patient])

    return {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
