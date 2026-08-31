from __future__ import annotations


def patient_id(record_name: str) -> str:
    """Return the patient-level identifier for a record name."""
    for token in ("118", "119"):
        if record_name.startswith(token):
            return token
    return record_name.split("e")[0] if "e" in record_name else record_name


def validate_no_patient_overlap(train_records, val_records, test_records) -> bool:
    """Return True if no patient appears across train/val/test partitions."""
    train_patients = {patient_id(r) for r in train_records}
    val_patients = {patient_id(r) for r in val_records}
    test_patients = {patient_id(r) for r in test_records}

    overlap = (train_patients & val_patients) or (train_patients & test_patients) or (val_patients & test_patients)
    return not bool(overlap)
