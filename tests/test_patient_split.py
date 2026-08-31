from src.split.patient_split import split_by_patient
from src.split.leakage_guard import validate_no_patient_overlap


def test_split_by_patient_creates_disjoint_patient_sets():
    records = [
        "118e00", "118e06", "118e12", "118e18", "118e24",
        "119e00", "119e06", "119e12", "119e18", "119e24",
    ]

    split = split_by_patient(records, val_fraction=0.2, test_fraction=0.2, seed=42)

    assert set(split["train"]) | set(split["val"]) | set(split["test"]) == set(records)
    assert set(split["train"]) & set(split["val"]) == set()
    assert set(split["train"]) & set(split["test"]) == set()
    assert set(split["val"]) & set(split["test"]) == set()


def test_split_by_patient_uses_patient_level_grouping():
    records = [
        "118e00", "118e06", "118e12",
        "119e00", "119e06", "119e12",
    ]

    split = split_by_patient(records, val_fraction=0.5, test_fraction=0.5, seed=7)

    patient_sets = {
        "train": {record.split("e")[0] for record in split["train"]},
        "val": {record.split("e")[0] for record in split["val"]},
        "test": {record.split("e")[0] for record in split["test"]},
    }

    assert patient_sets["train"] | patient_sets["val"] | patient_sets["test"] == {"118", "119"}
    assert len(patient_sets["train"] | patient_sets["val"] | patient_sets["test"]) >= 2


def test_validate_no_patient_overlap_detects_leakage():
    train = ["118e00", "118e06"]
    val = ["118e12", "119e00"]
    test = ["119e06"]

    assert validate_no_patient_overlap(train, val, test) is False
