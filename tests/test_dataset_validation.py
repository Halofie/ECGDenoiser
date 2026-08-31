from pathlib import Path

import pytest

from src.data.fetch_dataset import discover_records
from src.data.validation import validate_record, validate_dataset_dir


DATASET_DIR = Path(__file__).resolve().parents[1] / "mit-bih-noise-stress-test-database-1.0.0"


def test_discover_records_find_expected_local_records():
    records = discover_records(DATASET_DIR)

    names = {record["name"] for record in records}
    assert "118e00" in names
    assert "119e00" in names
    assert "bw" in names
    assert "em" in names


def test_validate_dataset_dir_accepts_project_dataset():
    result = validate_dataset_dir(DATASET_DIR)

    assert result["is_valid"] is True
    assert result["record_count"] > 0


def test_validate_record_rejects_missing_dat_file():
    bad_dir = DATASET_DIR / "bad_record"
    bad_dir.mkdir(exist_ok=True)
    bad_dir.joinpath("bad_record.hea").write_text("bad_record 2 360 1000\n", encoding="utf-8")

    try:
        with pytest.raises(FileNotFoundError):
            validate_record(bad_dir / "bad_record")
    finally:
        bad_dir.joinpath("bad_record.hea").unlink(missing_ok=True)
        bad_dir.rmdir()
