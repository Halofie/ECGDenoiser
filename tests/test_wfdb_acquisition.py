from pathlib import Path

import numpy as np
import wfdb

from src.data.fetch_dataset import acquire_clean_records, load_clean_record
from src.data.validation import validate_record


NOISE_DIR = Path(__file__).resolve().parents[1] / "mit-bih-noise-stress-test-database-1.0.0"


def test_existing_local_wfdb_record_has_valid_metadata():
    result = validate_record(NOISE_DIR / "em")

    assert result["is_valid"] is True
    assert result["sample_rate_hz"] == 360.0
    assert result["length"] > 0


def test_acquire_without_download_reports_missing_records(tmp_path):
    try:
        acquire_clean_records(["missing"], tmp_path, download_missing=False)
        assert False, "Expected missing record error"
    except FileNotFoundError as exc:
        assert "missing" in str(exc)


def test_load_clean_record_selects_requested_channel(tmp_path):
    path = tmp_path / "clean"
    signal = np.column_stack(
        [np.linspace(-1.0, 1.0, 360), np.linspace(1.0, -1.0, 360)]
    ).astype(np.float64)
    wfdb.wrsamp(
        "clean",
        write_dir=str(tmp_path),
        fs=360,
        units=["mV", "mV"],
        sig_name=["MLII", "V1"],
        p_signal=signal,
    )

    loaded = load_clean_record(path, channel="MLII")

    assert loaded["channel"] == "MLII"
    assert loaded["length"] == 360
    assert loaded["signal"].dtype == np.float32
