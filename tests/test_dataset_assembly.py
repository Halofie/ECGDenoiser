from pathlib import Path

import numpy as np

from src.data.dataset import combine_noise_signals, load_noise_records


NOISE_DIR = Path(__file__).resolve().parents[1] / "mit-bih-noise-stress-test-database-1.0.0"


def test_load_local_em_and_bw_noise_records():
    records = load_noise_records(NOISE_DIR, names=("em", "bw"))

    assert set(records) == {"em", "bw"}
    assert all(record["sample_rate_hz"] == 360.0 for record in records.values())
    assert all(record["signal"].ndim == 1 for record in records.values())


def test_combine_noise_signals_applies_normalized_weights_and_cycles_length():
    records = {
        "em": {"signal": np.ones(3, dtype=np.float32)},
        "bw": {"signal": np.array([2.0, 3.0], dtype=np.float32)},
    }

    combined = combine_noise_signals(records, {"em": 0.7, "bw": 0.3}, length=5)

    expected = np.array([1.3, 1.6, 1.3, 1.6, 1.3], dtype=np.float32)
    assert np.allclose(combined, expected)
