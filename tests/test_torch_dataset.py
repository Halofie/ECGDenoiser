from pathlib import Path

import numpy as np
import torch

from src.data.dataset import ECGDenoisingDataset, load_noise_records
from src.fpga.deployment import InputCalibration


NOISE_DIR = Path(__file__).resolve().parents[1] / "mit-bih-noise-stress-test-database-1.0.0"


def test_dataset_returns_model_ready_tensors_and_metadata():
    clean = {"name": "synthetic", "signal": np.arange(512, dtype=np.float32)}
    noise = load_noise_records(NOISE_DIR, names=("em", "bw"))
    dataset = ECGDenoisingDataset(
        clean_records=[clean],
        noise_records=noise,
        noise_weights={"em": 0.7, "bw": 0.3},
        snr_range_db=(-5.0, 15.0),
        seed=42,
        deterministic=True,
    )

    noisy, target, metadata = dataset[0]

    assert len(dataset) == 3
    assert noisy.shape == target.shape == (1, 256)
    assert noisy.dtype == target.dtype == torch.float32
    assert torch.all(target >= -1.0) and torch.all(target <= 1.0)
    assert metadata["record"] == "synthetic"
    assert -5.0 <= metadata["target_snr_db"] <= 15.0


def test_deterministic_dataset_repeats_same_sample():
    clean = {"name": "synthetic", "signal": np.sin(np.linspace(0, 10, 512)).astype(np.float32)}
    noise = {
        "em": {"signal": np.ones(256, dtype=np.float32)},
        "bw": {"signal": np.ones(256, dtype=np.float32)},
    }
    dataset = ECGDenoisingDataset([clean], noise, {"em": 0.7, "bw": 0.3}, deterministic=True)

    first = dataset[0]
    second = dataset[0]

    assert torch.equal(first[0], second[0])
    assert first[2] == second[2]


def test_dataset_uses_fixed_calibration_when_provided():
    clean = {"name": "synthetic", "signal": np.arange(512, dtype=np.float32)}
    noise = {
        "em": {"signal": np.ones(256, dtype=np.float32)},
        "bw": {"signal": np.zeros(256, dtype=np.float32)},
    }
    calibration = InputCalibration(
        offset=127.5, scale=127.5, input_min=0.0, input_max=255.0
    )
    dataset = ECGDenoisingDataset(
        [clean],
        noise,
        {"em": 1.0, "bw": 0.0},
        calibration=calibration,
    )

    noisy, target, _ = dataset[0]

    assert torch.allclose(target[0, :3], torch.tensor([-1.0, -0.9922, -0.9843]), atol=1e-3)
    assert torch.all(noisy <= 1.0) and torch.all(noisy >= -1.0)
