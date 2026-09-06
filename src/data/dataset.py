from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset

from src.noise.noise_mix import mix_clean_and_noise
from src.preprocessing.window import extract_window, generate_windows, normalize_window


def load_noise_record(record_path: str | Path) -> dict[str, Any]:
    """Load one nstdb noise record as a finite 1D physical signal."""
    base = Path(record_path)
    record = wfdb.rdrecord(str(base), physical=True)
    if record.fs != 360:
        raise ValueError(f"Expected sampling rate 360 Hz, got {record.fs}")
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float32)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError(f"Invalid noise signal in record {base}")
    return {
        "name": base.name,
        "path": str(base),
        "signal": signal,
        "sample_rate_hz": float(record.fs),
        "length": int(signal.size),
    }


def load_noise_records(noise_dir: str | Path, names: Iterable[str] = ("em", "bw")) -> dict[str, dict[str, Any]]:
    """Load the requested local nstdb noise records by name."""
    root = Path(noise_dir)
    loaded = {}
    for name in names:
        loaded[name] = load_noise_record(root / name)
    return loaded


def combine_noise_signals(
    noise_records: dict[str, dict[str, Any]],
    weights: dict[str, float],
    length: int,
) -> np.ndarray:
    """Create a weighted, length-aligned noise signal from loaded records."""
    if length <= 0:
        raise ValueError("length must be positive")
    if not weights or any(weight < 0.0 for weight in weights.values()):
        raise ValueError("Noise weights must be non-empty and non-negative")
    total_weight = sum(weights.values())
    if np.isclose(total_weight, 0.0):
        raise ValueError("At least one noise weight must be positive")

    combined = np.zeros(length, dtype=np.float32)
    for name, weight in weights.items():
        if np.isclose(weight, 0.0):
            continue
        if name not in noise_records:
            raise KeyError(f"Noise record {name!r} was not loaded")
        signal = noise_records[name]["signal"]
        indices = np.arange(length) % signal.size
        combined += (weight / total_weight) * signal[indices]
    return combined.astype(np.float32)


class ECGDenoisingDataset(Dataset):
    """PyTorch dataset producing noisy ECG inputs and clean training targets."""

    def __init__(
        self,
        clean_records: Iterable[dict[str, Any]],
        noise_records: dict[str, dict[str, Any]],
        noise_weights: dict[str, float],
        snr_range_db: tuple[float, float] = (-5.0, 15.0),
        window_length: int = 256,
        window_stride: int = 128,
        seed: int = 42,
        deterministic: bool = True,
    ):
        if snr_range_db[0] > snr_range_db[1]:
            raise ValueError("snr_range_db must be ordered as (minimum, maximum)")
        noise_length = max(
            int(record["signal"].size) for record in noise_records.values()
        )
        self.noise_signal = combine_noise_signals(noise_records, noise_weights, noise_length)
        self.snr_range_db = snr_range_db
        self.seed = seed
        self.deterministic = deterministic
        self._samples: list[dict[str, Any]] = []

        for record in clean_records:
            record_name = str(record["name"])
            windows = generate_windows(
                record["signal"],
                length=window_length,
                stride=window_stride,
                normalize=False,
            )
            for item in windows:
                self._samples.append({
                    "record": record_name,
                    "start": int(item["start"]),
                    "clean": np.asarray(item["window"], dtype=np.float32),
                })

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        sample = self._samples[index]
        clean_raw = sample["clean"]
        start = sample["start"]
        max_start = max(0, self.noise_signal.size - 256)
        noise_start = start % (max_start + 1)
        noise = extract_window(self.noise_signal, noise_start, length=256)

        if self.deterministic:
            rng = np.random.default_rng(self.seed + index)
        else:
            rng = np.random.default_rng()
        target_snr = float(rng.uniform(*self.snr_range_db))
        noisy_raw, alpha = mix_clean_and_noise(clean_raw, noise, target_snr)

        clean = normalize_window(clean_raw)
        noisy = normalize_window(noisy_raw)
        metadata = {
            "record": sample["record"],
            "start": start,
            "target_snr_db": target_snr,
            "noise_scale": alpha,
        }
        return (
            torch.from_numpy(noisy.copy()).unsqueeze(0),
            torch.from_numpy(clean.copy()).unsqueeze(0),
            metadata,
        )


def create_dataloader(
    dataset: ECGDenoisingDataset,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a PyTorch loader for ECG windows."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
