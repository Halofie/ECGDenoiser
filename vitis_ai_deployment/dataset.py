"""Real ECG dataset wiring for the 512-sample Vitis AI pipeline.

Reuses this repo's existing WFDB/noise/windowing building blocks
(src/data/fetch_dataset.py, src/noise, src/preprocessing) instead of writing
a new loader from scratch. It does NOT import anything from
src/data/dataset.py (ECGDenoisingDataset, combine_noise_signals,
load_noise_records), for three reasons:

  1. ECGDenoisingDataset hardcodes a 256-sample noise window internally
     (`extract_window(self.noise_signal, noise_start, length=256)`),
     independent of the window_length constructor argument. Passing
     window_length=512 through it would mix a 512-sample clean window with
     a 256-sample noise window and fail shape validation in
     mix_clean_and_noise.
  2. src/config.py's validate_config() hard-enforces window.length == 256 as
     part of the hls4ml/256-sample deployment contract. This 512-sample
     path is deliberately a different contract and should not be forced
     through that validator.
  3. src/data/dataset.py imports src.fpga.deployment.InputCalibration at
     module load time, which imports src.models.quant_autoencoder, which
     depends on Brevitas -- a dependency this Vitis AI/DPU pipeline has no
     other reason to need. Importing that module inside the Vitis AI Docker
     image (which only has the packages Xilinx bundled) fails on
     `ModuleNotFoundError: brevitas` even though this pipeline's model never
     touches Brevitas. combine_noise_signals/load_noise_records are
     therefore reimplemented directly below instead of imported.

This module re-implements the same window/noise-mix/normalize pipeline
directly at WINDOW_LENGTH=512, calling the same underlying functions the
256-sample path uses where those don't carry that transitive baggage.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import wfdb
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.fetch_dataset import DEFAULT_CLEAN_RECORDS, acquire_clean_records, load_clean_record
from src.noise.noise_mix import mix_clean_and_noise
from src.preprocessing.window import extract_window, generate_windows


def load_noise_records(noise_dir: str | Path, names: Iterable[str] = ("em", "bw")) -> dict[str, dict[str, Any]]:
    """Load the requested local nstdb noise records by name."""
    root = Path(noise_dir)
    loaded = {}
    for name in names:
        base = root / name
        record = wfdb.rdrecord(str(base), physical=True)
        if record.fs != 360:
            raise ValueError(f"Expected sampling rate 360 Hz, got {record.fs}")
        signal = np.asarray(record.p_signal[:, 0], dtype=np.float32)
        if signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError(f"Invalid noise signal in record {base}")
        loaded[name] = {"name": base.name, "path": str(base), "signal": signal}
    return loaded


def combine_noise_signals(
    noise_records: dict[str, dict[str, Any]], weights: dict[str, float], length: int
) -> np.ndarray:
    """Create a weighted, length-aligned noise signal from loaded records."""
    if length <= 0:
        raise ValueError("length must be positive")
    total_weight = sum(weights.values())
    if np.isclose(total_weight, 0.0):
        raise ValueError("At least one noise weight must be positive")
    combined = np.zeros(length, dtype=np.float32)
    for name, weight in weights.items():
        if np.isclose(weight, 0.0):
            continue
        signal = noise_records[name]["signal"]
        indices = np.arange(length) % signal.size
        combined += (weight / total_weight) * signal[indices]
    return combined.astype(np.float32)

WINDOW_LENGTH = 512
WINDOW_STRIDE = 256
SNR_RANGE_DB = (-5.0, 15.0)
NOISE_WEIGHTS = {"em": 0.7, "bw": 0.3, "ma": 0.0}

# Held out by record (patient), not by window, so validation windows never
# come from a signal the model trained on. Fixed split, last 5 of the 26
# default MIT-BIH records configured in src/data/fetch_dataset.py.
DEFAULT_VAL_RECORDS = DEFAULT_CLEAN_RECORDS[-5:]
DEFAULT_TRAIN_RECORDS = DEFAULT_CLEAN_RECORDS[:-5]


def default_train_val_split(
    records: Iterable[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split a record list into train/val by record, ~80/20, last 20% held out."""
    record_list = list(records) if records is not None else list(DEFAULT_CLEAN_RECORDS)
    if len(record_list) < 2:
        return record_list, []
    val_count = max(1, round(len(record_list) * 0.2))
    return record_list[:-val_count], record_list[-val_count:]


def acquire_noise_records(
    destination: str | Path, names: Iterable[str] = ("em", "bw"), download_missing: bool = True
) -> None:
    """Ensure the requested nstdb noise records exist locally (no acquire
    helper for nstdb exists elsewhere in this repo; src/data/dataset.py only
    loads noise records that are already present on disk).
    """
    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    missing = [
        name
        for name in names
        if not (destination_path / f"{name}.hea").exists()
        or not (destination_path / f"{name}.dat").exists()
    ]
    if not missing:
        return
    if not download_missing:
        raise FileNotFoundError(f"Missing nstdb noise records: {missing}")
    wfdb.dl_database("nstdb", str(destination_path), records=missing)


class ECG512Dataset(Dataset):
    """Real (noisy, clean) 512-sample ECG window pairs for the DPU model."""

    def __init__(
        self,
        clean_dir: str | Path,
        noise_dir: str | Path,
        records: Iterable[str] | None = None,
        window_length: int = WINDOW_LENGTH,
        window_stride: int = WINDOW_STRIDE,
        snr_range_db: tuple[float, float] = SNR_RANGE_DB,
        noise_weights: dict[str, float] = NOISE_WEIGHTS,
        seed: int = 42,
        deterministic: bool = True,
        download_missing: bool = True,
    ):
        clean_meta = acquire_clean_records(
            list(records) if records is not None else None,
            clean_dir,
            download_missing=download_missing,
        )
        clean_records: list[dict[str, Any]] = [
            load_clean_record(meta["path"]) for meta in clean_meta
        ]
        noise_records = load_noise_records(noise_dir, names=("em", "bw"))
        noise_length = max(int(r["signal"].size) for r in noise_records.values())
        self.noise_signal = combine_noise_signals(noise_records, noise_weights, noise_length)

        self.window_length = window_length
        self.snr_range_db = snr_range_db
        self.seed = seed
        self.deterministic = deterministic

        self._samples: list[dict[str, Any]] = []
        for record in clean_records:
            windows = generate_windows(
                record["signal"], length=window_length, stride=window_stride, normalize=False
            )
            for item in windows:
                self._samples.append(
                    {
                        "record": str(record["name"]),
                        "start": int(item["start"]),
                        "clean": np.asarray(item["window"], dtype=np.float32),
                    }
                )
        if not self._samples:
            raise RuntimeError(
                "No windows generated -- check that clean records are long enough "
                f"for a {window_length}-sample window."
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        sample = self._samples[index]
        clean_raw = sample["clean"]
        start = sample["start"]

        max_start = max(0, self.noise_signal.size - self.window_length)
        noise_start = start % (max_start + 1)
        noise = extract_window(self.noise_signal, noise_start, length=self.window_length)

        rng = np.random.default_rng(self.seed + index) if self.deterministic else np.random.default_rng()
        target_snr = float(rng.uniform(*self.snr_range_db))
        noisy_raw, alpha = mix_clean_and_noise(clean_raw, noise, target_snr)

        # Normalize noisy and clean with the SAME affine transform (derived
        # from the clean window), not independently. normalize_window()
        # applied separately to each would rescale the noisy input and the
        # clean target by two different, noise-realization-dependent
        # min/max values -- the network would then have to learn an extra
        # per-window rescaling on top of denoising itself, which is a much
        # harder and noisier target function. Sharing the clean window's
        # scale makes the target a literal "predict the signal on a fixed
        # scale" problem; the noisy input is clipped to [-1, 1] since a
        # low-SNR noise realization can exceed the clean window's range.
        min_val = float(np.min(clean_raw))
        max_val = float(np.max(clean_raw))
        span = max(max_val - min_val, 1e-6)
        clean = ((2.0 * (clean_raw - min_val) / span) - 1.0).astype(np.float32)
        noisy = np.clip((2.0 * (noisy_raw - min_val) / span) - 1.0, -1.0, 1.0).astype(np.float32)

        return (
            torch.from_numpy(noisy.copy()).unsqueeze(0),  # (1, 512)
            torch.from_numpy(clean.copy()).unsqueeze(0),
        )
