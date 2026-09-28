from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from src.models.quant_autoencoder import ECGDenoiseAutoencoder
from src.preprocessing.window import normalize_fixed_affine, validate_window_length


@dataclass(frozen=True)
class InputCalibration:
    """Fixed affine calibration shared by software and FPGA inference."""

    offset: float
    scale: float
    input_min: float
    input_max: float
    method: str = "global_minmax_to_minus1_1"
    version: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.offset) or not np.isfinite(self.scale):
            raise ValueError("Calibration offset and scale must be finite")
        if self.scale <= 0.0:
            raise ValueError("Calibration scale must be positive")
        if self.input_min > self.input_max:
            raise ValueError("Calibration input range must be ordered")


def calibrate_signals(signals: Iterable[np.ndarray]) -> InputCalibration:
    """Compute one affine [-1, 1] calibration from training signals."""
    minimum = np.inf
    maximum = -np.inf
    seen = False
    for signal in signals:
        arr = np.asarray(signal, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D calibration signal, got shape {arr.shape}")
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError("Calibration signals must be non-empty and finite")
        minimum = min(minimum, float(np.min(arr)))
        maximum = max(maximum, float(np.max(arr)))
        seen = True
    if not seen or np.isclose(maximum, minimum):
        raise ValueError("Calibration requires non-constant input signals")
    return InputCalibration(
        offset=(minimum + maximum) / 2.0,
        scale=(maximum - minimum) / 2.0,
        input_min=minimum,
        input_max=maximum,
    )


def save_calibration(calibration: InputCalibration, path: str | Path) -> None:
    """Write calibration metadata as stable, human-readable JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(calibration), indent=2) + "\n", encoding="utf-8")


def load_calibration(path: str | Path) -> InputCalibration:
    """Load and validate calibration metadata."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return InputCalibration(**data)


def quantize_int16(signal: np.ndarray, calibration: InputCalibration) -> np.ndarray:
    """Normalize a signal and encode it as signed int16 fixed-point samples."""
    normalized = normalize_fixed_affine(signal, calibration.offset, calibration.scale)
    return np.rint(normalized * 32767.0).astype(np.int16)


def dequantize_int16(samples: np.ndarray) -> np.ndarray:
    """Decode signed int16 samples into the model's [-1, 1] domain."""
    values = np.asarray(samples)
    if values.ndim != 1:
        raise ValueError(f"Expected 1D int16 samples, got shape {values.shape}")
    if values.dtype != np.int16:
        raise ValueError(f"Expected int16 samples, got {values.dtype}")
    return (values.astype(np.float32) / 32767.0).clip(-1.0, 1.0)


def run_reference_inference(
    samples: np.ndarray,
    checkpoint_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one calibrated int16 window through the PyTorch reference model."""
    validate_window_length(np.asarray(samples), expected_length=256)
    model = ECGDenoiseAutoencoder()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    model_input = torch.from_numpy(dequantize_int16(samples)).view(1, 1, 256)
    with torch.inference_mode():
        output = model(model_input).squeeze(0).squeeze(0).numpy()
    output_int16 = np.rint(np.clip(output, -1.0, 1.0) * 32767.0).astype(np.int16)
    return model_input.numpy().reshape(256), output_int16


def save_golden_vector(
    input_samples: np.ndarray,
    model_input: np.ndarray,
    output_samples: np.ndarray,
    path: str | Path,
) -> None:
    """Save the bit-oriented input and reference output for HLS comparisons."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        input_samples=np.asarray(input_samples, dtype=np.int16),
        model_input=np.asarray(model_input, dtype=np.float32),
        output_samples=np.asarray(output_samples, dtype=np.int16),
    )
