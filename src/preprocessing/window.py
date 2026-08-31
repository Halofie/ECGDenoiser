from __future__ import annotations

import numpy as np


def validate_window_length(window: np.ndarray, expected_length: int = 256) -> int:
    """Validate that a signal window has the required exact length.

    Phase 3 requirement: exactly 256 contiguous samples per ECG window.
    """
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal window, got shape {arr.shape}")
    if arr.shape[0] != expected_length:
        raise ValueError(
            f"Expected window length {expected_length}, got {arr.shape[0]}"
        )
    return arr.shape[0]


def extract_window(signal: np.ndarray, start: int, length: int = 256) -> np.ndarray:
    """Extract a contiguous signal window of the requested length."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {arr.shape}")
    if start < 0 or start + length > arr.shape[0]:
        raise ValueError(
            f"Window starting at index {start} with length {length} exceeds signal bounds"
        )
    return arr[start : start + length].copy()


def normalize_window(window: np.ndarray) -> np.ndarray:
    """Normalize a 1D signal to the range [-1, 1] using min-max scaling.

    Formula:
        x_norm = 2 * (x - min(x)) / (max(x) - min(x)) - 1

    This function intentionally accepts any 1D signal shape, while the exact
    256-sample validation remains available via validate_window_length().
    """
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {arr.shape}")

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    span = max_val - min_val
    if np.isclose(span, 0.0):
        return np.zeros_like(arr, dtype=np.float32)

    norm = 2.0 * (arr - min_val) / span - 1.0
    return norm.astype(np.float32)
