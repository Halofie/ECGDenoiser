from __future__ import annotations

import numpy as np


def moving_average_filter(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply a simple moving-average baseline smoother.

    Phase 6 introduces a baseline signal-processing benchmark to compare with the
    learned model. This is intentionally lightweight and deterministic.
    """
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {arr.shape}")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    padded = np.pad(arr, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def highpass_filter(signal: np.ndarray, cutoff_hz: float = 0.5, sample_rate_hz: float = 360.0) -> np.ndarray:
    """Apply a crude high-pass-like baseline suppression using a finite-difference filter."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {arr.shape}")

    # A simple first-difference operator approximates baseline removal while staying
    # deterministic and lightweight for benchmark comparisons.
    diff = np.diff(arr, prepend=arr[0])
    return diff.astype(np.float32)
