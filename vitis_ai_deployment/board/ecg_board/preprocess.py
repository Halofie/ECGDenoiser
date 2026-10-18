"""Deployable window normalization shared by training, host, and board.

The DPU model consumes a 512-sample window scaled to roughly [-1, 1]. On the
board only the *noisy* signal exists, so the affine transform must be fitted
from the noisy window alone; the model's output is mapped back to physical
units with the same transform.

Methods:
  noisy_minmax  -- the noisy window's own min/max map exactly to [-1, 1].
  noisy_robust  -- the noisy window's [p, 100-p] percentiles map to [-1, 1]
                   and values outside are clipped, so a single noise spike
                   does not squash the whole ECG into a narrow band.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

WINDOW_LENGTH = 512
METHODS = ("noisy_minmax", "noisy_robust")
DEFAULT_METHOD = "noisy_minmax"
DEFAULT_PERCENTILE = 1.0
_MIN_HALF_SPAN = 1e-6


class Affine(NamedTuple):
    """x_norm = (x - offset) / half_span ; x = x_norm * half_span + offset."""

    offset: float
    half_span: float


def fit_affine(window: np.ndarray, method: str = DEFAULT_METHOD, percentile: float = DEFAULT_PERCENTILE) -> Affine:
    values = np.asarray(window, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Expected a non-empty 1D window, got shape %s" % (values.shape,))
    if not np.isfinite(values).all():
        raise ValueError("Window contains non-finite values")
    if method == "noisy_minmax":
        low, high = float(values.min()), float(values.max())
    elif method == "noisy_robust":
        if not 0.0 <= percentile < 50.0:
            raise ValueError("percentile must be in [0, 50)")
        low, high = (float(v) for v in np.percentile(values, [percentile, 100.0 - percentile]))
    else:
        raise ValueError("Unknown normalization method %r; expected one of %s" % (method, METHODS))
    return Affine(offset=(high + low) / 2.0, half_span=max((high - low) / 2.0, _MIN_HALF_SPAN))


def apply_affine(window: np.ndarray, affine: Affine, clip: bool = True) -> np.ndarray:
    scaled = (np.asarray(window, dtype=np.float64) - affine.offset) / affine.half_span
    if clip:
        scaled = np.clip(scaled, -1.0, 1.0)
    return scaled.astype(np.float32)


def invert_affine(normalized: np.ndarray, affine: Affine) -> np.ndarray:
    return (np.asarray(normalized, dtype=np.float64) * affine.half_span + affine.offset).astype(np.float32)


def normalize(window: np.ndarray, method: str = DEFAULT_METHOD, percentile: float = DEFAULT_PERCENTILE):
    """Return (normalized float32 window, affine used)."""
    affine = fit_affine(window, method, percentile)
    return apply_affine(window, affine), affine


def to_int16(values: np.ndarray):
    """Round to int16, saturating. Returns (int16 array, number of clipped samples)."""
    rounded = np.round(np.asarray(values, dtype=np.float64))
    clipped = int(np.count_nonzero((rounded < -32768) | (rounded > 32767)))
    return np.clip(rounded, -32768, 32767).astype(np.int16), clipped
