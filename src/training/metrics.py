from __future__ import annotations

import numpy as np


def mse(clean: np.ndarray, pred: np.ndarray) -> float:
    """Compute mean squared error between clean and predicted signals."""
    clean_arr = np.asarray(clean, dtype=np.float32)
    pred_arr = np.asarray(pred, dtype=np.float32)
    if clean_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: {clean_arr.shape} vs {pred_arr.shape}")
    return float(np.mean(np.square(clean_arr - pred_arr)))


def snr_improvement(clean: np.ndarray, pred: np.ndarray) -> float:
    """Compute output SNR (not delta-SNR) as a basic training metric."""
    clean_arr = np.asarray(clean, dtype=np.float32)
    pred_arr = np.asarray(pred, dtype=np.float32)
    if clean_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: {clean_arr.shape} vs {pred_arr.shape}")
    error = clean_arr - pred_arr
    num = float(np.sum(np.square(clean_arr)))
    den = float(np.sum(np.square(error)))
    if np.isclose(den, 0.0):
        return float("inf")
    return float(10.0 * np.log10(num / den))


def prd(clean: np.ndarray, pred: np.ndarray) -> float:
    """Percentage Root-Mean-Square Difference used as a clinical fidelity metric."""
    clean_arr = np.asarray(clean, dtype=np.float32)
    pred_arr = np.asarray(pred, dtype=np.float32)
    if clean_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: {clean_arr.shape} vs {pred_arr.shape}")
    num = float(np.sum(np.square(clean_arr - pred_arr)))
    den = float(np.sum(np.square(clean_arr)))
    if np.isclose(den, 0.0):
        return 0.0
    return float(np.sqrt(num / den) * 100.0)
