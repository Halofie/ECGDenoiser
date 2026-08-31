from __future__ import annotations

import numpy as np

from src.training.metrics import mse, prd, snr_improvement


def aggregate_metrics(clean: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    """Compute aggregate quality metrics for a clean/predicted signal pair."""
    clean_arr = np.asarray(clean, dtype=np.float32)
    pred_arr = np.asarray(pred, dtype=np.float32)
    if clean_arr.shape != pred_arr.shape:
        raise ValueError(f"Shape mismatch: {clean_arr.shape} vs {pred_arr.shape}")

    return {
        "mse": float(mse(clean_arr, pred_arr)),
        "snr": float(snr_improvement(clean_arr, pred_arr)),
        "prd": float(prd(clean_arr, pred_arr)),
    }


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float | int]:
    """Compute mean summary statistics across a list of metric dicts."""
    if not metrics:
        raise ValueError("metrics must not be empty")

    count = len(metrics)
    mean_mse = float(np.mean([entry["mse"] for entry in metrics]))
    mean_snr = float(np.mean([entry["snr"] for entry in metrics]))
    mean_prd = float(np.mean([entry["prd"] for entry in metrics]))

    return {
        "count": count,
        "mean_mse": mean_mse,
        "mean_snr": mean_snr,
        "mean_prd": mean_prd,
    }
