from __future__ import annotations

import numpy as np

from src.noise.snr import compute_noise_scale


def mix_clean_and_noise(clean: np.ndarray, noise: np.ndarray, snr_db_target: float) -> tuple[np.ndarray, float]:
    """Blend clean signal with noise so the result achieves the target SNR.

    Returns (noisy_signal, alpha), where alpha is the scale applied to noise.
    """
    clean_arr = np.asarray(clean, dtype=np.float32)
    noise_arr = np.asarray(noise, dtype=np.float32)
    if clean_arr.shape != noise_arr.shape:
        raise ValueError(f"Clean/noise shapes differ: {clean_arr.shape} vs {noise_arr.shape}")

    alpha = compute_noise_scale(clean_arr, noise_arr, snr_db_target)
    noisy = clean_arr + alpha * noise_arr
    return noisy.astype(np.float32), float(alpha)
