from __future__ import annotations

import numpy as np


def signal_power(signal: np.ndarray) -> float:
    """Compute the signal power as the mean squared value."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("Signal must not be empty")
    return float(np.mean(np.square(arr)))


def snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """Compute SNR in dB for a clean signal and noise estimate."""
    clean_arr = np.asarray(clean, dtype=np.float32)
    noise_arr = np.asarray(noise, dtype=np.float32)
    if clean_arr.shape != noise_arr.shape:
        raise ValueError(f"Clean/noise shapes differ: {clean_arr.shape} vs {noise_arr.shape}")
    signal_pwr = signal_power(clean_arr)
    noise_pwr = signal_power(noise_arr)
    if np.isclose(noise_pwr, 0.0):
        return float("inf")
    return float(10.0 * np.log10(signal_pwr / noise_pwr))


def compute_noise_scale(clean: np.ndarray, noise: np.ndarray, snr_db_target: float) -> float:
    """Compute alpha such that the injected noise achieves the target SNR.

    Using the standard relationship:
        SNR_dB = 10 log10(P_signal / P_noise)
        alpha = sqrt(P_signal / (P_noise * 10^(SNR_dB/10)))
    """
    clean_arr = np.asarray(clean, dtype=np.float32)
    noise_arr = np.asarray(noise, dtype=np.float32)
    if clean_arr.shape != noise_arr.shape:
        raise ValueError(f"Clean/noise shapes differ: {clean_arr.shape} vs {noise_arr.shape}")

    signal_pwr = signal_power(clean_arr)
    noise_pwr = signal_power(noise_arr)
    if np.isclose(noise_pwr, 0.0):
        raise ValueError("Noise power must be non-zero to compute a target scale.")

    target_ratio = 10.0 ** (snr_db_target / 10.0)
    alpha = float(np.sqrt(signal_pwr / (noise_pwr * target_ratio)))
    return alpha
