import numpy as np

from src.noise.snr import compute_noise_scale, signal_power, snr_db
from src.noise.noise_mix import mix_clean_and_noise


def test_signal_power_and_snr_are_positive_and_reasonable():
    signal = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    noise = np.array([0.1, -0.2, 0.1, -0.2], dtype=np.float32)

    assert signal_power(signal) > 0
    assert snr_db(signal, noise) > 0


def test_compute_noise_scale_matches_target_snr_formula():
    signal = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    noise = np.random.default_rng(42).normal(0.0, 1.0, 256).astype(np.float32)

    alpha = compute_noise_scale(signal, noise, snr_db_target=6.0)
    noisy = signal + alpha * noise

    measured = snr_db(signal, noisy - signal)
    assert abs(measured - 6.0) < 0.5


def test_mix_clean_and_noise_preserves_length_and_shape():
    clean = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    noise = np.random.default_rng(7).normal(0.0, 1.0, 256).astype(np.float32)

    noisy, alpha = mix_clean_and_noise(clean, noise, 3.0)

    assert noisy.shape == clean.shape == noise.shape
    assert np.isfinite(noisy).all()
    assert alpha > 0
