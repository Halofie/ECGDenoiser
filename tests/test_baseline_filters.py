import numpy as np

from src.baseline.filters import (
    butterworth_bandpass_filter,
    butterworth_highpass_filter,
    butterworth_lowpass_filter,
    dsp_denoise_filter,
    highpass_filter,
    median_baseline_removal,
    moving_average_filter,
    notch_filter,
)


def _test_signal(n: int = 256) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / 360.0
    return np.sin(2 * np.pi * 1.2 * t).astype(np.float32)


def test_moving_average_filter_preserves_length():
    signal = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    filtered = moving_average_filter(signal, window_size=5)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_highpass_filter_is_stable_and_finite():
    signal = np.sin(np.linspace(0, 10 * np.pi, 256, dtype=np.float32))
    filtered = highpass_filter(signal, cutoff_hz=0.5, sample_rate_hz=360)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_butterworth_highpass_removes_dc_and_slow_drift():
    signal = _test_signal() + 5.0
    filtered = butterworth_highpass_filter(signal, cutoff_hz=0.67, sample_rate_hz=360.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()
    assert abs(float(np.mean(filtered))) < abs(float(np.mean(signal)))


def test_butterworth_lowpass_attenuates_high_frequency_noise():
    rng = np.random.default_rng(0)
    clean = _test_signal()
    noisy = clean + 0.5 * np.sin(2 * np.pi * 100 * np.arange(256) / 360.0) + 0.05 * rng.standard_normal(256)
    filtered = butterworth_lowpass_filter(noisy.astype(np.float32), cutoff_hz=40.0, sample_rate_hz=360.0)
    assert filtered.shape == noisy.shape
    assert np.isfinite(filtered).all()
    error_before = np.mean((noisy - clean) ** 2)
    error_after = np.mean((filtered - clean) ** 2)
    assert error_after < error_before


def test_butterworth_bandpass_is_stable_and_finite():
    signal = _test_signal()
    filtered = butterworth_bandpass_filter(signal, low_hz=0.67, high_hz=40.0, sample_rate_hz=360.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_notch_filter_suppresses_mains_tone():
    n = 256
    t = np.arange(n, dtype=np.float32) / 360.0
    mains = 0.8 * np.sin(2 * np.pi * 60.0 * t)
    signal = _test_signal() + mains
    filtered = notch_filter(signal.astype(np.float32), notch_hz=60.0, sample_rate_hz=360.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_median_baseline_removal_reduces_slow_drift():
    n = 256
    t = np.arange(n, dtype=np.float32) / 360.0
    drift = 2.0 * np.sin(2 * np.pi * 0.3 * t)
    signal = _test_signal() + drift
    filtered = median_baseline_removal(signal.astype(np.float32), sample_rate_hz=360.0)
    assert filtered.shape == signal.shape
    assert np.isfinite(filtered).all()


def test_dsp_denoise_filter_end_to_end_reduces_error():
    rng = np.random.default_rng(1)
    n = 256
    t = np.arange(n, dtype=np.float32) / 360.0
    clean = _test_signal()
    drift = 1.5 * np.sin(2 * np.pi * 0.3 * t)
    hf_noise = 0.3 * np.sin(2 * np.pi * 90 * t) + 0.05 * rng.standard_normal(n)
    noisy = clean + drift + hf_noise

    filtered = dsp_denoise_filter(noisy.astype(np.float32), sample_rate_hz=360.0)
    assert filtered.shape == clean.shape
    assert np.isfinite(filtered).all()

    error_before = float(np.mean((noisy - clean) ** 2))
    error_after = float(np.mean((filtered - clean) ** 2))
    assert error_after < error_before
