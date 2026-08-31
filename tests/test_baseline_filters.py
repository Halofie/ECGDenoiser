import numpy as np

from src.baseline.filters import moving_average_filter, highpass_filter


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
