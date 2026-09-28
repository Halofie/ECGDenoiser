import numpy as np
import pytest

from src.preprocessing.window import (
    extract_window,
    normalize_fixed_affine,
    normalize_window,
    validate_window_length,
)


def test_validate_window_length_accepts_exact_256_samples():
    arr = np.arange(256, dtype=np.float32)
    assert validate_window_length(arr) == 256


def test_validate_window_length_rejects_wrong_length():
    arr = np.arange(255, dtype=np.float32)
    try:
        validate_window_length(arr)
        assert False, "Expected ValueError for wrong-length window"
    except ValueError:
        pass


def test_extract_window_extracts_exact_length_from_signal():
    signal = np.linspace(0, 1, 1024, dtype=np.float32)
    window = extract_window(signal, start=100, length=256)
    assert window.shape == (256,)
    assert np.allclose(window, signal[100:356])


def test_normalize_window_maps_to_minus_one_to_one():
    arr = np.array([1.0, 3.0, 5.0], dtype=np.float32)
    norm = normalize_window(arr)
    assert norm.shape == arr.shape
    assert np.min(norm) >= -1.0 - 1e-6
    assert np.max(norm) <= 1.0 + 1e-6


def test_normalize_fixed_affine_uses_shared_calibration_and_clips():
    arr = np.array([-3.0, 0.0, 3.0], dtype=np.float32)
    norm = normalize_fixed_affine(arr, offset=0.0, scale=2.0)
    assert np.allclose(norm, [-1.0, 0.0, 1.0])


def test_normalize_fixed_affine_rejects_non_positive_scale():
    with pytest.raises(ValueError, match="scale must be positive"):
        normalize_fixed_affine(np.array([1.0], dtype=np.float32), offset=0.0, scale=0.0)
