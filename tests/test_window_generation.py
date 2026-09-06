import numpy as np

from src.preprocessing.window import generate_windows


def test_generate_windows_uses_128_stride_and_discards_tail():
    signal = np.arange(600, dtype=np.float32)

    windows = generate_windows(signal, length=256, stride=128, normalize=False)

    assert [item["start"] for item in windows] == [0, 128, 256]
    assert all(item["window"].shape == (256,) for item in windows)
    assert np.array_equal(windows[-1]["window"], signal[256:512])


def test_generate_windows_normalizes_each_window():
    signal = np.arange(256, dtype=np.float32)

    windows = generate_windows(signal, normalize=True)

    assert len(windows) == 1
    assert np.isclose(np.min(windows[0]["window"]), -1.0)
    assert np.isclose(np.max(windows[0]["window"]), 1.0)


def test_generate_windows_returns_no_partial_window():
    signal = np.arange(255, dtype=np.float32)

    assert generate_windows(signal) == []
