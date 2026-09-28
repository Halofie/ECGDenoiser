import numpy as np
import pytest

from src.fpga.deployment import (
    InputCalibration,
    calibrate_signals,
    dequantize_int16,
    load_calibration,
    quantize_int16,
    save_calibration,
)


def test_calibration_uses_global_range():
    calibration = calibrate_signals(
        [np.array([-2.0, 0.0], dtype=np.float32), np.array([1.0, 4.0], dtype=np.float32)]
    )
    assert calibration.offset == 1.0
    assert calibration.scale == 3.0
    assert calibration.input_min == -2.0
    assert calibration.input_max == 4.0


def test_int16_round_trip_matches_fixed_point_domain():
    calibration = InputCalibration(offset=0.0, scale=2.0, input_min=-2.0, input_max=2.0)
    signal = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
    encoded = quantize_int16(signal, calibration)
    assert encoded.dtype == np.int16
    assert np.allclose(dequantize_int16(encoded), [-1.0, 0.0, 1.0])


def test_calibration_json_round_trip(tmp_path):
    original = InputCalibration(offset=2.0, scale=3.0, input_min=-1.0, input_max=5.0)
    path = tmp_path / "calibration.json"
    save_calibration(original, path)
    assert load_calibration(path) == original


def test_calibration_rejects_constant_signals():
    with pytest.raises(ValueError, match="non-constant"):
        calibrate_signals([np.ones(4, dtype=np.float32)])
