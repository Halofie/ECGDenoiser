from pathlib import Path

import numpy as np

from src.fpga.deployment import calibrate_signals


def test_config_calibration_uses_all_signal_ranges():
    calibration = calibrate_signals(
        [
            np.array([-1.0, 0.0], dtype=np.float32),
            np.array([2.0, 3.0], dtype=np.float32),
        ]
    )
    assert calibration.input_min == -1.0
    assert calibration.input_max == 3.0
