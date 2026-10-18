from __future__ import annotations

import numpy as np
from scipy import signal


def _check_1d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D signal, got shape {arr.shape}")
    return arr


def moving_average_filter(signal_in: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply a simple moving-average baseline smoother."""
    arr = _check_1d(np.asarray(signal_in, dtype=np.float32))
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    padded = np.pad(arr, (window_size // 2, window_size - 1 - window_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _zero_phase_filter(sos: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Apply an SOS filter with zero phase, falling back to a causal pass on short signals."""
    try:
        return signal.sosfiltfilt(sos, arr).astype(np.float32)
    except ValueError:
        return signal.sosfilt(sos, arr).astype(np.float32)


def butterworth_highpass_filter(
    signal_in: np.ndarray,
    cutoff_hz: float = 0.67,
    sample_rate_hz: float = 360.0,
    order: int = 2,
) -> np.ndarray:
    """Remove baseline wander with a zero-phase Butterworth high-pass filter.

    A 0.5-0.7 Hz cutoff is the standard clinical-monitor choice: it removes
    respiration-driven baseline drift while leaving the ST segment intact.
    """
    arr = _check_1d(np.asarray(signal_in, dtype=np.float32))
    nyquist = sample_rate_hz / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"cutoff_hz must be in (0, {nyquist})")

    sos = signal.butter(order, cutoff_hz / nyquist, btype="highpass", output="sos")
    return _zero_phase_filter(sos, arr)


def butterworth_lowpass_filter(
    signal_in: np.ndarray,
    cutoff_hz: float = 40.0,
    sample_rate_hz: float = 360.0,
    order: int = 2,
) -> np.ndarray:
    """Suppress high-frequency EMG/electrode-motion noise with a zero-phase low-pass filter."""
    arr = _check_1d(np.asarray(signal_in, dtype=np.float32))
    nyquist = sample_rate_hz / 2.0
    if not 0 < cutoff_hz < nyquist:
        raise ValueError(f"cutoff_hz must be in (0, {nyquist})")

    sos = signal.butter(order, cutoff_hz / nyquist, btype="lowpass", output="sos")
    return _zero_phase_filter(sos, arr)


def butterworth_bandpass_filter(
    signal_in: np.ndarray,
    low_hz: float = 0.67,
    high_hz: float = 40.0,
    sample_rate_hz: float = 360.0,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass covering the diagnostic ECG band."""
    arr = _check_1d(np.asarray(signal_in, dtype=np.float32))
    nyquist = sample_rate_hz / 2.0
    if not 0 < low_hz < high_hz < nyquist:
        raise ValueError(f"Require 0 < low_hz < high_hz < {nyquist}")

    sos = signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="bandpass", output="sos")
    return _zero_phase_filter(sos, arr)


def notch_filter(
    signal_in: np.ndarray,
    notch_hz: float = 60.0,
    quality_factor: float = 30.0,
    sample_rate_hz: float = 360.0,
) -> np.ndarray:
    """Suppress narrowband mains-frequency interference (60 Hz, configurable to 50 Hz)."""
    arr = _check_1d(np.asarray(signal_in, dtype=np.float32))
    nyquist = sample_rate_hz / 2.0
    if not 0 < notch_hz < nyquist:
        raise ValueError(f"notch_hz must be in (0, {nyquist})")

    b, a = signal.iirnotch(notch_hz / nyquist, quality_factor)
    try:
        return signal.filtfilt(b, a, arr).astype(np.float32)
    except ValueError:
        return signal.lfilter(b, a, arr).astype(np.float32)


def median_baseline_removal(
    signal_in: np.ndarray,
    sample_rate_hz: float = 360.0,
    short_window_s: float = 0.2,
    long_window_s: float = 0.6,
) -> np.ndarray:
    """Estimate and subtract baseline wander with a cascaded two-stage median filter.

    This is the classic Pan-Tompkins-style baseline estimator: a short median
    filter removes the QRS/P/T waves to expose the baseline, a longer median
    filter smooths that estimate, and the result is subtracted from the
    original signal. It preserves QRS morphology better than a linear
    high-pass filter because it never rings around sharp transitions.
    """
    arr = _check_1d(np.asarray(signal_in, dtype=np.float32))

    def _odd_kernel(seconds: float) -> int:
        samples = max(3, int(round(seconds * sample_rate_hz)))
        return samples if samples % 2 == 1 else samples + 1

    short_kernel = min(_odd_kernel(short_window_s), arr.size - (1 - arr.size % 2))
    short_kernel = max(3, short_kernel)
    stage_one = signal.medfilt(arr, kernel_size=short_kernel)

    long_kernel = min(_odd_kernel(long_window_s), arr.size - (1 - arr.size % 2))
    long_kernel = max(3, long_kernel)
    baseline = signal.medfilt(stage_one, kernel_size=long_kernel)

    return (arr - baseline).astype(np.float32)


def dsp_denoise_filter(
    signal_in: np.ndarray,
    sample_rate_hz: float = 360.0,
    low_hz: float = 0.5,
    high_hz: float = 40.0,
    notch_hz: float | None = 60.0,
    order: int = 2,
) -> np.ndarray:
    """Composite classical DSP denoising pipeline used as the "Track A" baseline.

    Stage 1: zero-phase Butterworth band-pass (0.5-40 Hz) — the standard
    clinical-monitor front end, covering baseline-wander rejection on the low
    edge and EMG/electrode-motion rejection on the high edge.
    Stage 2 (optional): mains-frequency notch filter.

    This is deliberately the textbook linear-filter baseline, not a tuned
    cascade: an earlier iteration added a median-filter baseline-wander stage
    (`median_baseline_removal`), but on the 512-sample (~1.4 s) evaluation
    windows used here its ~0.6 s kernel is comparable to a single heartbeat
    period and measurably distorted QRS morphology more than it removed
    noise (see `reports/dsp_vs_ai_comparison.md`). It is kept as a standalone
    function for continuous, multi-beat recordings where that failure mode
    does not apply, but is not part of this default composite.

    The MIT-BIH NSTDB `em`/`bw` noise records carry substantial energy
    inside the 0.5-40 Hz diagnostic ECG band itself, so a linear
    frequency-selective filter cannot fully separate noise from signal here
    -- this is the expected, literature-consistent result, and the reason a
    learned model ("Track B") is evaluated against this baseline rather than
    assumed to be unnecessary.
    """
    arr = _check_1d(np.asarray(signal_in, dtype=np.float32))

    stage1 = butterworth_bandpass_filter(arr, low_hz=low_hz, high_hz=high_hz, sample_rate_hz=sample_rate_hz, order=order)
    if notch_hz is not None:
        stage1 = notch_filter(stage1, notch_hz=notch_hz, sample_rate_hz=sample_rate_hz)
    return stage1.astype(np.float32)


def highpass_filter(signal_in: np.ndarray, cutoff_hz: float = 0.5, sample_rate_hz: float = 360.0) -> np.ndarray:
    """Legacy crude high-pass-like baseline suppression using a finite-difference filter.

    Kept for backward compatibility with earlier baseline comparisons; prefer
    `butterworth_highpass_filter` or `dsp_denoise_filter` for new work.
    """
    arr = _check_1d(np.asarray(signal_in, dtype=np.float32))
    diff = np.diff(arr, prepend=arr[0])
    return diff.astype(np.float32)
