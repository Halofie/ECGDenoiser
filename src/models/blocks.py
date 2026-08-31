from __future__ import annotations

import numpy as np


class QuantConv1D:
    """A compact Conv1D abstraction used to define the model contract."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        if arr.ndim != 3:
            raise ValueError(f"Expected shape (batch, length, channels), got {arr.shape}")

        batch, length, channels = arr.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {channels}")

        pad_total = self.padding * 2
        padded = np.pad(arr, ((0, 0), (self.padding, self.padding), (0, 0)), mode="constant")
        out_length = (length + pad_total - self.kernel_size) // self.stride + 1
        out = np.zeros((batch, out_length, self.out_channels), dtype=np.float32)

        for i in range(out_length):
            start = i * self.stride
            segment = padded[:, start : start + self.kernel_size, :]
            base = np.mean(segment, axis=1)  # shape: (batch, channels)
            out[:, i, :] = np.repeat(base[:, 0:1], self.out_channels, axis=1)
        return out


class AutoencoderBlock:
    """Phase 7 placeholder block used to validate model composition."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        self.conv = QuantConv1D(in_channels, out_channels, kernel_size, stride, padding)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.conv(x)


class QuantReLU:
    """ReLU activation used for Phase 7 contract validation."""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return np.maximum(arr, 0.0).astype(np.float32)
