from __future__ import annotations

import numpy as np

from src.models.blocks import AutoencoderBlock, QuantConv1D, QuantReLU


class ECGDenoiseAutoencoder:
    """Contract-level 1D autoencoder skeleton for the project PRD."""

    def __init__(self):
        self.encoder1 = AutoencoderBlock(1, 16, 7, stride=2, padding=3)
        self.relu1 = QuantReLU()
        self.encoder2 = AutoencoderBlock(16, 32, 5, stride=2, padding=2)
        self.relu2 = QuantReLU()
        self.encoder3 = AutoencoderBlock(32, 32, 3, stride=2, padding=1)
        self.relu3 = QuantReLU()
        self.output_conv = QuantConv1D(32, 1, 7, stride=1, padding=3)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Expected shape (batch, length), got {arr.shape}")
        if arr.shape[1] != 256:
            raise ValueError(f"Expected input length 256, got {arr.shape[1]}")

        x_batched = arr[:, :, None]
        x1 = self.encoder1(x_batched)
        x1 = self.relu1(x1)
        x2 = self.encoder2(x1)
        x2 = self.relu2(x2)
        x3 = self.encoder3(x2)
        x3 = self.relu3(x3)

        # Up-sample back to a 256-length output for architecture contract testing.
        x3 = np.repeat(x3, 2, axis=1)
        x3 = np.repeat(x3, 2, axis=1)
        x3 = np.repeat(x3, 2, axis=1)
        x3 = x3[:, :256, :]

        out = self.output_conv(x3)
        return out[:, :, 0]
