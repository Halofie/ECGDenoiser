from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import QuantConv1D, QuantConvBlock


class ECGDenoiseAutoencoder(nn.Module):
    """Brevitas QAT 1D convolutional autoencoder for 256-sample ECG windows."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            QuantConvBlock(1, 16, kernel_size=7, stride=2, padding=3),
            QuantConvBlock(16, 32, kernel_size=5, stride=2, padding=2),
            QuantConvBlock(32, 32, kernel_size=3, stride=2, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            QuantConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode="nearest"),
            QuantConvBlock(32, 16, kernel_size=5, stride=1, padding=2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            QuantConv1D(16, 1, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (1, 256):
            raise ValueError(f"Expected input shape (batch, 1, 256), got {tuple(inputs.shape)}")
        return self.decoder(self.encoder(inputs))
