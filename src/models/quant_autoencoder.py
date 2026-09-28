from __future__ import annotations

import torch
import torch.nn as nn

from src.models.blocks import QuantConv1D, QuantConvBlock


class ECGDenoiseAutoencoder(nn.Module):
    """Brevitas QAT fixed-length denoising autoencoder for 256-sample windows."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            QuantConvBlock(1, 16, kernel_size=7, stride=1, padding=3),
            QuantConvBlock(16, 32, kernel_size=5, stride=1, padding=2),
            QuantConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            QuantConvBlock(32, 16, kernel_size=3, stride=1, padding=1),
            QuantConvBlock(16, 8, kernel_size=5, stride=1, padding=2),
            QuantConv1D(8, 1, kernel_size=7, stride=1, padding=3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or tuple(inputs.shape[1:]) != (1, 256):
            raise ValueError(f"Expected input shape (batch, 1, 256), got {tuple(inputs.shape)}")
        return self.layers(inputs)
