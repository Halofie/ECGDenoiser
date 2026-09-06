from __future__ import annotations

import torch.nn as nn
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat


class QuantConv1D(qnn.QuantConv1d):
    """8-bit per-tensor quantized Conv1D used by the autoencoder."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            weight_bit_width=8,
        )


class QuantReLU(qnn.QuantReLU):
    """8-bit quantized ReLU activation."""

    def __init__(self):
        super().__init__(act_quant=Int8ActPerTensorFloat, bit_width=8)


class QuantConvBlock(nn.Module):
    """Quantized convolution, BatchNorm, and quantized ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = QuantConv1D(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.activation = QuantReLU()

    def forward(self, inputs):
        return self.activation(self.batch_norm(self.conv(inputs)))
