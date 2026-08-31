import numpy as np

from src.models.blocks import AutoencoderBlock, QuantConv1D, QuantReLU
from src.models.quant_autoencoder import ECGDenoiseAutoencoder


def test_quant_conv1d_preserves_signal_length_for_same_padding():
    layer = QuantConv1D(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
    x = np.random.default_rng(0).normal(0.0, 1.0, (1, 256)).astype(np.float32)
    y = layer(x)
    assert y.shape[0] == 1
    assert y.shape[1] == 128
    assert y.shape[2] == 16


def test_quant_relu_is_signed_and_finite():
    x = np.array([[-1.0, 0.0, 2.0]], dtype=np.float32)
    y = QuantReLU()(x)
    assert y.shape == x.shape
    assert np.all(y >= 0.0)
    assert np.isfinite(y).all()


def test_autoencoder_outputs_expected_shape():
    model = ECGDenoiseAutoencoder()
    x = np.random.default_rng(1).normal(0.0, 1.0, (1, 256)).astype(np.float32)
    y = model(x)
    assert y.shape == (1, 256)
