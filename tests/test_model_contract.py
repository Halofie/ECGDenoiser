import torch

from src.models.blocks import QuantConv1D, QuantReLU
from src.models.quant_autoencoder import ECGDenoiseAutoencoder


def test_quant_conv1d_preserves_signal_length_for_same_padding():
    layer = QuantConv1D(in_channels=1, out_channels=16, kernel_size=7, stride=2, padding=3)
    x = torch.randn(1, 1, 256)
    y = layer(x)
    assert tuple(y.shape) == (1, 16, 128)


def test_quant_conv1d_same_length_path_preserves_window_size():
    layer = QuantConv1D(in_channels=1, out_channels=16, kernel_size=7, padding=3)
    x = torch.randn(1, 1, 256)
    y = layer(x)
    assert tuple(y.shape) == (1, 16, 256)


def test_quant_relu_is_signed_and_finite():
    x = torch.tensor([[-1.0, 0.0, 2.0]])
    y = QuantReLU()(x)
    assert tuple(y.shape) == tuple(x.shape)
    assert torch.all(y >= 0.0)
    assert torch.isfinite(y).all()


def test_autoencoder_outputs_expected_shape():
    model = ECGDenoiseAutoencoder()
    x = torch.randn(1, 1, 256)
    y = model(x)
    assert tuple(y.shape) == (1, 1, 256)


def test_autoencoder_is_trainable_and_has_quantized_layers():
    model = ECGDenoiseAutoencoder()
    x = torch.randn(2, 1, 256)
    loss = torch.nn.functional.mse_loss(model(x), x)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())
    assert sum(1 for module in model.modules() if isinstance(module, QuantConv1D)) == 6
