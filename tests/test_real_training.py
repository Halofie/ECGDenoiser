import numpy as np
import torch

from src.data.dataset import ECGDenoisingDataset, create_dataloader
from src.models.quant_autoencoder import ECGDenoiseAutoencoder
from src.training.trainer import TrainingConfig, fit


def test_dataloader_and_training_update_model(tmp_path):
    clean_records = [
        {"name": "record_a", "signal": np.sin(np.linspace(0, 20, 512)).astype(np.float32)},
    ]
    noise_records = {
        "em": {"signal": np.ones(512, dtype=np.float32)},
        "bw": {"signal": np.full(512, 0.5, dtype=np.float32)},
    }
    dataset = ECGDenoisingDataset(clean_records, noise_records, {"em": 0.7, "bw": 0.3})
    loader = create_dataloader(dataset, batch_size=2, shuffle=True)
    model = ECGDenoiseAutoencoder()
    before = [parameter.detach().clone() for parameter in model.parameters()]

    history = fit(
        model,
        loader,
        loader,
        TrainingConfig(learning_rate=1e-3, epochs=1, device="cpu"),
        checkpoint_path=tmp_path / "model.pt",
    )

    assert len(history) == 1
    assert (tmp_path / "model.pt").exists()
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    assert np.isfinite(history[0]["train_mse"])
