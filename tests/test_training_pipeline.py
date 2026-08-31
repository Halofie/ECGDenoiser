import numpy as np

from src.training.metrics import mse, prd, snr_improvement
from src.training.trainer import TrainingConfig, train_step


def test_training_metrics_are_numeric_and_finite():
    clean = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    pred = clean * 0.9

    assert np.isfinite(mse(clean, pred))
    assert np.isfinite(prd(clean, pred))
    assert np.isfinite(snr_improvement(clean, pred))


def test_train_step_updates_loss_and_batch_count():
    cfg = TrainingConfig(learning_rate=1e-3, epochs=1)
    batch_x = np.random.default_rng(0).normal(size=(4, 256)).astype(np.float32)
    batch_y = batch_x.copy()

    loss, batch_count = train_step(cfg, batch_x, batch_y)

    assert np.isfinite(loss)
    assert batch_count == 1
