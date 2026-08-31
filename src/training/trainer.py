from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.training.metrics import mse


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    epochs: int = 1


def train_step(cfg: TrainingConfig, batch_x: np.ndarray, batch_y: np.ndarray) -> tuple[float, int]:
    """A minimal training step placeholder used to validate the pipeline contract.

    This is intentionally not a complete optimizer implementation; it validates the
    training loop entry-point, loss calculation, and batch bookkeeping expected by
    Phase 8.
    """
    x = np.asarray(batch_x, dtype=np.float32)
    y = np.asarray(batch_y, dtype=np.float32)
    if x.shape != y.shape:
        raise ValueError(f"Batch shape mismatch: {x.shape} vs {y.shape}")

    loss = float(mse(y, x))
    return loss, 1
