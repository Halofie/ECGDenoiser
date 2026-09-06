from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.training.metrics import mse


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    epochs: int = 1
    device: str = "auto"
    early_stopping_patience: int | None = None


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


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _run_epoch(model: nn.Module, loader, optimizer, device: torch.device | None, epoch: int, phase: str) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    total_batches = len(loader)
    for batch_index, (noisy, clean, _metadata) in enumerate(loader, start=1):
        noisy = noisy.to(device)
        clean = clean.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = model(noisy)
        loss = nn.functional.mse_loss(prediction, clean)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().cpu())
        batches += 1
        if training and (batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0):
            print(
                f"Epoch {epoch} {phase}: batch {batch_index}/{total_batches} loss={float(loss.detach().cpu()):.6f}",
                flush=True,
            )
    if batches == 0:
        raise ValueError("DataLoader produced no batches")
    return total_loss / batches


def _evaluate_epoch(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_prd = 0.0
    batches = 0
    with torch.no_grad():
        for noisy, clean, _metadata in loader:
            prediction = model(noisy.to(device))
            target = clean.to(device)
            error_power = torch.sum((target - prediction) ** 2, dim=(1, 2))
            target_power = torch.sum(target ** 2, dim=(1, 2)).clamp_min(1e-12)
            total_mse += float(nn.functional.mse_loss(prediction, target).cpu())
            total_prd += float(torch.mean(torch.sqrt(error_power / target_power) * 100.0).cpu())
            batches += 1
    if batches == 0:
        raise ValueError("DataLoader produced no validation batches")
    return total_mse / batches, total_prd / batches


def fit(
    model: nn.Module,
    train_loader,
    validation_loader,
    config: TrainingConfig,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float | int]]:
    """Train a model with Adam and select checkpoints by validation PRD proxy.

    The current validation proxy uses normalized MSE, which is monotonic with PRD
    for a fixed clean-target energy. Full PRD aggregation is handled by evaluation.
    """
    if config.epochs <= 0 or config.learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    device = _resolve_device(config.device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float | int]] = []
    best_validation_prd = float("inf")
    stale_epochs = 0

    for epoch in range(1, config.epochs + 1):
        print(f"Epoch {epoch}/{config.epochs} starting on {device}", flush=True)
        train_mse = _run_epoch(model, train_loader, optimizer, device, epoch, "train")
        validation_mse, validation_prd = _evaluate_epoch(model, validation_loader, device)
        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation_mse,
            "validation_prd": validation_prd,
        }
        history.append(row)
        print(
            f"Epoch {epoch}/{config.epochs} complete: train_mse={train_mse:.6f} "
            f"validation_mse={validation_mse:.6f} validation_prd={validation_prd:.4f}",
            flush=True,
        )

        if validation_prd < best_validation_prd:
            best_validation_prd = validation_prd
            stale_epochs = 0
            if checkpoint_path is not None:
                destination = Path(checkpoint_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "validation_prd": validation_prd}, destination)
        else:
            stale_epochs += 1
            if config.early_stopping_patience is not None and stale_epochs >= config.early_stopping_patience:
                break
    return history
