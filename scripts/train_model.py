from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, validate_config
from src.data.dataset import ECGDenoisingDataset, create_dataloader, load_noise_records
from src.data.fetch_dataset import load_clean_record
from src.fpga.deployment import load_calibration
from src.models.quant_autoencoder import ECGDenoiseAutoencoder
from src.split.manifest import load_split_manifest
from src.training.trainer import TrainingConfig, fit


def set_seed(seed: int) -> None:
    """Set reproducibility controls for deterministic training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_from_config(
    config_path: str | Path,
    checkpoint_path: str | Path,
    epochs: int | None = None,
    progress_path: str | Path | None = None,
    disable_early_stopping: bool = False,
) -> list[dict]:
    """Build configured datasets and train the quantized autoencoder."""
    config_path = Path(config_path)
    root = config_path.resolve().parents[1]
    config = load_config(config_path)
    validate_config(config)
    seed = config["data"]["noise_injection"]["seed"]
    set_seed(seed)

    clean_config = config["data"]["clean"]
    clean_by_name = {}
    for record_name in clean_config["records"]:
        clean_by_name[record_name] = load_clean_record(
            root / clean_config["path"] / record_name,
            channel=clean_config["channel"],
        )

    manifest = load_split_manifest(root / config["splits"]["manifest"])
    noise_config = config["data"]["noise"]
    noise_records = load_noise_records(root / noise_config["path"], noise_config["types"])
    calibration_path = root / config["fpga"]["deployment_input"]["calibration_artifact"]
    calibration = load_calibration(calibration_path)
    weights = noise_config["weights"]
    window = config["data"]["window"]
    snr_range = tuple(config["data"]["noise_injection"]["snr_range_db"])

    datasets = {}
    for split_name in ("train", "validation", "test"):
        datasets[split_name] = ECGDenoisingDataset(
            [clean_by_name[name] for name in manifest["splits"][split_name]],
            noise_records,
            weights,
            snr_range_db=snr_range,
            window_length=window["length"],
            window_stride=window["stride"],
            seed=seed,
            deterministic=split_name != "test",
            calibration=calibration,
        )

    batch_size = config["training"]["batch_size"]
    train_loader = create_dataloader(datasets["train"], batch_size, shuffle=True)
    validation_loader = create_dataloader(datasets["validation"], batch_size, shuffle=False)
    training = config["training"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return fit(
        ECGDenoiseAutoencoder(),
        train_loader,
        validation_loader,
        TrainingConfig(
            learning_rate=training["learning_rate"],
            epochs=epochs if epochs is not None else training["max_epochs"],
            device=device,
            early_stopping_patience=None if disable_early_stopping else training["early_stopping"]["patience"],
            progress_path=progress_path,
        ),
        checkpoint_path=checkpoint_path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/defaults.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--epochs", type=int, default=None, help="Override configured epoch count for an isolated experiment")
    parser.add_argument("--progress", default=None, help="Write live epoch progress JSON to this path")
    parser.add_argument("--disable-early-stopping", action="store_true", help="Run every requested epoch")
    args = parser.parse_args()
    history = train_from_config(
        args.config,
        args.checkpoint,
        epochs=args.epochs,
        progress_path=args.progress,
        disable_early_stopping=args.disable_early_stopping,
    )
    print(f"Completed epochs: {len(history)}")
    print(f"Best validation PRD: {min(row['validation_prd'] for row in history):.4f}")


if __name__ == "__main__":
    main()
