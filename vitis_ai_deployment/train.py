"""Train ECGResUNet1D, with an optional Vitis AI Quantization-Aware Training pass.

Two modes:

  float   -- ordinary full-precision training (MSE denoising loss), with a
             real record-level train/validation split, best-checkpoint
             selection by validation PRD, and early stopping. Always run
             this first; QAT fine-tunes from a converged float checkpoint,
             it does not train from scratch well.

  qat     -- loads a float checkpoint and fine-tunes it through
             pytorch_nndct's QatProcessor, which inserts fake-quantization
             ops matching the DPU's INT8 arithmetic into the training graph.
             This directly answers "do a quantized training" rather than
             relying on post-training quantization (quantize.py) alone; PTQ
             calibration is still run afterwards as a sanity check, but QAT
             tends to recover more accuracy for a UNet-style skip-heavy graph.

The `qat` mode requires the Vitis AI PyTorch quantizer (pytorch_nndct) to be
installed, which in turn requires the version of PyTorch that Vitis AI 3.5
ships (torch 1.13.1 in the vitis-ai-pytorch-cpu/gpu Docker image). This
script has not been run inside that container in this workspace -- verify
the pytorch_nndct API shape against your installed version's docs before
trusting the exact QatProcessor call signature below; Xilinx has changed
this API across releases.

Usage:
    python train.py --mode float --epochs 50 \
        --clean-dir data/clean --noise-dir mit-bih-noise-stress-test-database-1.0.0 \
        --checkpoint checkpoints/ecg_resunet1d.pt \
        --progress runs/ecg_resunet1d_progress.json
    python train.py --mode qat --epochs 5 \
        --checkpoint checkpoints/ecg_resunet1d.pt \
        --qat-checkpoint checkpoints/ecg_resunet1d_qat.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import ECG512Dataset, default_train_val_split
from model import INPUT_LENGTH, ECGResUNet1D, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.metrics import mse as compute_mse
from src.training.metrics import prd as compute_prd
from src.training.metrics import snr_improvement as compute_snr_improvement


class PlaceholderECGDataset(Dataset):
    """Synthetic fallback dataset, used only when --clean-dir/--noise-dir are
    not supplied. Prefer ECG512Dataset (dataset.py), which wires real MIT-BIH
    clean records and nstdb noise records through this repo's existing
    windowing/noise-mix pipeline at 512 samples.
    """

    def __init__(self, num_samples: int = 512):
        self.num_samples = num_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        clean = torch.randn(1, INPUT_LENGTH) * 0.1
        noisy = clean + torch.randn(1, INPUT_LENGTH) * 0.05
        return noisy, clean


def build_datasets(args) -> tuple[Dataset, Dataset | None]:
    """Build (train, val) datasets, split by record so validation windows
    never come from a signal the model trained on.
    """
    if not (args.clean_dir and args.noise_dir):
        print(
            "WARNING: --clean-dir/--noise-dir not given; training on synthetic "
            "placeholder data. Results are not meaningful for deployment."
        )
        return PlaceholderECGDataset(), None

    all_records = args.records.split(",") if args.records else None
    if args.val_records:
        val_records = args.val_records.split(",")
        train_records = [r for r in (all_records or []) if r not in val_records] or None
        if train_records is None:
            # No explicit --records given: use the full default list minus val_records.
            from src.data.fetch_dataset import DEFAULT_CLEAN_RECORDS

            train_records = [r for r in DEFAULT_CLEAN_RECORDS if r not in val_records]
    else:
        train_records, val_records = default_train_val_split(all_records)

    print(f"Train records ({len(train_records)}): {train_records}")
    print(f"Val records ({len(val_records)}): {val_records}")

    train_dataset = ECG512Dataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        records=train_records,
        deterministic=True,
        seed=42,
        download_missing=args.download_missing,
    )
    val_dataset = None
    if val_records:
        val_dataset = ECG512Dataset(
            clean_dir=args.clean_dir,
            noise_dir=args.noise_dir,
            records=val_records,
            deterministic=True,
            seed=1000,  # different seed stream than training, still fixed across epochs
            download_missing=args.download_missing,
        )
    return train_dataset, val_dataset


@torch.no_grad()
def evaluate(model: nn.Module, val_dataset: Dataset, device: str, batch_size: int = 32) -> dict:
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    mse_values, snr_values, prd_values = [], [], []
    for noisy, clean in loader:
        noisy_in = noisy.unsqueeze(2).to(device)
        output = model(noisy_in).squeeze(2).cpu()  # (N, 1, 512)
        for i in range(output.shape[0]):
            pred = output[i, 0].numpy()
            target = clean[i, 0].numpy()
            mse_values.append(compute_mse(target, pred))
            snr_values.append(compute_snr_improvement(target, pred))
            prd_values.append(compute_prd(target, pred))
    model.train()
    n = max(1, len(mse_values))
    return {
        "mse": sum(mse_values) / n,
        "output_snr_db": sum(snr_values) / n,
        "prd_percent": sum(prd_values) / n,
    }


def train_float(
    model: nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    epochs: int,
    device: str,
    checkpoint_path: Path,
    progress_path: Path | None,
    patience: int = 10,
    batch_size: int = 32,
    num_workers: int = 0,
) -> nn.Module:
    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )
    # weight_decay combats the overfitting seen with plain Adam (train loss
    # kept falling while val PRD plateaued/oscillated); ReduceLROnPlateau
    # shrinks the step size once val PRD stops improving, which plain Adam
    # at a fixed lr could not do, and is a common cause of a metric
    # oscillating around a plateau instead of settling.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )
    loss_fn = nn.MSELoss()

    best_prd = float("inf")
    epochs_without_improvement = 0
    history = []
    start_time = time.time()

    model.to(device).train()
    for epoch in range(epochs):
        running_loss = 0.0
        for noisy, clean in loader:
            noisy_in = noisy.unsqueeze(2).to(device)
            clean_in = clean.unsqueeze(2).to(device)
            optimizer.zero_grad()
            output = model(noisy_in)
            loss = loss_fn(output, clean_in)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)
        train_loss = running_loss / len(loader.dataset)

        record = {
            "epoch": epoch + 1,
            "total_epochs": epochs,
            "train_loss": train_loss,
            "elapsed_seconds": time.time() - start_time,
        }

        if val_dataset is not None:
            val_metrics = evaluate(model, val_dataset, device, batch_size=batch_size)
            record.update({f"val_{k}": v for k, v in val_metrics.items()})
            scheduler.step(val_metrics["prd_percent"])
            record["lr"] = optimizer.param_groups[0]["lr"]
            print(
                f"[float] epoch {epoch + 1}/{epochs} train_loss={train_loss:.6f} "
                f"val_mse={val_metrics['mse']:.6f} val_snr_db={val_metrics['output_snr_db']:.2f} "
                f"val_prd_pct={val_metrics['prd_percent']:.2f} lr={record['lr']:.2e}"
            )
            if val_metrics["prd_percent"] < best_prd:
                best_prd = val_metrics["prd_percent"]
                epochs_without_improvement = 0
                save_checkpoint(model, checkpoint_path)
                record["saved_best"] = True
            else:
                epochs_without_improvement += 1
                record["saved_best"] = False
        else:
            print(f"[float] epoch {epoch + 1}/{epochs} train_loss={train_loss:.6f}")
            save_checkpoint(model, checkpoint_path)
            record["saved_best"] = True

        history.append(record)
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "completed_epochs": epoch + 1,
                        "total_epochs": epochs,
                        "best_val_prd_percent": best_prd if val_dataset is not None else None,
                        "history": history,
                    },
                    handle,
                    indent=2,
                )

        if val_dataset is not None and epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1} (no val_prd improvement for {patience} epochs).")
            break

    if val_dataset is None:
        # No held-out set to select a "best" epoch by; final weights are already saved.
        pass
    else:
        load_checkpoint(model, checkpoint_path, map_location=device)  # reload best
    return model


def train_qat(
    float_checkpoint: Path, dataset: Dataset, epochs: int, device: str, output_dir: Path
) -> nn.Module:
    try:
        from pytorch_nndct import QatProcessor
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "QAT requires the Vitis AI PyTorch quantizer (pytorch_nndct). "
            "Run this inside the vitis-ai-pytorch-cpu/gpu Docker image."
        ) from error

    model = ECGResUNet1D()
    load_checkpoint(model, float_checkpoint, map_location=device)

    dummy_input = torch.randn(1, 1, 1, INPUT_LENGTH)
    qat_processor = QatProcessor(model, (dummy_input,), bitwidth=8, mix_bit=False)
    quantized_model = qat_processor.trainable_model()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(quantized_model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    quantized_model.to(device).train()
    for epoch in range(epochs):
        running_loss = 0.0
        for noisy, clean in loader:
            noisy = noisy.unsqueeze(2).to(device)
            clean = clean.unsqueeze(2).to(device)
            optimizer.zero_grad()
            output = quantized_model(noisy)
            loss = loss_fn(output, clean)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * noisy.size(0)
        print(f"[qat] epoch {epoch + 1}/{epochs} loss={running_loss / len(loader.dataset):.6f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    # Exports quantized.xmodel + quant config directly from the QAT graph,
    # equivalent in output to running quantize.py's PTQ 'test' pass.
    qat_processor.export_xmodel(str(output_dir), deploy_check=True)
    print(f"Exported QAT xmodel artifacts to {output_dir}")
    return quantized_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the ECG ResUNet1D denoiser.")
    parser.add_argument("--mode", choices=("float", "qat"), default="float")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--checkpoint", default="checkpoints/ecg_resunet1d.pt", type=Path)
    parser.add_argument("--qat-checkpoint", default="checkpoints/ecg_resunet1d_qat.pt", type=Path)
    parser.add_argument("--qat-output-dir", default="artifacts/qat_xmodel", type=Path)
    parser.add_argument("--progress", default=None, type=Path, help="Path to write JSON training progress.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean-dir", default=None, help="Local MIT-BIH clean record directory (e.g. data/clean).")
    parser.add_argument("--noise-dir", default=None, help="Local nstdb noise record directory.")
    parser.add_argument("--records", default=None, help="Comma-separated MIT-BIH record names (default: full list).")
    parser.add_argument("--val-records", default=None, help="Comma-separated record names held out for validation.")
    parser.add_argument("--download-missing", action="store_true", default=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    if args.mode == "float":
        train_dataset, val_dataset = build_datasets(args)
        print(f"Train dataset size: {len(train_dataset)} windows")
        if val_dataset is not None:
            print(f"Val dataset size: {len(val_dataset)} windows")

        model = ECGResUNet1D()
        model = train_float(
            model,
            train_dataset,
            val_dataset,
            args.epochs,
            args.device,
            args.checkpoint,
            args.progress,
            patience=args.patience,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        print(f"Saved best float checkpoint to {args.checkpoint}")
    else:
        args.clean_dir = args.clean_dir
        train_dataset, _ = build_datasets(args)
        quantized_model = train_qat(args.checkpoint, train_dataset, args.epochs, args.device, args.qat_output_dir)
        save_checkpoint(quantized_model, args.qat_checkpoint)
        print(f"Saved QAT checkpoint to {args.qat_checkpoint}")


if __name__ == "__main__":
    main()
