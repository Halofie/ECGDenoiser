"""Score a denoiser exactly the way the board runs it.

physical noisy window -> ecg_board.preprocess normalization -> model ->
inverse affine back to physical units -> metrics in the clean window's
[-1, 1] scale (the convention used by every report in reports/).

Used by scripts/eval_deployable_normalization.py (float model on the laptop)
and quantize.py --evaluate (fake-quantized INT8 model inside the Vitis AI
Docker image), so float and INT8 numbers are directly comparable.
"""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from dataset import ECG512Dataset
from ecg_board import preprocess

SNR_BINS = ((-5.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0))

# (N, 512) normalized float32 -> (N, 512) normalized float32
ModelFn = Callable[[np.ndarray], np.ndarray]


def clean_domain(values: np.ndarray, clean_raw: np.ndarray) -> np.ndarray:
    low, high = float(clean_raw.min()), float(clean_raw.max())
    return (2.0 * (values - low) / max(high - low, 1e-6) - 1.0).astype(np.float32)


def window_metrics(clean: np.ndarray, pred: np.ndarray) -> dict[str, np.ndarray]:
    """Row-wise MSE / PRD / output SNR for (N, L) arrays."""
    err = np.sum(np.square(clean - pred), axis=1)
    energy = np.maximum(np.sum(np.square(clean), axis=1), 1e-12)
    return {
        "mse": np.mean(np.square(clean - pred), axis=1),
        "prd_percent": np.sqrt(err / energy) * 100.0,
        "snr_db": 10.0 * np.log10(energy / np.maximum(err, 1e-12)),
    }


def summarize(metrics: dict[str, np.ndarray], input_snr: np.ndarray) -> dict:
    out = {name: float(np.mean(values)) for name, values in metrics.items()}
    out["prd_percent_median"] = float(np.median(metrics["prd_percent"]))
    out["prd_by_input_snr_db"] = {}
    for low, high in SNR_BINS:
        mask = (input_snr >= low) & (input_snr < high)
        if mask.any():
            out["prd_by_input_snr_db"][f"{low:+.0f}..{high:+.0f}"] = float(np.mean(metrics["prd_percent"][mask]))
    return out


class EvalSet:
    """Fixed held-out windows in physical units plus their clean-domain targets."""

    def __init__(self, dataset: ECG512Dataset, indices: Iterable[int]):
        pairs = [dataset.raw_pair(int(i)) for i in indices]
        self.dataset = dataset
        self.clean_raw = np.stack([p[0] for p in pairs]).astype(np.float32)
        self.noisy_raw = np.stack([p[1] for p in pairs]).astype(np.float32)
        self.clean_eval = np.stack([clean_domain(c, c) for c in self.clean_raw])
        noise_power = np.mean(np.square(self.noisy_raw - self.clean_raw), axis=1)
        self.input_snr = 10.0 * np.log10(np.mean(np.square(self.clean_raw), axis=1) / np.maximum(noise_power, 1e-12))

    def __len__(self) -> int:
        return len(self.clean_raw)

    def noisy_baseline(self) -> dict:
        noisy_eval = np.clip(np.stack([clean_domain(n, c) for n, c in zip(self.noisy_raw, self.clean_raw)]), -1, 1)
        return summarize(window_metrics(self.clean_eval, noisy_eval), self.input_snr)

    def evaluate(self, model_fn: ModelFn, method: str) -> dict:
        if method == "clean_oracle":
            inputs = np.stack([self.dataset_normalize_oracle(c, n) for c, n in zip(self.clean_raw, self.noisy_raw)])
            pred_eval = model_fn(inputs)  # already in the clean window's scale
        else:
            normalized = [preprocess.normalize(n, method) for n in self.noisy_raw]
            outputs = model_fn(np.stack([x for x, _ in normalized]))
            pred_eval = np.stack([
                clean_domain(preprocess.invert_affine(out, affine), clean)
                for out, (_, affine), clean in zip(outputs, normalized, self.clean_raw)
            ])
        return summarize(window_metrics(self.clean_eval, pred_eval), self.input_snr)

    def dataset_normalize_oracle(self, clean_raw: np.ndarray, noisy_raw: np.ndarray) -> np.ndarray:
        previous = self.dataset.normalization
        self.dataset.normalization = "clean_oracle"
        try:
            return self.dataset.normalize_pair(clean_raw, noisy_raw)[1]
        finally:
            self.dataset.normalization = previous


def torch_model_fn(model, device: str = "cpu", batch_size: int = 256) -> ModelFn:
    """Wrap a (N, 1, 1, 512) PyTorch model as a numpy batch function."""
    import torch

    def run(inputs: np.ndarray) -> np.ndarray:
        outputs = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(inputs[start:start + batch_size]))
                batch = batch.view(-1, 1, 1, inputs.shape[1]).to(device)
                outputs.append(model(batch).reshape(batch.shape[0], -1).cpu().numpy())
        return np.concatenate(outputs).astype(np.float32)

    return run
