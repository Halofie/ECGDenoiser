"""Compare the classical DSP filter (Track A) against the trained ECGResUNet1D (Track B).

Runs both denoising approaches over the same held-out validation windows used
by `vitis_ai_deployment/train.py` (same noisy input, same target SNR draws
per window index) and reports MSE / SNR-improvement / PRD plus parameter
count and per-window CPU latency for each approach.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
VITIS_DIR = ROOT / "vitis_ai_deployment"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VITIS_DIR))

from src.baseline.filters import dsp_denoise_filter
from src.evaluation.evaluate import aggregate_metrics, summarize_metrics
from src.noise.noise_mix import mix_clean_and_noise
from src.preprocessing.window import extract_window

from dataset import ECG512Dataset, default_train_val_split  # noqa: E402  (vitis_ai_deployment module)
from model import ECGResUNet1D, load_checkpoint  # noqa: E402  (vitis_ai_deployment module)

SAMPLE_RATE_HZ = 360.0


def build_val_dataset(clean_dir: Path, noise_dir: Path, val_seed: int, val_records: list[str] | None) -> ECG512Dataset:
    if val_records is None:
        _, val_records = default_train_val_split()
    return ECG512Dataset(
        clean_dir=clean_dir,
        noise_dir=noise_dir,
        records=val_records,
        deterministic=True,
        seed=val_seed,
        download_missing=False,
    )


def _reconstruct_window(dataset: ECG512Dataset, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce one ECG512Dataset sample, exposing the pre-normalization physical signal.

    Mirrors ECG512Dataset.__getitem__ exactly (same noise window, same SNR
    draw, same shared clean-derived affine normalization) but also returns
    the raw physical clean/noisy windows so the DSP filter can be applied in
    physical units -- where a band-pass filter's cutoff frequencies are
    physically meaningful -- rather than after the per-window [-1, 1]
    rescaling, which anchors to this window's own clean min/max and is not a
    zero-mean signal a linear filter can be judged fairly against.
    """
    sample = dataset._samples[index]
    clean_raw = sample["clean"]
    start = sample["start"]

    max_start = max(0, dataset.noise_signal.size - dataset.window_length)
    noise_start = start % (max_start + 1)
    noise = extract_window(dataset.noise_signal, noise_start, length=dataset.window_length)

    rng = np.random.default_rng(dataset.seed + index) if dataset.deterministic else np.random.default_rng()
    target_snr = float(rng.uniform(*dataset.snr_range_db))
    noisy_raw, _alpha = mix_clean_and_noise(clean_raw, noise, target_snr)

    min_val = float(np.min(clean_raw))
    max_val = float(np.max(clean_raw))
    span = max(max_val - min_val, 1e-6)
    clean_n = ((2.0 * (clean_raw - min_val) / span) - 1.0).astype(np.float32)
    noisy_n = np.clip((2.0 * (noisy_raw - min_val) / span) - 1.0, -1.0, 1.0).astype(np.float32)
    return clean_raw, noisy_raw, clean_n, noisy_n


def run_comparison(
    checkpoint_path: str | Path,
    clean_dir: str | Path = "data/clean",
    noise_dir: str | Path = "mit-bih-noise-stress-test-database-1.0.0",
    val_seed: int = 1000,
    max_windows: int | None = None,
) -> dict:
    dataset = build_val_dataset(ROOT / clean_dir, ROOT / noise_dir, val_seed, None)

    model = ECGResUNet1D()
    load_checkpoint(model, ROOT / checkpoint_path)
    model.eval()

    n = len(dataset) if max_windows is None else min(max_windows, len(dataset))

    noisy_metrics, dsp_metrics, ai_metrics = [], [], []
    dsp_latency_s, ai_latency_s = [], []
    per_record: dict[str, dict[str, list[dict]]] = {}
    record_names = [sample["record"] for sample in dataset._samples[:n]]

    for index in range(n):
        clean_raw, noisy_raw, clean_n, noisy_n = _reconstruct_window(dataset, index)

        noisy_metrics.append(aggregate_metrics(clean_n, noisy_n))

        t0 = time.perf_counter()
        dsp_raw = dsp_denoise_filter(noisy_raw, sample_rate_hz=SAMPLE_RATE_HZ)
        dsp_latency_s.append(time.perf_counter() - t0)
        min_val = float(np.min(clean_raw))
        span = max(float(np.max(clean_raw)) - min_val, 1e-6)
        dsp_n = np.clip((2.0 * (dsp_raw - min_val) / span) - 1.0, -1.0, 1.0).astype(np.float32)
        dsp_metrics.append(aggregate_metrics(clean_n, dsp_n))

        t0 = time.perf_counter()
        with torch.inference_mode():
            model_input = torch.from_numpy(noisy_n).view(1, 1, 1, dataset.window_length)
            ai_out = model(model_input).squeeze(0).squeeze(0).squeeze(0).numpy()
        ai_latency_s.append(time.perf_counter() - t0)
        ai_metrics.append(aggregate_metrics(clean_n, ai_out))

        record = record_names[index]
        bucket = per_record.setdefault(record, {"noisy": [], "dsp": [], "ai": []})
        bucket["noisy"].append(noisy_metrics[-1])
        bucket["dsp"].append(dsp_metrics[-1])
        bucket["ai"].append(ai_metrics[-1])

    ai_params = sum(p.numel() for p in model.parameters())

    result = {
        "checkpoint": str(checkpoint_path),
        "val_seed": val_seed,
        "window_count": n,
        "window_length": dataset.window_length,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "overall": {
            "noisy": summarize_metrics(noisy_metrics),
            "dsp_filter": summarize_metrics(dsp_metrics),
            "ai_model": summarize_metrics(ai_metrics),
        },
        "per_record": {
            record: {
                "noisy": summarize_metrics(buckets["noisy"]),
                "dsp_filter": summarize_metrics(buckets["dsp"]),
                "ai_model": summarize_metrics(buckets["ai"]),
            }
            for record, buckets in per_record.items()
        },
        "efficiency": {
            "dsp_filter": {
                "trainable_parameters": 0,
                "requires_training_data": False,
                "mean_latency_ms": float(np.mean(dsp_latency_s) * 1000.0),
                "p95_latency_ms": float(np.percentile(dsp_latency_s, 95) * 1000.0),
            },
            "ai_model": {
                "trainable_parameters": int(ai_params),
                "requires_training_data": True,
                "mean_latency_ms": float(np.mean(ai_latency_s) * 1000.0),
                "p95_latency_ms": float(np.percentile(ai_latency_s, 95) * 1000.0),
            },
        },
    }
    return result


def render_markdown(result: dict) -> str:
    overall = result["overall"]
    efficiency = result["efficiency"]

    def row(label: str, m: dict) -> str:
        return f"| {label} | {m['mean_mse']:.6f} | {m['mean_prd']:.4f}% | {m['mean_snr']:.4f} dB |"

    lines = [
        "# DSP Filter (Track A) vs AI Model / ECGResUNet1D (Track B) Comparison",
        "",
        f"- Test windows: {result['window_count']} (held-out validation records, {result['window_length']}-sample windows)",
        f"- Sample rate: {result['sample_rate_hz']} Hz",
        f"- Val seed: {result['val_seed']}",
        f"- Checkpoint: `{result['checkpoint']}`",
        "",
        "## Overall Quality Metrics",
        "",
        "| Approach | Mean MSE | Mean PRD | Mean SNR |",
        "|---|---:|---:|---:|",
        row("Noisy input (no filtering)", overall["noisy"]),
        row("DSP filter (Track A)", overall["dsp_filter"]),
        row("AI model (Track B)", overall["ai_model"]),
        "",
        "## Efficiency / Deployment Comparison",
        "",
        "| Metric | DSP filter (Track A) | AI model (Track B) |",
        "|---|---:|---:|",
        f"| Trainable parameters | {efficiency['dsp_filter']['trainable_parameters']} | {efficiency['ai_model']['trainable_parameters']} |",
        f"| Requires labeled training data | {efficiency['dsp_filter']['requires_training_data']} | {efficiency['ai_model']['requires_training_data']} |",
        f"| Mean latency / window (CPU) | {efficiency['dsp_filter']['mean_latency_ms']:.4f} ms | {efficiency['ai_model']['mean_latency_ms']:.4f} ms |",
        f"| p95 latency / window (CPU) | {efficiency['dsp_filter']['p95_latency_ms']:.4f} ms | {efficiency['ai_model']['p95_latency_ms']:.4f} ms |",
        "",
        "## Per-Record Quality Metrics",
        "",
        "| Record | Approach | Mean MSE | Mean PRD | Mean SNR |",
        "|---|---|---:|---:|---:|",
    ]
    for record, buckets in sorted(result["per_record"].items()):
        lines.append(f"| {record} | Noisy | {buckets['noisy']['mean_mse']:.6f} | {buckets['noisy']['mean_prd']:.4f}% | {buckets['noisy']['mean_snr']:.4f} dB |")
        lines.append(f"| {record} | DSP filter | {buckets['dsp_filter']['mean_mse']:.6f} | {buckets['dsp_filter']['mean_prd']:.4f}% | {buckets['dsp_filter']['mean_snr']:.4f} dB |")
        lines.append(f"| {record} | AI model | {buckets['ai_model']['mean_mse']:.6f} | {buckets['ai_model']['mean_prd']:.4f}% | {buckets['ai_model']['mean_snr']:.4f} dB |")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/ecg_resunet1d.pt")
    parser.add_argument("--clean-dir", default="data/clean")
    parser.add_argument("--noise-dir", default="mit-bih-noise-stress-test-database-1.0.0")
    parser.add_argument("--val-seed", type=int, default=1000)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--out-json", default="reports/dsp_vs_ai_comparison.json")
    parser.add_argument("--out-md", default="reports/dsp_vs_ai_comparison.md")
    args = parser.parse_args()

    result = run_comparison(
        args.checkpoint,
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        val_seed=args.val_seed,
        max_windows=args.max_windows,
    )

    out_json = ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    out_md = ROOT / args.out_md
    out_md.write_text(render_markdown(result), encoding="utf-8")

    print(f"Windows evaluated: {result['window_count']}")
    print(f"DSP filter  mean PRD: {result['overall']['dsp_filter']['mean_prd']:.4f}%")
    print(f"AI model    mean PRD: {result['overall']['ai_model']['mean_prd']:.4f}%")
    print(f"Report written to {out_md}")


if __name__ == "__main__":
    main()
