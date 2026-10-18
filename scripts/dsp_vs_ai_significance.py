"""Statistical significance test for the DSP filter vs AI model comparison.

Re-runs the same held-out windows as compare_dsp_vs_ai.py, but keeps the
full per-window PRD arrays (not just the mean) so a paired Wilcoxon
signed-rank test and a paired t-test can be reported alongside the effect
size -- the numbers a research-paper "Results and Analysis" section needs
beyond a bare mean-difference claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[0].parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_dsp_vs_ai import _reconstruct_window, build_val_dataset  # noqa: E402

VITIS_DIR = ROOT / "vitis_ai_deployment"
sys.path.insert(0, str(VITIS_DIR))
from model import ECGResUNet1D, load_checkpoint  # noqa: E402

from src.baseline.filters import dsp_denoise_filter  # noqa: E402
from src.training.metrics import mse, prd, snr_improvement  # noqa: E402

import torch  # noqa: E402


def main() -> None:
    checkpoint = ROOT / "checkpoints" / "ecg_resunet1d.pt"
    dataset = build_val_dataset(
        ROOT / "data/clean", ROOT / "mit-bih-noise-stress-test-database-1.0.0", 1000, None
    )
    model = ECGResUNet1D()
    load_checkpoint(model, checkpoint)
    model.eval()

    n = len(dataset)
    noisy_prd, dsp_prd, ai_prd = np.empty(n), np.empty(n), np.empty(n)
    noisy_mse, dsp_mse, ai_mse = np.empty(n), np.empty(n), np.empty(n)
    noisy_snr, dsp_snr, ai_snr = np.empty(n), np.empty(n), np.empty(n)

    for i in range(n):
        clean_raw, noisy_raw, clean_n, noisy_n = _reconstruct_window(dataset, i)

        noisy_prd[i] = prd(clean_n, noisy_n)
        noisy_mse[i] = mse(clean_n, noisy_n)
        noisy_snr[i] = snr_improvement(clean_n, noisy_n)

        dsp_raw = dsp_denoise_filter(noisy_raw, sample_rate_hz=360.0)
        min_val = float(np.min(clean_raw))
        span = max(float(np.max(clean_raw)) - min_val, 1e-6)
        dsp_n = np.clip((2.0 * (dsp_raw - min_val) / span) - 1.0, -1.0, 1.0).astype(np.float32)
        dsp_prd[i] = prd(clean_n, dsp_n)
        dsp_mse[i] = mse(clean_n, dsp_n)
        dsp_snr[i] = snr_improvement(clean_n, dsp_n)

        with torch.inference_mode():
            model_input = torch.from_numpy(noisy_n).view(1, 1, 1, dataset.window_length)
            ai_out = model(model_input).squeeze(0).squeeze(0).squeeze(0).numpy()
        ai_prd[i] = prd(clean_n, ai_out)
        ai_mse[i] = mse(clean_n, ai_out)
        ai_snr[i] = snr_improvement(clean_n, ai_out)

        if (i + 1) % 2000 == 0:
            print(f"{i + 1}/{n}")

    def paired_tests(a: np.ndarray, b: np.ndarray, label: str) -> dict:
        diff = a - b
        t_stat, t_p = stats.ttest_rel(a, b)
        w_stat, w_p = stats.wilcoxon(a, b)
        pooled_std = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
        cohens_d = float(np.mean(diff) / pooled_std) if pooled_std > 0 else float("nan")
        return {
            "label": label,
            "mean_diff": float(np.mean(diff)),
            "std_diff": float(np.std(diff, ddof=1)),
            "t_statistic": float(t_stat),
            "t_p_value": float(t_p),
            "wilcoxon_statistic": float(w_stat),
            "wilcoxon_p_value": float(w_p),
            "cohens_d": cohens_d,
            "n": int(a.size),
        }

    result = {
        "window_count": n,
        "descriptive": {
            "noisy": {
                "prd_mean": float(np.mean(noisy_prd)), "prd_std": float(np.std(noisy_prd, ddof=1)),
                "mse_mean": float(np.mean(noisy_mse)), "mse_std": float(np.std(noisy_mse, ddof=1)),
                "snr_mean": float(np.mean(noisy_snr)), "snr_std": float(np.std(noisy_snr, ddof=1)),
            },
            "dsp_filter": {
                "prd_mean": float(np.mean(dsp_prd)), "prd_std": float(np.std(dsp_prd, ddof=1)),
                "mse_mean": float(np.mean(dsp_mse)), "mse_std": float(np.std(dsp_mse, ddof=1)),
                "snr_mean": float(np.mean(dsp_snr)), "snr_std": float(np.std(dsp_snr, ddof=1)),
            },
            "ai_model": {
                "prd_mean": float(np.mean(ai_prd)), "prd_std": float(np.std(ai_prd, ddof=1)),
                "mse_mean": float(np.mean(ai_mse)), "mse_std": float(np.std(ai_mse, ddof=1)),
                "snr_mean": float(np.mean(ai_snr)), "snr_std": float(np.std(ai_snr, ddof=1)),
            },
        },
        "significance_tests_prd": {
            "dsp_vs_noisy": paired_tests(dsp_prd, noisy_prd, "DSP filter vs Noisy input"),
            "ai_vs_noisy": paired_tests(ai_prd, noisy_prd, "AI model vs Noisy input"),
            "ai_vs_dsp": paired_tests(ai_prd, dsp_prd, "AI model vs DSP filter"),
        },
        "fraction_windows_improved": {
            "dsp_beats_noisy": float(np.mean(dsp_prd < noisy_prd)),
            "ai_beats_noisy": float(np.mean(ai_prd < noisy_prd)),
            "ai_beats_dsp": float(np.mean(ai_prd < dsp_prd)),
        },
    }

    out_path = ROOT / "reports" / "dsp_vs_ai_significance.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
