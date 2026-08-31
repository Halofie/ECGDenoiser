import numpy as np

from src.evaluation.evaluate import aggregate_metrics, summarize_metrics


def test_aggregate_metrics_return_expected_fields():
    clean = np.linspace(-1.0, 1.0, 256, dtype=np.float32)
    pred = clean * 0.9

    stats = aggregate_metrics(clean, pred)

    assert set(stats.keys()) == {"mse", "snr", "prd"}
    assert np.isfinite(stats["mse"])
    assert np.isfinite(stats["snr"])
    assert np.isfinite(stats["prd"])


def test_summarize_metrics_reports_summary():
    metrics = [
        {"mse": 0.01, "snr": 15.0, "prd": 4.0},
        {"mse": 0.02, "snr": 12.0, "prd": 5.0},
    ]

    summary = summarize_metrics(metrics)
    assert summary["count"] == 2
    assert summary["mean_mse"] > 0
    assert summary["mean_snr"] > 0
    assert summary["mean_prd"] > 0
