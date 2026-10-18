"""Deployable normalization in the dataset, and the INT8-vs-float evaluation hook in quantize.py."""

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
VITIS_DIR = ROOT / "vitis_ai_deployment"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(VITIS_DIR))

from dataset import ECG512Dataset  # noqa: E402
from ecg_board import preprocess  # noqa: E402
from model import ECGResUNet1D, checkpoint_normalization, require_normalization, save_checkpoint  # noqa: E402

CLEAN_DIR = ROOT / "data" / "clean"
NOISE_DIR = ROOT / "mit-bih-noise-stress-test-database-1.0.0"
needs_data = pytest.mark.skipif(not (CLEAN_DIR / "222.dat").is_file() or not (NOISE_DIR / "em.dat").is_file(),
                                reason="MIT-BIH data not present")


def _bare_dataset(normalization):
    dataset = ECG512Dataset.__new__(ECG512Dataset)
    dataset.normalization = normalization
    return dataset


def test_deployable_normalization_uses_only_the_noisy_window():
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 12, 512)).astype(np.float32)
    noisy_a = clean + rng.normal(scale=0.3, size=512).astype(np.float32)
    clean_n, noisy_n = _bare_dataset("noisy_minmax").normalize_pair(clean, noisy_a)
    assert noisy_n.min() == pytest.approx(-1.0) and noisy_n.max() == pytest.approx(1.0)
    # Target uses the same affine as the input, so the model learns the board's exact mapping.
    affine = preprocess.fit_affine(noisy_a, "noisy_minmax")
    np.testing.assert_allclose(clean_n, preprocess.apply_affine(clean, affine, clip=False), atol=1e-6)
    # Changing the clean signal must not change the input (no oracle leakage).
    _, noisy_n2 = _bare_dataset("noisy_minmax").normalize_pair(clean * 3, noisy_a)
    np.testing.assert_array_equal(noisy_n, noisy_n2)


def test_oracle_normalization_unchanged():
    clean = np.linspace(-2, 2, 512).astype(np.float32)
    clean_n, _ = _bare_dataset("clean_oracle").normalize_pair(clean, clean * 5)
    assert clean_n.min() == pytest.approx(-1.0) and clean_n.max() == pytest.approx(1.0)


def test_checkpoint_records_and_enforces_normalization(tmp_path):
    path = tmp_path / "m.pt"
    save_checkpoint(ECGResUNet1D(), path, normalization="noisy_minmax")
    assert checkpoint_normalization(path) == "noisy_minmax"
    require_normalization(path, "noisy_minmax")
    with pytest.raises(ValueError, match="trained with normalization='noisy_minmax'"):
        require_normalization(path, "clean_oracle")
    legacy = tmp_path / "legacy.pt"
    torch.save({"model_state_dict": ECGResUNet1D().state_dict()}, legacy)
    assert checkpoint_normalization(legacy) == "clean_oracle"


@needs_data
def test_quantize_evaluate_hook_with_float_stand_in(tmp_path):
    import quantize

    model = ECGResUNet1D().eval()
    args = types.SimpleNamespace(normalization="noisy_minmax", clean_dir=str(CLEAN_DIR), noise_dir=str(NOISE_DIR),
                                 download_missing=False, evaluate=4, device="cpu", output_dir=tmp_path)
    results = quantize.evaluate_quantized(args, model, model)
    assert results["num_windows"] == 4
    assert results["int8_minus_float_prd_points"] == pytest.approx(0.0, abs=1e-6)
    assert (tmp_path / "quant_eval.json").is_file()
