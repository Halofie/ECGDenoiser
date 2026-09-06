from pathlib import Path

import numpy as np

from scripts.train_model import set_seed


def test_training_entrypoint_seed_is_reproducible():
    set_seed(42)
    first = np.random.random(4)
    set_seed(42)
    second = np.random.random(4)

    assert np.array_equal(first, second)


def test_required_entrypoints_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "prepare_data.py").exists()
    assert (root / "scripts" / "train_model.py").exists()
