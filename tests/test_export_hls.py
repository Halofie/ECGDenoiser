from pathlib import Path

import pytest

from scripts.export_hls import load_checkpoint


def test_load_checkpoint_reports_missing_checkpoint():
    with pytest.raises(FileNotFoundError, match="Train a model"):
        load_checkpoint(Path("missing-checkpoint.pt"))
