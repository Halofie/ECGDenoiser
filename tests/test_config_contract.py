from pathlib import Path

import pytest

from src.config import ConfigError, load_config, validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "defaults.yaml"


def test_answered_project_configuration_is_complete():
    config = load_config(CONFIG_PATH)
    validate_config(config)

    assert config["data"]["clean"]["records"] == [
        "100", "101", "103", "105", "106", "112", "113", "115",
        "117", "121", "122", "123", "200", "201", "202", "210",
        "212", "213", "214", "219", "220", "222", "230", "231",
        "233", "234",
    ]
    assert config["data"]["window"]["stride"] == 128
    assert config["data"]["noise"]["weights"] == {"em": 0.7, "bw": 0.3, "ma": 0.0}
    assert config["splits"]["fractions"] == {"train": 0.7, "validation": 0.15, "test": 0.15}
    assert config["model"]["framework"] == "pytorch_brevitas"
    assert config["training"]["batch_size"] == 64


def test_invalid_split_fractions_are_rejected():
    config = load_config(CONFIG_PATH)
    config["splits"]["fractions"]["train"] = 0.8

    with pytest.raises(ConfigError, match="split fractions"):
        validate_config(config)
