from pathlib import Path


def test_repo_structure_exists():
    root = Path(__file__).resolve().parents[1]

    assert (root / "README.md").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "configs").is_dir()
    assert (root / "src").is_dir()
    assert (root / "tests").is_dir()


def test_default_config_loads():
    from src.config import load_config, validate_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "defaults.yaml")

    assert config["project"] == "ECG-Denoise-FPGA-1DCAE"
    assert config["data"]["window"]["length"] == 256
    assert config["training"]["learning_rate"] == 0.001
    validate_config(config)
