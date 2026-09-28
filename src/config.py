from __future__ import annotations

from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when the project configuration violates the experiment contract."""


def load_config(path: str | Path) -> dict:
    """Load a YAML config file.

    Phase 1 only provides the schema and config loader; project implementation
    and dataset/model logic will be added in later phases.
    """
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def validate_config(config: dict) -> None:
    """Validate the answered project inputs before data or model execution."""
    required_sections = {"data", "splits", "model", "training", "fpga"}
    missing = required_sections.difference(config)
    if missing:
        raise ConfigError(f"Missing configuration sections: {sorted(missing)}")

    signal = config["data"]["signal"]
    window = config["data"]["window"]
    noise = config["data"]["noise"]
    if signal["sample_rate_hz"] != 360 or window["length"] != 256:
        raise ConfigError("The project requires 360 Hz signals and 256-sample windows")
    if window["stride"] != 128 or window["drop_incomplete"] is not True:
        raise ConfigError("The project requires 50% overlap and discarded incomplete windows")
    if signal["normalization"] != "per_window_minmax_minus1_1":
        raise ConfigError("The project requires per-window min-max normalization to [-1, 1]")
    if noise["weights"] != {"em": 0.7, "bw": 0.3, "ma": 0.0}:
        raise ConfigError("Noise weights must be EM=0.70, BW=0.30, MA=0.00")

    fractions = config["splits"]["fractions"]
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ConfigError("The split fractions must sum to 1.0")
    if config["splits"]["manifest"] != "manifests/split_manifest.json":
        raise ConfigError("The split manifest must be manifests/split_manifest.json")

    model = config["model"]
    if model["framework"] != "pytorch_brevitas":
        raise ConfigError("The selected model framework must be PyTorch with Brevitas")
    if model["output_activation"] != "linear":
        raise ConfigError("The reconstruction output must use a linear activation")
    if model["weight_precision"] != "ap_fixed<8,1>" or model["activation_precision"] != "ap_fixed<8,2>":
        raise ConfigError("Model precision must match the approved fixed-point contract")

    training = config["training"]
    if training["optimizer"] != "Adam" or training["learning_rate"] != 0.001:
        raise ConfigError("Training must use Adam with learning rate 0.001")
    if training["batch_size"] != 64 or training["max_epochs"] != 50:
        raise ConfigError("Training must use batch size 64 and maximum 50 epochs")
    if training["early_stopping"]["patience"] != 10 or training["checkpoint_metric"] != "validation_prd":
        raise ConfigError("Training must select by validation PRD with patience 10")

    fpga = config["fpga"]
    deployment_input = fpga.get("deployment_input")
    if not isinstance(deployment_input, dict):
        raise ConfigError("The FPGA deployment input contract is required")
    if deployment_input.get("normalization") != "fixed_affine_minus1_1":
        raise ConfigError("FPGA input normalization must use fixed affine scaling to [-1, 1]")
    if deployment_input.get("calibration_artifact") != "artifacts/input_calibration.json":
        raise ConfigError("The FPGA calibration artifact path is fixed by the deployment contract")
    if deployment_input.get("sample_format") != "int16" or deployment_input.get("output_format") != "int16":
        raise ConfigError("The FPGA UART contract requires int16 input and output samples")
    if deployment_input.get("sample_rate_hz") != 360 or deployment_input.get("window_length") != 256:
        raise ConfigError("The FPGA input contract requires 360 Hz and 256-sample windows")
    if deployment_input.get("window_stride") != 128:
        raise ConfigError("The FPGA input contract requires a 128-sample window stride")
    if deployment_input.get("uart_baud_rate") != 115200:
        raise ConfigError("The FPGA UART contract requires 115200 baud")
    if deployment_input.get("packet_crc") != "crc32":
        raise ConfigError("The FPGA UART contract requires CRC32 packet validation")
