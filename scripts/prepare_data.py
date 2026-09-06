from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config, validate_config
from src.data.fetch_dataset import acquire_clean_records, load_clean_record
from src.data.validation import validate_record
from src.split.manifest import create_split_manifest


def prepare_data(config_path: str | Path) -> dict:
    """Download, validate, and split the configured clean MIT-BIH records."""
    root = Path(config_path).resolve().parents[1]
    config = load_config(config_path)
    validate_config(config)

    clean = config["data"]["clean"]
    destination = root / clean["path"]
    records = acquire_clean_records(
        clean["records"],
        destination,
        download_missing=clean["download_missing"],
        database=clean["database"],
    )

    validated = []
    for item in records:
        record_path = Path(item["path"])
        validated.append(validate_record(record_path, channel=clean["channel"]))
        load_clean_record(record_path, channel=clean["channel"])

    split_config = config["splits"]
    fractions = split_config["fractions"]
    manifest = create_split_manifest(
        clean["records"],
        root / split_config["manifest"],
        train_fraction=fractions["train"],
        validation_fraction=fractions["validation"],
        test_fraction=fractions["test"],
        seed=split_config["seed"],
    )
    return {"records": validated, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/defaults.yaml")
    args = parser.parse_args()
    result = prepare_data(args.config)
    print(f"Validated records: {len(result['records'])}")
    print(f"Split manifest: {args.config} -> {result['manifest']['splits']}")


if __name__ == "__main__":
    main()
