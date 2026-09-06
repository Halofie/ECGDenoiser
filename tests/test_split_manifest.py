import json

from src.split.manifest import create_split_manifest, load_split_manifest


def test_create_and_load_split_manifest(tmp_path):
    records = ["100", "101", "103", "105", "106", "112", "113", "115"]
    path = tmp_path / "manifests" / "split_manifest.json"

    created = create_split_manifest(records, path, seed=42)
    loaded = load_split_manifest(path)

    assert loaded == created
    assert loaded["unit"] == "source_record"
    assert set(loaded["splits"]) == {"train", "validation", "test"}
    assigned = [record for values in loaded["splits"].values() for record in values]
    assert sorted(assigned) == sorted(records)


def test_manifest_is_valid_json(tmp_path):
    path = tmp_path / "split_manifest.json"
    create_split_manifest(["100", "101", "102", "103"], path)

    with path.open(encoding="utf-8") as handle:
        assert json.load(handle)["seed"] == 42
