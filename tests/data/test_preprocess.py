# =============================================================================
# tests/data/test_preprocess.py
# =============================================================================

import json
import yaml
import pytest
import pandas as pd
from pathlib import Path

from src.data.preprocess import preprocess, run_preprocessing


# ---------------------------------------------------------------------------
# preprocess() — pure function tests
# ---------------------------------------------------------------------------

def test_preprocess_selects_only_feature_and_target_columns():
    df = pd.DataFrame({
        "a": [1, 2], "b": [3, 4], "c": [5, 6], "label": [0, 1]
    })
    result = preprocess(df, target="label", features=["a", "b"])
    assert list(result.columns) == ["a", "b", "label"]


def test_preprocess_does_not_include_unrequested_columns():
    df = pd.DataFrame({"a": [1], "b": [2], "extra": [99], "label": [0]})
    result = preprocess(df, target="label", features=["a"])
    assert "extra" not in result.columns
    assert "b" not in result.columns


def test_preprocess_returns_a_copy_not_a_view():
    df = pd.DataFrame({"a": [1, 2], "label": [0, 1]})
    result = preprocess(df, target="label", features=["a"])
    result["a"] = 999
    assert df["a"].tolist() == [1, 2]  # original must be untouched


def test_preprocess_preserves_row_count():
    df = pd.DataFrame({"x": range(50), "y": range(50)})
    result = preprocess(df, target="y", features=["x"])
    assert len(result) == 50


# ---------------------------------------------------------------------------
# run_preprocessing() — helpers and metadata/idempotency tests
# ---------------------------------------------------------------------------

_PREP_CONFIG_YAML = """\
fail_on_nulls: true
min_rows: 1
validate_types: false
validate_labels: true
validate_on_skip: false
numeric_features: null
categorical_features: null
encoding:
  enabled: false
  strategy: onehot
  handle_unknown: ignore
  min_frequency: null
scaling:
  enabled: false
  strategy: standard
missing_values:
  policy: passthrough
  numeric_strategy: mean
  categorical_strategy: most_frequent
  fill_value: null
"""


def _write_prep_config(tmp_path: Path) -> Path:
    """Write a minimal passthrough preprocessing config and return its path."""
    p = tmp_path / "preprocessing.yaml"
    p.write_text(_PREP_CONFIG_YAML, encoding="utf-8")
    return p


def _build_version_dir(
    tmp_path: Path,
    features: list[str],
    target: str,
    version_id: str = "abc123",
) -> Path:
    """Helper: create a minimal versioned dataset directory structure."""
    version_dir = tmp_path / "iris" / version_id
    version_dir.mkdir(parents=True)

    schema = {col: "float" for col in features}
    schema[target] = "string"
    metadata = {
        "name": "iris",
        "task_type": "classification",
        "features": features,
        "target": target,
        "schema": schema,
        "constraints": {"min_rows": 1, "max_null_fraction": 0.0},
    }
    with open(version_dir / "dataset.yaml", "w") as f:
        yaml.dump(metadata, f)

    row: dict = {col: [1.0] for col in features}
    row[target] = ["setosa"]
    df = pd.DataFrame(row)

    # top-level combined CSV (required by validate_dataset)
    df.to_csv(version_dir / "data.csv", index=False)

    for split in ["train", "val", "test"]:
        split_dir = version_dir / split
        split_dir.mkdir()
        df.to_csv(split_dir / "data.csv", index=False)

    return version_dir


def test_metadata_json_is_written_after_preprocessing(tmp_path):
    """After run_preprocessing(), preprocessed/metadata.json must exist."""
    version_dir = _build_version_dir(tmp_path, features=["a", "b"], target="label")
    prep_cfg = _write_prep_config(tmp_path)

    run_preprocessing("iris", "abc123", prep_config_path=prep_cfg, processed_dir=tmp_path)

    assert (version_dir / "preprocessed" / "metadata.json").exists()


def test_metadata_json_contains_required_keys(tmp_path):
    """metadata.json must include preprocess_hash, output_features, target, and preprocessed_at."""
    _build_version_dir(tmp_path, features=["a", "b"], target="label")
    prep_cfg = _write_prep_config(tmp_path)

    run_preprocessing("iris", "abc123", prep_config_path=prep_cfg, processed_dir=tmp_path)

    meta_path = tmp_path / "iris" / "abc123" / "preprocessed" / "metadata.json"
    with open(meta_path) as f:
        meta = json.load(f)

    assert "preprocess_hash" in meta
    assert "preprocessed_at" in meta
    assert "output_features" in meta
    assert "target" in meta
    assert meta["output_features"] == ["a", "b"]
    assert meta["target"] == "label"


def test_run_preprocessing_skips_when_hash_matches(tmp_path):
    """If config unchanged, second call must skip and not overwrite metadata.json."""
    _build_version_dir(tmp_path, features=["a", "b"], target="label")
    prep_cfg = _write_prep_config(tmp_path)

    run_preprocessing("iris", "abc123", prep_config_path=prep_cfg, processed_dir=tmp_path)

    meta_path = tmp_path / "iris" / "abc123" / "preprocessed" / "metadata.json"
    mtime_after_first_run = meta_path.stat().st_mtime

    run_preprocessing("iris", "abc123", prep_config_path=prep_cfg, processed_dir=tmp_path)

    assert meta_path.stat().st_mtime == mtime_after_first_run, (
        "metadata.json was overwritten on second run with identical config — skip logic broken"
    )


def test_run_preprocessing_reruns_when_feature_list_changes(tmp_path):
    """If features change, preprocessing must rerun even if CSV files exist."""
    version_dir = _build_version_dir(tmp_path, features=["a", "b"], target="label")
    prep_cfg = _write_prep_config(tmp_path)
    run_preprocessing("iris", "abc123", prep_config_path=prep_cfg, processed_dir=tmp_path)

    # Change the feature list in dataset.yaml (drop column b)
    yaml_path = version_dir / "dataset.yaml"
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    meta["features"] = ["a"]
    meta["schema"] = {"a": "float", "label": "string"}
    with open(yaml_path, "w") as f:
        yaml.dump(meta, f)

    # Shrink all CSVs to match
    df_small = pd.DataFrame({"a": [1.0], "label": ["setosa"]})
    df_small.to_csv(version_dir / "data.csv", index=False)
    for split in ["train", "val", "test"]:
        df_small.to_csv(version_dir / split / "data.csv", index=False)

    run_preprocessing("iris", "abc123", prep_config_path=prep_cfg, processed_dir=tmp_path)

    result_df = pd.read_csv(version_dir / "preprocessed" / "train.csv")
    assert list(result_df.columns) == ["a", "label"]
    assert "b" not in result_df.columns
