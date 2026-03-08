# =============================================================================
# tests/data/test_preprocess_stats.py — Observability stats in metadata.json
# =============================================================================

import json
import pytest
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

from src.data.preprocess import run_preprocessing


# ---------------------------------------------------------------------------
# Helpers
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
    p = tmp_path / "preprocessing.yaml"
    p.write_text(_PREP_CONFIG_YAML, encoding="utf-8")
    return p


def _build_version_dir_with_nulls(
    tmp_path: Path,
    features: list[str],
    target: str,
    task_type: str = "classification",
) -> Path:
    version_dir = tmp_path / "iris" / "statver"
    version_dir.mkdir(parents=True)

    schema = {f: "float" for f in features}
    schema[target] = "string" if task_type == "classification" else "float"
    metadata = {
        "name": "iris",
        "task_type": task_type,
        "features": features,
        "target": target,
        "schema": schema,
        "constraints": {"min_rows": 1, "max_null_fraction": 1.0},
    }
    with open(version_dir / "dataset.yaml", "w") as f:
        yaml.dump(metadata, f)

    # train: 10 rows, one null in feature "a"
    train_data = {f: [float(i) for i in range(10)] for f in features}
    train_data[features[0]][0] = float("nan")
    if task_type == "classification":
        train_data[target] = ["setosa"] * 5 + ["versicolor"] * 5
    else:
        train_data[target] = [float(i) for i in range(10)]
    train_df = pd.DataFrame(train_data)

    # val / test: 4 rows, clean
    small_data = {f: [1.0, 2.0, 3.0, 4.0] for f in features}
    if task_type == "classification":
        small_data[target] = ["setosa", "setosa", "versicolor", "versicolor"]
    else:
        small_data[target] = [1.0, 2.0, 3.0, 4.0]
    small_df = pd.DataFrame(small_data)

    # top-level combined CSV (required by validate_dataset)
    all_df = pd.concat([train_df, small_df, small_df], ignore_index=True)
    all_df.to_csv(version_dir / "data.csv", index=False)

    for split, df in [("train", train_df), ("val", small_df), ("test", small_df)]:
        split_dir = version_dir / split
        split_dir.mkdir()
        df.to_csv(split_dir / "data.csv", index=False)

    return version_dir


def _load_meta(version_dir: Path) -> dict:
    with open(version_dir / "preprocessed" / "metadata.json") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# metadata.json must have a splits block
# ---------------------------------------------------------------------------

def test_metadata_contains_splits_block(tmp_path):
    version_dir = _build_version_dir_with_nulls(tmp_path, ["a", "b"], "label")
    prep_cfg = _write_prep_config(tmp_path)
    run_preprocessing("iris", "statver", prep_config_path=prep_cfg, processed_dir=tmp_path)
    meta = _load_meta(version_dir)
    assert "splits" in meta


def test_splits_block_has_train_val_test_keys(tmp_path):
    version_dir = _build_version_dir_with_nulls(tmp_path, ["a", "b"], "label")
    prep_cfg = _write_prep_config(tmp_path)
    run_preprocessing("iris", "statver", prep_config_path=prep_cfg, processed_dir=tmp_path)
    meta = _load_meta(version_dir)
    assert set(meta["splits"].keys()) == {"train", "val", "test"}


# ---------------------------------------------------------------------------
# Row counts
# ---------------------------------------------------------------------------

def test_rows_out_matches_actual_csv_row_count(tmp_path):
    version_dir = _build_version_dir_with_nulls(tmp_path, ["a", "b"], "label")
    prep_cfg = _write_prep_config(tmp_path)
    run_preprocessing("iris", "statver", prep_config_path=prep_cfg, processed_dir=tmp_path)
    meta = _load_meta(version_dir)

    for split in ["train", "val", "test"]:
        csv_rows = len(pd.read_csv(version_dir / "preprocessed" / f"{split}.csv"))
        assert meta["splits"][split]["rows_out"] == csv_rows, (
            f"[{split}] rows_out in metadata doesn't match actual CSV row count"
        )


# ---------------------------------------------------------------------------
# Null counts
# ---------------------------------------------------------------------------

def test_null_counts_captured_per_column(tmp_path):
    """Train has one null in column 'a' — null_counts_per_split must reflect that."""
    version_dir = _build_version_dir_with_nulls(tmp_path, ["a", "b"], "label")
    prep_cfg = _write_prep_config(tmp_path)
    run_preprocessing("iris", "statver", prep_config_path=prep_cfg, processed_dir=tmp_path)
    meta = _load_meta(version_dir)

    # null_counts_per_split records raw input null counts per feature
    assert meta["null_counts_per_split"]["train"]["a"] == 1
    assert meta["null_counts_per_split"]["train"]["b"] == 0


# ---------------------------------------------------------------------------
# Label distribution (classification)
# ---------------------------------------------------------------------------

def test_label_distribution_recorded_for_classification(tmp_path):
    version_dir = _build_version_dir_with_nulls(tmp_path, ["a", "b"], "label")
    prep_cfg = _write_prep_config(tmp_path)
    run_preprocessing("iris", "statver", prep_config_path=prep_cfg, processed_dir=tmp_path)
    meta = _load_meta(version_dir)

    dist = meta["splits"]["train"].get("label_distribution")
    assert dist is not None, "label_distribution missing from train split stats"
    assert "setosa" in dist
    assert "versicolor" in dist


def test_label_distribution_absent_for_regression(tmp_path):
    version_dir = _build_version_dir_with_nulls(
        tmp_path, ["a", "b"], "label", task_type="regression"
    )
    prep_cfg = _write_prep_config(tmp_path)
    run_preprocessing("iris", "statver", prep_config_path=prep_cfg, processed_dir=tmp_path)
    meta = _load_meta(version_dir)

    assert "label_distribution" not in meta["splits"]["train"]


# ---------------------------------------------------------------------------
# Feature stats (numeric columns)
# ---------------------------------------------------------------------------

def test_feature_stats_contain_mean_std_min_max(tmp_path):
    version_dir = _build_version_dir_with_nulls(tmp_path, ["a", "b"], "label")
    prep_cfg = _write_prep_config(tmp_path)
    run_preprocessing("iris", "statver", prep_config_path=prep_cfg, processed_dir=tmp_path)
    meta = _load_meta(version_dir)

    stats = meta["splits"]["train"].get("feature_stats", {})
    assert "a" in stats or "b" in stats, "feature_stats must include numeric feature columns"

    for col in ["a", "b"]:
        if col in stats:
            for key in ["mean", "std", "min", "max"]:
                assert key in stats[col], f"feature_stats['{col}'] missing '{key}'"
