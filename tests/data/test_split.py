# =============================================================================
# tests/data/test_split.py
# =============================================================================

import pytest
import pandas as pd
import yaml
from pathlib import Path

from src.data.split import split_dataset


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_raw_dataset(tmp_path: Path, n_per_class: int = 30) -> tuple[Path, str]:
    """Create a versioned dataset directory with a balanced 3-class CSV."""
    version_id = "testver"
    version_dir = tmp_path / "mydata" / version_id
    version_dir.mkdir(parents=True)

    rows = []
    for label in ["setosa", "versicolor", "virginica"]:
        for i in range(n_per_class):
            rows.append({"feature": float(i), "species": label})
    df = pd.DataFrame(rows)
    df.to_csv(version_dir / "data.csv", index=False)

    meta = {
        "name": "mydata",
        "task_type": "classification",
        "features": ["feature"],
        "target": "species",
        "schema": {"feature": "float", "species": "string"},
    }
    with open(version_dir / "dataset.yaml", "w") as f:
        yaml.dump(meta, f)

    return tmp_path, version_id


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_split_is_deterministic(tmp_path):
    """Same seed must produce identical row content across two runs."""
    base, vid = _make_raw_dataset(tmp_path)
    split_dataset("mydata", vid, random_seed=42, processed_dir=base)

    train_first = pd.read_csv(base / "mydata" / vid / "train" / "data.csv")

    # Remove split dirs and rerun
    import shutil
    for split in ["train", "val", "test"]:
        shutil.rmtree(base / "mydata" / vid / split)

    split_dataset("mydata", vid, random_seed=42, processed_dir=base)
    train_second = pd.read_csv(base / "mydata" / vid / "train" / "data.csv")

    pd.testing.assert_frame_equal(
        train_first.reset_index(drop=True),
        train_second.reset_index(drop=True),
    )


def test_different_seeds_produce_different_splits(tmp_path):
    base, vid = _make_raw_dataset(tmp_path)
    split_dataset("mydata", vid, random_seed=42, processed_dir=base)
    train_42 = pd.read_csv(base / "mydata" / vid / "train" / "data.csv")

    import shutil
    for split in ["train", "val", "test"]:
        shutil.rmtree(base / "mydata" / vid / split)

    split_dataset("mydata", vid, random_seed=99, processed_dir=base)
    train_99 = pd.read_csv(base / "mydata" / vid / "train" / "data.csv")

    assert not train_42.equals(train_99)


# ---------------------------------------------------------------------------
# Stratification (classification)
# ---------------------------------------------------------------------------

def test_all_classes_present_in_train(tmp_path):
    base, vid = _make_raw_dataset(tmp_path, n_per_class=30)
    split_dataset("mydata", vid, random_seed=42, processed_dir=base)

    train_df = pd.read_csv(base / "mydata" / vid / "train" / "data.csv")
    classes_in_train = set(train_df["species"].unique())
    assert classes_in_train == {"setosa", "versicolor", "virginica"}


def test_all_classes_present_in_val(tmp_path):
    base, vid = _make_raw_dataset(tmp_path, n_per_class=30)
    split_dataset("mydata", vid, random_seed=42, processed_dir=base)

    val_df = pd.read_csv(base / "mydata" / vid / "val" / "data.csv")
    classes_in_val = set(val_df["species"].unique())
    assert classes_in_val == {"setosa", "versicolor", "virginica"}


def test_all_classes_present_in_test(tmp_path):
    base, vid = _make_raw_dataset(tmp_path, n_per_class=30)
    split_dataset("mydata", vid, random_seed=42, processed_dir=base)

    test_df = pd.read_csv(base / "mydata" / vid / "test" / "data.csv")
    classes_in_test = set(test_df["species"].unique())
    assert classes_in_test == {"setosa", "versicolor", "virginica"}


def test_class_proportions_are_balanced_across_splits(tmp_path):
    """Each class must appear in roughly equal proportions across train/val/test."""
    base, vid = _make_raw_dataset(tmp_path, n_per_class=30)
    split_dataset("mydata", vid, random_seed=42, processed_dir=base)

    for split in ["train", "val", "test"]:
        df = pd.read_csv(base / "mydata" / vid / split / "data.csv")
        counts = df["species"].value_counts(normalize=True)
        for label in ["setosa", "versicolor", "virginica"]:
            proportion = counts.get(label, 0.0)
            assert abs(proportion - 1 / 3) < 0.10, (
                f"[{split}] class '{label}' proportion {proportion:.2%} "
                "deviates >10% from expected 33%"
            )


# ---------------------------------------------------------------------------
# label_distribution written to dataset.yaml
# ---------------------------------------------------------------------------

def test_label_distribution_written_to_dataset_yaml(tmp_path):
    base, vid = _make_raw_dataset(tmp_path, n_per_class=30)
    split_dataset("mydata", vid, random_seed=42, processed_dir=base)

    yaml_path = base / "mydata" / vid / "dataset.yaml"
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    assert "label_distribution" in meta.get("split", {}), (
        "dataset.yaml split block must include label_distribution after splitting"
    )
    dist = meta["split"]["label_distribution"]
    for split_name in ["train", "val", "test"]:
        assert split_name in dist
        for label in ["setosa", "versicolor", "virginica"]:
            assert label in dist[split_name], (
                f"label '{label}' missing from split '{split_name}' distribution"
            )
