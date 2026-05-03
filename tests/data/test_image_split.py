# =============================================================================
# tests/data/test_image_split.py
# =============================================================================

import json
import shutil

import pytest
import yaml
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.image_split import split_image_dataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_dataset(
    tmp_path: Path,
    classes: dict[str, int] | None = None,
) -> tuple[Path, str]:
    """Create a versioned image dataset directory with synthetic images."""
    if classes is None:
        classes = {"cats": 10, "dogs": 10}

    version_id = "testver"
    version_dir = tmp_path / "myimages" / version_id
    images_dir = version_dir / "images"

    rng = np.random.RandomState(42)
    for class_name, count in classes.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            arr = rng.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"img_{i:03d}.png")

    meta = {
        "name": "myimages",
        "task_type": "image_classification",
        "features": [],
        "target": "label",
        "schema": {},
        "image_properties": {
            "expected_formats": [".png"],
            "min_images_per_class": 1,
        },
        "constraints": {
            "min_rows": 2,
            "max_null_fraction": 0.0,
            "label_classes": sorted(classes.keys()),
        },
    }
    with open(version_dir / "dataset.yaml", "w") as f:
        yaml.dump(meta, f)

    return tmp_path, version_id


def _make_prefixed_image_dataset(
    tmp_path: Path,
    classes: dict[str, tuple[int, int]] | None = None,
) -> tuple[Path, str]:
    """Create a versioned dataset with explicit train_/test_ filename prefixes."""
    if classes is None:
        classes = {"cats": (6, 4), "dogs": (6, 4)}

    version_id = "testver"
    version_dir = tmp_path / "myimages" / version_id
    images_dir = version_dir / "images"

    rng = np.random.RandomState(7)
    for class_name, (train_count, test_count) in classes.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(train_count):
            arr = rng.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"train_{i:03d}.png")
        for i in range(test_count):
            arr = rng.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"test_{i:03d}.png")

    meta = {
        "name": "myimages",
        "task_type": "image_classification",
        "features": [],
        "target": "label",
        "schema": {},
        "image_properties": {
            "expected_formats": [".png"],
            "min_images_per_class": 1,
        },
        "constraints": {
            "min_rows": 2,
            "max_null_fraction": 0.0,
            "label_classes": sorted(classes.keys()),
        },
    }
    with open(version_dir / "dataset.yaml", "w") as f:
        yaml.dump(meta, f)

    return tmp_path, version_id


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_split_is_deterministic(tmp_path):
    """Same seed must produce identical splits across two runs."""
    base, vid = _make_image_dataset(tmp_path)
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)

    train_files_1 = sorted(
        f.name for f in (base / "myimages" / vid / "train" / "images").rglob("*.png")
    )

    # Remove splits and re-run
    for split in ["train", "val", "test"]:
        shutil.rmtree(base / "myimages" / vid / split)

    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)
    train_files_2 = sorted(
        f.name for f in (base / "myimages" / vid / "train" / "images").rglob("*.png")
    )

    assert train_files_1 == train_files_2


def test_different_seeds_produce_different_splits(tmp_path):
    base, vid = _make_image_dataset(tmp_path)
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)
    train_42 = sorted(
        f.name for f in (base / "myimages" / vid / "train" / "images").rglob("*.png")
    )

    for split in ["train", "val", "test"]:
        shutil.rmtree(base / "myimages" / vid / split)

    split_image_dataset("myimages", vid, random_seed=99, processed_dir=base)
    train_99 = sorted(
        f.name for f in (base / "myimages" / vid / "train" / "images").rglob("*.png")
    )

    assert train_42 != train_99


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def test_all_classes_present_in_all_splits(tmp_path):
    base, vid = _make_image_dataset(tmp_path, {"cats": 10, "dogs": 10})
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)

    for split in ["train", "val", "test"]:
        split_images_dir = base / "myimages" / vid / split / "images"
        class_dirs = {d.name for d in split_images_dir.iterdir() if d.is_dir()}
        assert class_dirs == {"cats", "dogs"}, f"Missing class in {split}"


def test_class_proportions_preserved(tmp_path):
    base, vid = _make_image_dataset(tmp_path, {"cats": 20, "dogs": 20})
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)

    for split in ["train", "val", "test"]:
        split_images_dir = base / "myimages" / vid / split / "images"
        cats_count = len(list((split_images_dir / "cats").iterdir()))
        dogs_count = len(list((split_images_dir / "dogs").iterdir()))
        total = cats_count + dogs_count
        proportion = cats_count / total
        assert abs(proportion - 0.5) < 0.15, (
            f"[{split}] cat proportion {proportion:.2%} deviates >15% from expected 50%"
        )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def test_per_split_metadata_written(tmp_path):
    base, vid = _make_image_dataset(tmp_path)
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)

    for split in ["train", "val", "test"]:
        meta_path = base / "myimages" / vid / split / "metadata.json"
        assert meta_path.exists(), f"metadata.json missing for {split}"
        with open(meta_path) as f:
            meta = json.load(f)
        assert "rows" in meta
        assert "label_distribution" in meta
        assert meta["split"] == split


def test_dataset_yaml_updated_with_split_block(tmp_path):
    base, vid = _make_image_dataset(tmp_path)
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)

    yaml_path = base / "myimages" / vid / "dataset.yaml"
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)

    assert "split" in meta
    assert "label_distribution" in meta["split"]
    for split in ["train", "val", "test"]:
        assert split in meta["split"]["label_distribution"]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def test_idempotent_skip(tmp_path):
    """Second call should skip when outputs already exist."""
    base, vid = _make_image_dataset(tmp_path)
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)

    meta_path = base / "myimages" / vid / "train" / "metadata.json"
    mtime = meta_path.stat().st_mtime

    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)
    assert meta_path.stat().st_mtime == mtime


# ---------------------------------------------------------------------------
# Total image count preserved
# ---------------------------------------------------------------------------

def test_total_images_preserved(tmp_path):
    """All images must appear in exactly one split."""
    classes = {"cats": 10, "dogs": 10}
    base, vid = _make_image_dataset(tmp_path, classes)
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)

    total = 0
    for split in ["train", "val", "test"]:
        split_images_dir = base / "myimages" / vid / split / "images"
        total += len(list(split_images_dir.rglob("*.png")))

    assert total == sum(classes.values())


def test_preserves_prefixed_official_test_boundary(tmp_path):
    """test_ images must remain in test when the dataset already encodes a test split."""
    base, vid = _make_prefixed_image_dataset(tmp_path)
    split_image_dataset("myimages", vid, random_seed=42, processed_dir=base)

    version_dir = base / "myimages" / vid
    train_files = sorted(f.name for f in (version_dir / "train" / "images").rglob("*.png"))
    val_files = sorted(f.name for f in (version_dir / "val" / "images").rglob("*.png"))
    test_files = sorted(f.name for f in (version_dir / "test" / "images").rglob("*.png"))

    assert train_files
    assert val_files
    assert test_files
    assert all(name.startswith("train_") for name in train_files)
    assert all(name.startswith("train_") for name in val_files)
    assert all(name.startswith("test_") for name in test_files)

    with open(version_dir / "dataset.yaml") as f:
        metadata = yaml.safe_load(f)

    assert metadata["split"]["train"] + metadata["split"]["val"] == 12
    assert metadata["split"]["test"] == 8
    assert metadata["split"]["preserved_original_test"] is True
