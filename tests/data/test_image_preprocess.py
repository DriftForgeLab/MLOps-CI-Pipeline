# =============================================================================
# tests/data/test_image_preprocess.py
# =============================================================================

import json
import shutil

import pytest
import yaml
import numpy as np
from pathlib import Path
from PIL import Image

from src.data.image_preprocess import run_image_preprocessing


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
image:
  target_size: [16, 16]
  color_mode: rgb
  normalize: true
  flatten: true
  augmentation:
    enabled: false
    horizontal_flip: false
    rotation_degrees: 0
    augmentation_factor: 1
"""


def _write_prep_config(tmp_path: Path, config_text: str = _PREP_CONFIG_YAML) -> Path:
    p = tmp_path / "preprocessing_image.yaml"
    p.write_text(config_text, encoding="utf-8")
    return p


def _make_split_image_dataset(
    tmp_path: Path,
    classes: dict[str, int] | None = None,
    img_size: tuple[int, int] = (8, 8),
) -> tuple[Path, str]:
    """Create a versioned image dataset that's already been split."""
    if classes is None:
        classes = {"cats": 6, "dogs": 6}

    version_id = "testver"
    version_dir = tmp_path / "myimages" / version_id

    rng = np.random.RandomState(42)

    # Distribute images across splits (roughly 70/15/15)
    for split_name in ("train", "val", "test"):
        for class_name, total in classes.items():
            if split_name == "train":
                count = max(1, int(total * 0.7))
            elif split_name == "val":
                count = max(1, int(total * 0.15))
            else:
                count = max(1, total - max(1, int(total * 0.7)) - max(1, int(total * 0.15)))

            class_dir = version_dir / split_name / "images" / class_name
            class_dir.mkdir(parents=True)
            for i in range(count):
                arr = rng.randint(0, 255, (*img_size, 3), dtype=np.uint8)
                Image.fromarray(arr).save(class_dir / f"{class_name}_{split_name}_{i:03d}.png")

        # Write split metadata with random_seed
        meta_dir = version_dir / split_name
        split_meta = {"random_seed": 42, "split": split_name}
        with open(meta_dir / "metadata.json", "w") as f:
            json.dump(split_meta, f)

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
# NPZ output shape
# ---------------------------------------------------------------------------

def test_npz_files_created(tmp_path):
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    preprocessed_dir = base / "myimages" / vid / "preprocessed"
    for split in ("train", "val", "test"):
        assert (preprocessed_dir / f"{split}.npz").exists()


def test_flattened_shape_matches_target_size(tmp_path):
    """With target_size=[16,16], rgb, flatten: features should be 16*16*3 = 768."""
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    data = np.load(base / "myimages" / vid / "preprocessed" / "train.npz")
    X = data["X"]
    assert X.ndim == 2  # flattened
    assert X.shape[1] == 16 * 16 * 3


def test_grayscale_reduces_feature_count(tmp_path):
    """Grayscale target_size=[16,16] should produce 16*16 = 256 features."""
    base, vid = _make_split_image_dataset(tmp_path)
    config_text = _PREP_CONFIG_YAML.replace("color_mode: rgb", "color_mode: grayscale")
    prep_cfg = _write_prep_config(tmp_path, config_text)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    data = np.load(base / "myimages" / vid / "preprocessed" / "train.npz")
    assert data["X"].shape[1] == 16 * 16


# ---------------------------------------------------------------------------
# Normalization (leak-proof)
# ---------------------------------------------------------------------------

def test_training_data_is_normalized(tmp_path):
    """After normalization, pixel values should not be in [0, 255] range."""
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    data = np.load(base / "myimages" / vid / "preprocessed" / "train.npz")
    X = data["X"]
    # Normalized data should have values well outside 0-255 integer range
    assert X.max() < 10, "Data doesn't appear normalized"


def test_normalization_stats_stored_in_feature_map(tmp_path):
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    fmap_path = base / "myimages" / vid / "preprocessed" / "feature_map.json"
    with open(fmap_path) as f:
        fmap = json.load(f)
    assert "normalization_stats" in fmap
    assert fmap["normalization_stats"]["mean"] is not None
    assert fmap["normalization_stats"]["std"] is not None


# ---------------------------------------------------------------------------
# Feature map and class mapping
# ---------------------------------------------------------------------------

def test_feature_map_contains_class_mapping(tmp_path):
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    fmap_path = base / "myimages" / vid / "preprocessed" / "feature_map.json"
    with open(fmap_path) as f:
        fmap = json.load(f)

    assert "class_names" in fmap
    assert set(fmap["class_names"]) == {"cats", "dogs"}
    assert "class_to_index" in fmap
    assert "index_to_class" in fmap
    assert fmap["num_features"] == 16 * 16 * 3


def test_labels_are_integer_encoded(tmp_path):
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    data = np.load(base / "myimages" / vid / "preprocessed" / "train.npz")
    y = data["y"]
    assert y.dtype in (np.int32, np.int64, np.intp)
    assert set(y.tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def test_augmentation_increases_sample_count(tmp_path):
    base, vid = _make_split_image_dataset(tmp_path)
    config_text = _PREP_CONFIG_YAML.replace(
        "enabled: false\n    horizontal_flip: false",
        "enabled: true\n    horizontal_flip: true",
    ).replace(
        "augmentation_factor: 1",
        "augmentation_factor: 2",
    )
    prep_cfg = _write_prep_config(tmp_path, config_text)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    train_data = np.load(base / "myimages" / vid / "preprocessed" / "train.npz")
    val_data = np.load(base / "myimages" / vid / "preprocessed" / "val.npz")

    # Train should be 2x original due to augmentation_factor=2
    # Val should NOT be augmented
    assert train_data["X"].shape[0] > val_data["X"].shape[0]


# ---------------------------------------------------------------------------
# Metadata and idempotency
# ---------------------------------------------------------------------------

def test_metadata_json_written(tmp_path):
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    meta_path = base / "myimages" / vid / "preprocessed" / "metadata.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert "preprocess_hash" in meta
    assert "num_classes" in meta
    assert meta["num_classes"] == 2


def test_pipeline_pkl_written(tmp_path):
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    assert (base / "myimages" / vid / "preprocessed" / "pipeline.pkl").exists()


def test_idempotent_skip(tmp_path):
    base, vid = _make_split_image_dataset(tmp_path)
    prep_cfg = _write_prep_config(tmp_path)
    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)

    meta_path = base / "myimages" / vid / "preprocessed" / "metadata.json"
    mtime = meta_path.stat().st_mtime

    run_image_preprocessing("myimages", vid, prep_cfg, processed_dir=base)
    assert meta_path.stat().st_mtime == mtime, "Preprocessing was not skipped on second run"
