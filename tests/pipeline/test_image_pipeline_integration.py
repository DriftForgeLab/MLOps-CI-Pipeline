# =============================================================================
# tests/pipeline/test_image_pipeline_integration.py
# =============================================================================
# End-to-end integration test: synthetic image dataset flows through all
# pipeline stages (versioning, splitting, preprocessing, training, evaluation).
# =============================================================================

import json

import numpy as np
import pytest
import yaml
from pathlib import Path
from PIL import Image

from src.data.versioning import create_dataset_version
from src.data.split import split_dataset
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

def _create_raw_image_dataset(raw_dir: Path, dataset_name: str = "test_images"):
    """Create a raw image dataset with ImageFolder structure."""
    dataset_dir = raw_dir / dataset_name
    images_dir = dataset_dir / "images"

    rng = np.random.RandomState(42)
    classes = {"class_a": 10, "class_b": 10}

    for class_name, count in classes.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            arr = rng.randint(0, 255, (8, 8, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"img_{i:03d}.png")

    meta = {
        "name": dataset_name,
        "task_type": "image_classification_cnn",
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
    with open(dataset_dir / "dataset.yaml", "w") as f:
        yaml.dump(meta, f)

    return dataset_name


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_full_image_pipeline(tmp_path):
    """End-to-end: version -> split -> preprocess on synthetic images."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    dataset_name = _create_raw_image_dataset(raw_dir)

    # Write config files
    prep_cfg = tmp_path / "preprocessing_image.yaml"
    prep_cfg.write_text(_PREP_CONFIG_YAML, encoding="utf-8")

    # 1. Versioning
    version_id = create_dataset_version(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    assert version_id is not None
    version_dir = processed_dir / dataset_name / version_id
    assert (version_dir / "dataset.yaml").exists()
    assert (version_dir / "images").exists()

    # 2. Splitting
    split_dataset(
        dataset_name=dataset_name,
        version_id=version_id,
        random_seed=42,
        processed_dir=processed_dir,
    )
    for split in ["train", "val", "test"]:
        assert (version_dir / split / "images").exists()
        assert (version_dir / split / "metadata.json").exists()

    # 3. Preprocessing
    run_image_preprocessing(
        dataset_name=dataset_name,
        version_id=version_id,
        prep_config_path=prep_cfg,
        processed_dir=processed_dir,
    )
    preprocessed_dir = version_dir / "preprocessed"
    assert (preprocessed_dir / "train.npz").exists()
    assert (preprocessed_dir / "val.npz").exists()
    assert (preprocessed_dir / "test.npz").exists()
    assert (preprocessed_dir / "feature_map.json").exists()
    assert (preprocessed_dir / "pipeline.pkl").exists()
    assert (preprocessed_dir / "metadata.json").exists()

    # Verify NPZ shapes
    train_data = np.load(preprocessed_dir / "train.npz")
    assert train_data["X"].ndim == 2  # flattened for compatibility
    assert train_data["X"].shape[1] == 16 * 16 * 3

    # Verify feature map
    with open(preprocessed_dir / "feature_map.json") as f:
        fmap = json.load(f)
    assert set(fmap["class_names"]) == {"class_a", "class_b"}
    assert fmap["num_features"] == 16 * 16 * 3
