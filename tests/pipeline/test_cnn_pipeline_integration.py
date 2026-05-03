# =============================================================================
# tests/pipeline/test_cnn_pipeline_integration.py
# =============================================================================
# End-to-end integration test for the CNN image classification pipeline.
# Covers: versioning → splitting → preprocessing (flatten=false) →
#         CNN training → model artifact saving → evaluation with confusion matrix.
# =============================================================================

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from PIL import Image

from src.data.versioning import create_dataset_version
from src.data.split import split_dataset
from src.data.image_preprocess import run_image_preprocessing
from src.training.image_classification_cnn.train import run_training as run_cnn_training
from src.training import TrainingResult
from src.registry.metadata import save_model_artifact
from src.evaluation.evaluate import evaluate


# ---------------------------------------------------------------------------
# Inline config fixtures
# ---------------------------------------------------------------------------

_CNN_PREP_CONFIG_YAML = """\
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
  target_size: [8, 8]
  color_mode: rgb
  normalize: true
  flatten: false
  augmentation:
    enabled: false
    horizontal_flip: false
    rotation_degrees: 0
    augmentation_factor: 1
"""

# Minimal CNN training config: 1 conv layer, 1 epoch — fast for testing.
_CNN_TRAINING_CONFIG_YAML = """\
model:
  algorithm: cnn
  architecture:
    conv_layers:
      - out_channels: 8
        kernel_size: 3
    fc_units: 16
    dropout: 0.0
  hyperparameters:
    epochs: 1
    batch_size: 4
    learning_rate: 0.001
"""

_EVALUATION_CONFIG_YAML = """\
classification:
  averaging: weighted
"""

_PROMOTION_CONFIG_YAML = """\
classification:
  rules:
    - id: accuracy_min
      metric: accuracy
      operator: ">="
      threshold: 0.0
      description: "Minimum accuracy (always passes)"
regression:
  rules: []
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_raw_image_dataset(raw_dir: Path, dataset_name: str = "cnn_test_images") -> str:
    """Create a synthetic ImageFolder dataset with two classes."""
    dataset_dir = raw_dir / dataset_name
    images_dir = dataset_dir / "images"

    rng = np.random.RandomState(0)
    classes = {"cat": 10, "dog": 10}

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


def _create_prefixed_raw_image_dataset(raw_dir: Path, dataset_name: str = "cnn_prefixed_images") -> str:
    """Create a dataset whose filenames encode an original train/test boundary."""
    dataset_dir = raw_dir / dataset_name
    images_dir = dataset_dir / "images"

    rng = np.random.RandomState(1)
    classes = {"cat": (6, 4), "dog": (6, 4)}

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


def _write_configs(tmp_path: Path) -> dict[str, Path]:
    prep_cfg = tmp_path / "preprocessing_cnn.yaml"
    train_cfg = tmp_path / "training_cnn.yaml"
    eval_cfg = tmp_path / "evaluation.yaml"
    promo_cfg = tmp_path / "promotion.yaml"

    prep_cfg.write_text(_CNN_PREP_CONFIG_YAML, encoding="utf-8")
    train_cfg.write_text(_CNN_TRAINING_CONFIG_YAML, encoding="utf-8")
    eval_cfg.write_text(_EVALUATION_CONFIG_YAML, encoding="utf-8")
    promo_cfg.write_text(_PROMOTION_CONFIG_YAML, encoding="utf-8")

    return {
        "preprocessing": prep_cfg,
        "training": train_cfg,
        "evaluation": eval_cfg,
        "promotion": promo_cfg,
    }


def _make_mock_config(dataset_name: str, processed_dir: Path, cfg_paths: dict) -> MagicMock:
    mock = MagicMock()
    mock.dataset = dataset_name
    mock.data.processed = str(processed_dir)
    mock.configs.training = str(cfg_paths["training"])
    mock.configs.evaluation = str(cfg_paths["evaluation"])
    mock.configs.promotion = str(cfg_paths["promotion"])
    mock.random_seed = 42
    mock.task_type = "image_classification_cnn"
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cnn_training_produces_valid_result(tmp_path):
    """CNN training returns a TrainingResult with a fitted model."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    dataset_name = _create_raw_image_dataset(raw_dir)
    cfg_paths = _write_configs(tmp_path)

    version_id = create_dataset_version(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    ).name
    split_dataset(dataset_name=dataset_name, version_id=version_id, random_seed=42, processed_dir=processed_dir)
    run_image_preprocessing(
        dataset_name=dataset_name,
        version_id=version_id,
        prep_config_path=cfg_paths["preprocessing"],
        processed_dir=processed_dir,
    )

    mock_config = _make_mock_config(dataset_name, processed_dir, cfg_paths)
    result = run_cnn_training(mock_config, version_id)

    assert isinstance(result, TrainingResult)
    assert result.algorithm == "cnn"
    assert result.train_rows > 0
    assert result.model is not None


def test_cnn_preprocessing_produces_4d_arrays(tmp_path):
    """CNN preprocessing (flatten=false) produces 4D arrays (N, H, W, C)."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    dataset_name = _create_raw_image_dataset(raw_dir)
    cfg_paths = _write_configs(tmp_path)

    version_id = create_dataset_version(dataset_name=dataset_name, raw_dir=raw_dir, processed_dir=processed_dir).name
    split_dataset(dataset_name=dataset_name, version_id=version_id, random_seed=42, processed_dir=processed_dir)
    run_image_preprocessing(
        dataset_name=dataset_name,
        version_id=version_id,
        prep_config_path=cfg_paths["preprocessing"],
        processed_dir=processed_dir,
    )

    preprocessed_dir = processed_dir / dataset_name / version_id / "preprocessed"
    train_data = np.load(preprocessed_dir / "train.npz")

    assert train_data["X"].ndim == 4, "CNN preprocessing must produce 4D arrays"
    assert train_data["X"].shape[1:] == (8, 8, 3)


def test_cnn_model_artifact_is_saved(tmp_path):
    """Training result is persisted as model.pt with accompanying metadata.json."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"
    dataset_name = _create_raw_image_dataset(raw_dir)
    cfg_paths = _write_configs(tmp_path)

    version_id = create_dataset_version(dataset_name=dataset_name, raw_dir=raw_dir, processed_dir=processed_dir).name
    split_dataset(dataset_name=dataset_name, version_id=version_id, random_seed=42, processed_dir=processed_dir)
    run_image_preprocessing(
        dataset_name=dataset_name,
        version_id=version_id,
        prep_config_path=cfg_paths["preprocessing"],
        processed_dir=processed_dir,
    )

    mock_config = _make_mock_config(dataset_name, processed_dir, cfg_paths)
    result = run_cnn_training(mock_config, version_id)
    model_dir = save_model_artifact(result, run_id=version_id, task_type="image_classification_cnn", artifact_dir=artifact_dir)

    assert (model_dir / "model.pt").exists(), "model.pt must be saved for CNN models"
    assert (model_dir / "metadata.json").exists()

    with open(model_dir / "metadata.json") as f:
        meta = json.load(f)
    assert meta["algorithm"] == "cnn"


def test_cnn_evaluation_produces_confusion_matrix(tmp_path):
    """
    Full CNN pipeline through evaluation: confusion_matrix must be present
    in the evaluation report for task_type=image_classification_cnn.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"
    dataset_name = _create_raw_image_dataset(raw_dir)
    cfg_paths = _write_configs(tmp_path)

    # --- Versioning, splitting, preprocessing ---
    version_id = create_dataset_version(dataset_name=dataset_name, raw_dir=raw_dir, processed_dir=processed_dir).name
    split_dataset(dataset_name=dataset_name, version_id=version_id, random_seed=42, processed_dir=processed_dir)
    run_image_preprocessing(
        dataset_name=dataset_name,
        version_id=version_id,
        prep_config_path=cfg_paths["preprocessing"],
        processed_dir=processed_dir,
    )

    # --- Training + artifact saving ---
    mock_config = _make_mock_config(dataset_name, processed_dir, cfg_paths)
    result = run_cnn_training(mock_config, version_id)
    save_model_artifact(result, run_id=version_id, task_type="image_classification_cnn", artifact_dir=artifact_dir)

    # --- Evaluation (no production baseline) ---
    with patch("src.registry.model_registry.get_production_model_metrics", return_value=None):
        report = evaluate(mock_config, version_id, artifact_dir=artifact_dir)

    # Report structure
    assert report["task_type"] == "image_classification_cnn"
    assert "metrics" in report

    metrics = report["metrics"]
    assert "accuracy" in metrics
    assert "f1_score" in metrics

    # Confusion matrix must be present and correctly shaped (2×2 for 2 classes)
    assert "confusion_matrix" in metrics, "confusion_matrix must be in evaluation report"
    cm = metrics["confusion_matrix"]
    assert isinstance(cm, list), "confusion_matrix must be a list"
    assert len(cm) == 2, "confusion_matrix must have one row per class"
    assert all(len(row) == 2 for row in cm), "each row must have one entry per class"


def test_cnn_model_can_predict_on_val_split(tmp_path):
    """Trained CNN model's predict() method works on the validation split."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    dataset_name = _create_raw_image_dataset(raw_dir)
    cfg_paths = _write_configs(tmp_path)

    version_id = create_dataset_version(dataset_name=dataset_name, raw_dir=raw_dir, processed_dir=processed_dir).name
    split_dataset(dataset_name=dataset_name, version_id=version_id, random_seed=42, processed_dir=processed_dir)
    run_image_preprocessing(
        dataset_name=dataset_name,
        version_id=version_id,
        prep_config_path=cfg_paths["preprocessing"],
        processed_dir=processed_dir,
    )

    mock_config = _make_mock_config(dataset_name, processed_dir, cfg_paths)
    result = run_cnn_training(mock_config, version_id)

    preprocessed_dir = processed_dir / dataset_name / version_id / "preprocessed"
    val_data = np.load(preprocessed_dir / "val.npz")
    X_val = val_data["X"].transpose(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)

    predictions = result.model.predict(X_val)
    assert len(predictions) == X_val.shape[0]
    assert set(predictions).issubset({0, 1})


def test_cnn_evaluation_includes_official_test_report_when_preserved(tmp_path):
    """evaluate() should expose a separate official test report when the split preserves it."""
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    artifact_dir = tmp_path / "artifacts"
    dataset_name = _create_prefixed_raw_image_dataset(raw_dir)
    cfg_paths = _write_configs(tmp_path)

    version_id = create_dataset_version(dataset_name=dataset_name, raw_dir=raw_dir, processed_dir=processed_dir).name
    split_dataset(dataset_name=dataset_name, version_id=version_id, random_seed=42, processed_dir=processed_dir)
    run_image_preprocessing(
        dataset_name=dataset_name,
        version_id=version_id,
        prep_config_path=cfg_paths["preprocessing"],
        processed_dir=processed_dir,
    )

    mock_config = _make_mock_config(dataset_name, processed_dir, cfg_paths)
    result = run_cnn_training(mock_config, version_id)
    save_model_artifact(result, run_id=version_id, task_type="image_classification_cnn", artifact_dir=artifact_dir)

    with patch("src.registry.model_registry.get_production_model_metrics", return_value=None):
        report = evaluate(mock_config, version_id, artifact_dir=artifact_dir)

    assert "official_test_report" in report
    assert report["official_test_report"]["split"] == "test"
    assert "accuracy" in report["official_test_report"]["metrics"]
    assert "f1_score" in report["official_test_report"]["metrics"]
