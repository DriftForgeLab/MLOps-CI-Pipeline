# =============================================================================
# src/evaluation/evaluate.py — Model evaluation
# =============================================================================
# Responsibility: Load a trained model artifact, run predictions on the
# preprocessed evaluation split, and return a structured evaluation report.
#
# Design: evaluate() performs I/O (reads config files, queries MLflow registry)
# but has no pipeline side effects (does not write artifacts or modify state).
# It can be called independently of the pipeline for debugging or ad-hoc use.
# =============================================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from src.config.loader import PipelineConfig, load_evaluation_config, load_promotion_config
from src.config.schema import CLASSIFICATION_TASK_TYPES, IMAGE_TASK_TYPES
from src.data.preprocess import PREPROCESSED_SUBDIR
from src.promotion.comparator import compare_metrics, no_baseline_comparison

logger = logging.getLogger(__name__)


def _load_model(model_dir: Path):
    pt_path = model_dir / "model.pt"
    joblib_path = model_dir / "model.joblib"

    if pt_path.exists():
        import torch
        from src.common.device import resolve_device

        device = resolve_device()
        model = torch.load(pt_path, weights_only=False, map_location="cpu")
        if hasattr(model, "to"):
            model.to(device)
        return model
    if joblib_path.exists():
        return joblib.load(joblib_path)

    raise FileNotFoundError(
        f"Model artifact not found at '{model_dir}'. "
        "Run the training stage before evaluation."
    )


def _load_split_arrays(
    config: PipelineConfig,
    preprocessed_dir: Path,
    output_features: list[str],
    target: str,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one preprocessed split for evaluation."""
    if config.task_type in IMAGE_TASK_TYPES:
        split_npz_path = preprocessed_dir / f"{split_name}.npz"
        if not split_npz_path.exists():
            raise FileNotFoundError(
                f"Preprocessed {split_name} split not found at '{split_npz_path}'. "
                "Run the preprocessing stage before evaluation."
            )
        data = np.load(split_npz_path)
        X, y_true = data["X"], data["y"]
        if config.task_type == "image_classification_cnn" and X.ndim == 4:
            X = X.transpose(0, 3, 1, 2)
        return X, y_true

    split_path = preprocessed_dir / f"{split_name}.csv"
    if not split_path.exists():
        raise FileNotFoundError(
            f"Preprocessed {split_name} split not found at '{split_path}'. "
            "Run the preprocessing stage before evaluation."
        )
    df = pd.read_csv(split_path)
    return df[output_features].values, df[target].values


def _build_split_report(
    version_id: str,
    task_type: str,
    split_name: str,
    metrics: dict,
) -> dict:
    return {
        "model_version": version_id,
        "dataset_version": version_id,
        "task_type": task_type,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": split_name,
        "metrics": metrics,
    }


def _should_run_official_test(version_dir: Path) -> bool:
    yaml_path = version_dir / "dataset.yaml"
    if not yaml_path.exists():
        return False
    with open(yaml_path) as f:
        metadata = yaml.safe_load(f) or {}
    split_meta = metadata.get("split") or {}
    return bool(split_meta.get("preserved_original_test"))


def evaluate(
    config: PipelineConfig,
    version_id: str,
    artifact_dir: Path = Path("artifacts/runs"),
) -> dict:
    """
    Load a trained model and evaluate it on the preprocessed val split.

    Args:
        config:       Validated PipelineConfig from load_config()
        version_id:   Dataset version ID from versioning step
        artifact_dir: Base directory for model artifacts

    Returns:
        Structured evaluation report as a dict.

    Raises:
        FileNotFoundError: If model artifact or preprocessed data is missing.
    """
    # --- Load model ---
    eval_config = load_evaluation_config(Path(config.configs.evaluation))
    model_dir = artifact_dir / version_id / "model"
    model = _load_model(model_dir)

    # --- Load feature map ---
    version_dir = Path(config.data.processed) / config.dataset / version_id
    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR
    feature_map_path = preprocessed_dir / "feature_map.json"
    if not feature_map_path.exists():
        raise FileNotFoundError(
            f"feature_map.json not found at '{feature_map_path}'. "
            "Run the preprocessing stage before evaluation."
        )
    with open(feature_map_path) as f:
        feature_map = json.load(f)

    output_features: list[str] = feature_map["output_features"]
    target: str = feature_map["target"]

    # --- Load val split ---
    X, y_true = _load_split_arrays(config, preprocessed_dir, output_features, target, "val")
    y_pred = model.predict(X)

    # --- Compute metrics ---
    metrics = _compute_metrics(y_true, y_pred, config.task_type, eval_config)
    
    # --- Compare against production model ---
    promotion_config = load_promotion_config(Path(config.configs.promotion))
    task_config = (
        promotion_config.classification
        if config.task_type in CLASSIFICATION_TASK_TYPES
        else promotion_config.regression
    )
    metrics_to_compare = [rule.metric for rule in task_config.rules]

    # --- Production model lookup via MLflow Model Registry ---
    from src.registry.model_registry import get_production_model_metrics
    production_metrics = get_production_model_metrics(config)
    if production_metrics is None:
        comparison = no_baseline_comparison()
        logger.info("  Comparison: no Production model in registry — bootstrap scenario.")
    else:
        comparison = compare_metrics(metrics, production_metrics, metrics_to_compare)
        logger.info("  Comparison verdict: %s", comparison.get("overall_verdict", "unknown").upper())


    report = {
        "model_version": version_id,
        "dataset_version": version_id,
        "task_type": config.task_type,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "val",
        "metrics": metrics,
        "comparison": comparison,
    }

    if _should_run_official_test(version_dir):
        X_test, y_test = _load_split_arrays(config, preprocessed_dir, output_features, target, "test")
        y_test_pred = model.predict(X_test)
        test_metrics = _compute_metrics(y_test, y_test_pred, config.task_type, eval_config)
        report["official_test_report"] = _build_split_report(
            version_id=version_id,
            task_type=config.task_type,
            split_name="test",
            metrics=test_metrics,
        )
        logger.info("  Official test evaluation complete: metrics=%s", test_metrics)

    logger.info(
        "  Evaluation complete: task_type=%s, metrics=%s",
        config.task_type,
        metrics,
    )

    return report


def _compute_metrics(y_true, y_pred, task_type: str, eval_config) -> dict:
    """Compute metrics based on task type."""
    if task_type in CLASSIFICATION_TASK_TYPES:
        from sklearn.metrics import (
            accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
        )
        avg = eval_config.classification.averaging
        metrics = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, average=avg, zero_division=0)), 4),
            "averaging": avg,
        }
        if task_type == "image_classification_cnn":
            metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
        return metrics
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mse = float(mean_squared_error(y_true, y_pred))
        return {
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "mse": round(mse, 4),
            "rmse": round(float(np.sqrt(mse)), 4),
            "r2": round(float(r2_score(y_true, y_pred)), 4),
        }
