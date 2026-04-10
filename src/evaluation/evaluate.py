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

from src.config.loader import PipelineConfig, load_evaluation_config, load_promotion_config
from src.config.schema import CLASSIFICATION_TASK_TYPES, IMAGE_TASK_TYPES
from src.data.preprocess import PREPROCESSED_SUBDIR
from src.promotion.comparator import compare_metrics, no_baseline_comparison

logger = logging.getLogger(__name__)


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
    pt_path = model_dir / "model.pt"
    joblib_path = model_dir / "model.joblib"

    if pt_path.exists():
        import torch
        model = torch.load(pt_path, weights_only=False, map_location="cpu")
    elif joblib_path.exists():
        model = joblib.load(joblib_path)
    else:
        raise FileNotFoundError(
            f"Model artifact not found at '{model_dir}'. "
            "Run the training stage before evaluation."
        )

    # --- Load feature map ---
    preprocessed_dir = (
        Path(config.data.processed)
        / config.dataset
        / version_id
        / PREPROCESSED_SUBDIR
    )
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
    if config.task_type in IMAGE_TASK_TYPES:
        val_npz_path = preprocessed_dir / "val.npz"
        if not val_npz_path.exists():
            raise FileNotFoundError(
                f"Preprocessed val split not found at '{val_npz_path}'. "
                "Run the preprocessing stage before evaluation."
            )
        data = np.load(val_npz_path)
        X, y_true = data["X"], data["y"]
        if config.task_type == "image_classification_cnn" and X.ndim == 4:
            X = X.transpose(0, 3, 1, 2)
    else:
        val_path = preprocessed_dir / "val.csv"
        if not val_path.exists():
            raise FileNotFoundError(
                f"Preprocessed val split not found at '{val_path}'. "
                "Run the preprocessing stage before evaluation."
            )
        df = pd.read_csv(val_path)
        X = df[output_features].values
        y_true = df[target].values

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
        "metrics": metrics,
        "comparison": comparison
    }

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
