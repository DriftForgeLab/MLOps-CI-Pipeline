# =============================================================================
# src/registry/metadata.py — Model artifact persistence
# =============================================================================
# Responsibility: Save a trained model to a versioned artifact path and write
# a companion metadata.json alongside it.
#
# Artifact structure:
#   artifacts/runs/<run_id>/model/
#       model.joblib       ← serialized model
#       metadata.json      ← training context
# =============================================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.training import TrainingResult

logger = logging.getLogger(__name__)


def save_model_artifact(
    result: TrainingResult,
    run_id: str,
    artifact_dir: Path = Path("artifacts/runs"),
) -> Path:
    """
    Save a trained model and its metadata to a versioned artifact directory.

    Args:
        result:       TrainingResult from run_training()
        run_id:       Unique identifier for this pipeline run (e.g. config_hash)
        artifact_dir: Base directory for all run artifacts

    Returns:
        Path to the directory containing model.joblib and metadata.json
    """
    model_dir = artifact_dir / run_id / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(result.model, model_path)
    logger.info("  Model saved to: %s", model_path.resolve())

    metadata = {
        "task_type": "classification" if "Classifier" in type(result.model).__name__ else "regression",
        "algorithm": result.algorithm,
        "hyperparameters": result.hyperparameters,
        "dataset_version_id": result.dataset_version_id,
        "random_seed": result.random_seed,
        "trained_at": result.trained_at,
        "train_rows": result.train_rows,
    }

    metadata_path = model_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("  Metadata saved to: %s", metadata_path.resolve())

    return model_dir
