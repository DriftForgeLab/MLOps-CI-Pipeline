# =============================================================================
# src/deployment/load_model.py — Production model loading
# =============================================================================
# Responsibility: Load the active production model artifact from disk at
# API startup. Uses MLflow Model Registry to find the current production
# model, then loads the artifact directly from the local artifact store.
# =============================================================================
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

_MODEL_ARTIFACT_DIR = Path("artifacts/runs")


def _get_tracking_uri() -> str:
    raw = os.environ.get("MLFLOW_TRACKING_URI")
    if raw:
        return raw
    return Path("mlruns").resolve().as_uri()


def load_production_model() -> tuple[object, dict]:
    """
    Load the active production model and its metadata.

    Queries the MLflow Model Registry for the current Production version,
    then loads the corresponding model artifact from the local artifact store.

    Returns:
        Tuple of (model, metadata) where model is the production model
        and metadata is the dict from metadata.json.

    Raises:
        RuntimeError: If no production model is registered or artifact is missing.
    """
    tracking_uri = _get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    # Find registered model — search all registered models for one with a Production version
    try:
        registered_models = client.search_registered_models()
    except Exception as e:
        raise RuntimeError(f"Could not connect to MLflow registry: {e}") from e

    production_version = None
    model_name = None
    for rm in registered_models:
        versions = client.get_latest_versions(rm.name, stages=["Production"])
        if versions:
            production_version = versions[0]
            model_name = rm.name
            break

    if production_version is None:
        raise RuntimeError(
            "No production model found in MLflow Model Registry. "
            "Run the pipeline and approve a model before starting the API."
        )

    logger.info(
        "Production model found: name=%s version=%s run_id=%s",
        model_name, production_version.version, production_version.run_id,
    )

    # Load model artifact from local artifact store
    run_id = production_version.run_id
    model_path = _MODEL_ARTIFACT_DIR / _find_version_id_for_run(client, run_id) / "model" / "model.joblib"

    if not model_path.exists():
        # Fallback: load directly via MLflow
        model_uri = f"models:/{model_name}/Production"
        logger.info("Loading model via MLflow URI: %s", model_uri)
        model = mlflow.sklearn.load_model(model_uri)
        metadata = _get_metadata_from_run(client, run_id)
        return model, metadata

    model = joblib.load(model_path)
    logger.info("Model loaded from: %s", model_path.resolve())

    metadata_path = model_path.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}

    return model, metadata


def _find_version_id_for_run(client: MlflowClient, run_id: str) -> str:
    """Get dataset_version_id from the MLflow run tags."""
    run = client.get_run(run_id)
    version_id = run.data.tags.get("pipeline.dataset_version_id", "")
    if not version_id:
        raise RuntimeError(
            f"Could not find 'pipeline.dataset_version_id' tag on run '{run_id}'."
        )
    return version_id


def _get_metadata_from_run(client: MlflowClient, run_id: str) -> dict:
    """Build a metadata dict from MLflow run params and tags."""
    run = client.get_run(run_id)
    return {
        "algorithm": run.data.params.get("algorithm", "unknown"),
        "task_type": run.data.tags.get("pipeline.task_type", "unknown"),
        "dataset_version_id": run.data.tags.get("pipeline.dataset_version_id", ""),
        "trained_at": run.data.params.get("trained_at", ""),
    }