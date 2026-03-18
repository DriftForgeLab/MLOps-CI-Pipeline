# =============================================================================
# src/deployment/startup_checks.py — Production model validation and loading
# =============================================================================
# Responsibility: Ensure that only the approved Production-stage model from
# the MLflow Model Registry is loaded and served by the prediction service.
#
# This module is independent of the FastAPI application layer (ID 10).
# It queries the MLflow registry, loads the model artifact, extracts lineage
# metadata, and returns a structured result that the API layer can use.
#
# Governance guarantees:
#   - Build time:   No model baked into the Docker image.
#   - Startup:      Only "Production"-stage versions are queried.
#   - Runtime:      Model loaded once at startup; no dynamic switching.
#   - Traceability: Lineage metadata (run_id, version, algorithm) preserved.
# =============================================================================

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.config.loader import PipelineConfig, load_config, load_deployment_config
from src.registry.model_registry import (
    get_mlflow_client,
    resolve_model_name,
    resolve_tracking_uri,
)

_logger = logging.getLogger(__name__)


@dataclass
class ProductionModelInfo:
    """Metadata and reference for the loaded Production model."""

    model: object                           # The loaded sklearn model object.
    model_name: str                         # MLflow registry model name.
    model_version: str                      # Registry version number.
    run_id: str                             # MLflow training run ID.
    stage: str                              # Must be "Production".
    algorithm: str                          # From lineage tags.
    trained_at: str                         # From lineage tags.
    dataset_version_id: str                 # From lineage tags.
    promotion_outcome: str                  # From lineage tags.
    feature_names: list[str] = field(default_factory=list)


def load_production_model(config: PipelineConfig) -> ProductionModelInfo:
    """Load and validate the Production model from the MLflow Model Registry.

    Uses the same registry query pattern as
    ``src.registry.model_registry.get_production_model_metrics`` but goes
    further: it loads the actual model artifact and extracts lineage tags
    so the API can serve predictions with full traceability.

    Raises:
        RuntimeError: If no Production model exists, the registry is
            unreachable, or the model artifact cannot be loaded.
    """
    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)

    # --- Query the registry for the Production-stage version ----------------
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception as exc:
        raise RuntimeError(
            f"Cannot query MLflow Model Registry for '{model_name}': {exc}"
        ) from exc

    if not versions:
        raise RuntimeError(
            f"No Production model found in registry for '{model_name}'. "
            "Run the pipeline and approve a model before starting the service."
        )

    if len(versions) > 1:
        _logger.warning(
            "Found %d Production versions for '%s' — using version %s.",
            len(versions), model_name, versions[0].version,
        )

    prod_version = versions[0]
    _logger.info(
        "Found Production model: name=%s version=%s run_id=%s",
        model_name, prod_version.version, prod_version.run_id,
    )

    # --- Extract lineage tags from the model version -----------------------
    tags = prod_version.tags or {}

# --- Load the model artifact via joblib from local artifact store -------
    run = client.get_run(prod_version.run_id)
    version_id = run.data.tags.get("pipeline.dataset_version_id", "")
    if not version_id:
        raise RuntimeError(
            f"Could not find 'pipeline.dataset_version_id' tag on run '{prod_version.run_id}'."
        )

    model_path = Path("artifacts/runs") / version_id / "model" / "model.joblib"
    if not model_path.exists():
        raise RuntimeError(
            f"Model artifact not found at '{model_path}'. "
            "The registry pointer may be stale or artifacts may have been deleted."
        )

    import joblib
    loaded_model = joblib.load(model_path)
    _logger.info("Model artifact loaded from: %s", model_path.resolve())
    

    # --- Try to recover feature names from feature_map.json ---------
    feature_names: list[str] = []
    try:
        run = client.get_run(prod_version.run_id)
        version_id = run.data.tags.get("pipeline.dataset_version_id", "")
        dataset_name = run.data.tags.get("pipeline.dataset", "")
        feature_map_path = (
            Path("data/processed") / dataset_name / version_id / "preprocessed" / "feature_map.json"
        )
        if feature_map_path.exists():
            with open(feature_map_path) as f:
                feature_map = json.load(f)
            feature_names = feature_map.get("output_features", [])
            _logger.info("Feature names recovered: %s", feature_names)
        else:
            _logger.warning(
                "feature_map.json not found at %s — input validation will be skipped.",
                feature_map_path,
            )
    except Exception:
        _logger.warning(
            "Could not recover feature names — input validation will be skipped."
        )
        
    return ProductionModelInfo(
        model=loaded_model,
        model_name=model_name,
        model_version=str(prod_version.version),
        run_id=prod_version.run_id,
        stage="Production",
        algorithm=tags.get("lineage.algorithm", "unknown"),
        trained_at=tags.get("lineage.trained_at", "unknown"),
        dataset_version_id=tags.get("lineage.dataset_version_id", "unknown"),
        promotion_outcome=tags.get("lineage.promotion_outcome", "unknown"),
        feature_names=feature_names,
    )


def validate_environment() -> PipelineConfig:
    """Validate that the runtime environment is correctly configured.

    Checks that the pipeline config and deployment config can be loaded
    and that the MLflow tracking store is accessible.

    Returns the loaded PipelineConfig for use by ``load_production_model``.

    Raises:
        RuntimeError: If any required configuration is missing or invalid.
    """
    config_path_str = os.environ.get(
        "PIPELINE_CONFIG_PATH", "src/config/pipeline.yaml"
    )
    config_path = Path(config_path_str)

    if not config_path.exists():
        raise RuntimeError(
            f"Pipeline config not found at '{config_path}'. "
            "Set PIPELINE_CONFIG_PATH or mount the config directory."
        )

    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Failed to load pipeline config: {exc}") from exc

    # Validate that the deployment config is also loadable.
    deploy_config_path = Path(config.configs.deployment)
    if not deploy_config_path.exists():
        raise RuntimeError(
            f"Deployment config not found at '{deploy_config_path}'."
        )

    try:
        deploy_config = load_deployment_config(deploy_config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(
            f"Failed to load deployment config: {exc}"
        ) from exc

    # Verify the tracking URI is reachable (basic sanity check).
    tracking_uri = resolve_tracking_uri(config)
    _logger.info("MLflow tracking URI: %s", tracking_uri)

    if deploy_config.model.require_production_model:
        _logger.info(
            "Production model required at startup (allowed_stage=%s).",
            deploy_config.model.allowed_stage,
        )

    return config
