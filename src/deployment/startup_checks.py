# =============================================================================
# src/deployment/startup_checks.py — Production model validation and loading
# =============================================================================
# Responsibility: Ensure that only approved Production-stage models from
# the MLflow Model Registry are loaded and served by the prediction service.
#
# This module is independent of the FastAPI application layer.
# It queries the MLflow registry, loads model artifacts, extracts lineage
# metadata, and returns structured results that the API layer can use.
#
# Governance guarantees:
#   - Build time:   No model baked into the Docker image.
#   - Startup:      Only "Production"-stage versions are queried.
#   - Runtime:      Models loaded once at startup; no dynamic switching.
#   - Traceability: Lineage metadata (run_id, version, algorithm) preserved.
# =============================================================================

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
import joblib
import torch

from src.config.loader import PipelineConfig, load_config, load_deployment_config
from src.config.schema import DeploymentConfig
from src.registry.model_registry import (
    get_mlflow_client,
    resolve_tracking_uri,
)

_logger = logging.getLogger(__name__)


@dataclass
class ProductionModelInfo:
    """Metadata and reference for a loaded Production model."""

    model: object                           # The loaded sklearn or PyTorch model.
    model_name: str                         # MLflow registry model name.
    model_version: str                      # Registry version number.
    run_id: str                             # MLflow training run ID.
    stage: str                              # Must be "Production".
    algorithm: str                          # From lineage tags.
    task_type: str                          # From lineage tags.
    trained_at: str                         # From lineage tags.
    dataset_version_id: str                 # From lineage tags.
    dataset_name: str                       # From run tags.
    promotion_outcome: str                  # From lineage tags.
    model_format: str = "sklearn"             # "sklearn" or "pytorch"
    feature_names: list[str] = field(default_factory=list)
    # Image model fields (None for tabular models)
    image_shape: list | None = None         # [H, W, C] from feature_map
    normalization_stats: dict | None = None # {"mean": [...], "std": [...]}
    index_to_class: dict | None = None      # {"0": "cat", "1": "dog"}


def _find_all_production_versions(
    client, config: PipelineConfig, *, allowed_stage: str = "Production",
) -> list[tuple[str, object]]:
    """Find all model versions at the given stage in the MLflow registry.

    Resolution order:
      1. ``MLFLOW_MODEL_NAME`` env var — look up only that model name.
      2. Scan all registered models for any with a version at *allowed_stage*.

    Returns:
        List of (model_name, version) tuples, one per matching model.

    Raises:
        RuntimeError: If the registry is unreachable or no matching model exists.
    """
    explicit_name = os.environ.get("MLFLOW_MODEL_NAME")
    if explicit_name:
        _logger.info("Using MLFLOW_MODEL_NAME from environment: %s", explicit_name)
        try:
            versions = client.get_latest_versions(explicit_name, stages=[allowed_stage])
        except Exception as exc:
            raise RuntimeError(
                f"Cannot query MLflow registry for '{explicit_name}': {exc}"
            ) from exc
        if not versions:
            raise RuntimeError(
                f"No {allowed_stage} model found in registry for '{explicit_name}'. "
                "Run the pipeline and approve a model before starting the service."
            )
        return [(explicit_name, versions[0])]

    # Scan all registered models for versions at the allowed stage.
    try:
        registered = client.search_registered_models()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot search MLflow Model Registry: {exc}"
        ) from exc

    found: list[tuple[str, object]] = []
    for reg_model in registered:
        try:
            versions = client.get_latest_versions(reg_model.name, stages=[allowed_stage])
        except Exception as exc:
            _logger.warning("Could not query versions for '%s': %s", reg_model.name, exc)
            continue
        if versions:
            found.append((reg_model.name, versions[0]))

    if not found:
        raise RuntimeError(
            f"No {allowed_stage} model found anywhere in the MLflow Model Registry. "
            "Run the pipeline and approve a model before starting the service. "
            "To target a specific model, set the MLFLOW_MODEL_NAME environment variable."
        )

    return found


def _load_single_production_model(
    client, model_name: str, prod_version
) -> ProductionModelInfo:
    """Load a single Production model artifact and recover its feature map.

    Raises:
        RuntimeError: If the model artifact or required run tags are missing.
    """
    tags = prod_version.tags or {}

    run = client.get_run(prod_version.run_id)
    version_id = run.data.tags.get("pipeline.dataset_version_id", "")
    if not version_id:
        raise RuntimeError(
            f"Could not find 'pipeline.dataset_version_id' tag on run '{prod_version.run_id}'."
        )

    dataset_name = run.data.tags.get("pipeline.dataset", "")

    # --- Load model artifact ------------------------------------------------
    model_dir = Path("artifacts/runs") / version_id / "model"
    pt_path = model_dir / "model.pt"
    joblib_path = model_dir / "model.joblib"

    if pt_path.exists():
        loaded_model = torch.load(pt_path, weights_only=False)
        model_format = "pytorch"
        _logger.info("PyTorch model loaded from: %s", pt_path.resolve())
    elif joblib_path.exists():
        loaded_model = joblib.load(joblib_path)
        model_format = "sklearn"
        _logger.info("sklearn model loaded from: %s", joblib_path.resolve())
    else:
        raise RuntimeError(
            f"Model artifact not found at '{model_dir}'. "
            "The registry pointer may be stale or artifacts may have been deleted."
        )

    # --- Recover feature map fields -----------------------------------------
    feature_names: list[str] = []
    image_shape: list | None = None
    normalization_stats: dict | None = None
    index_to_class: dict | None = None
    try:
        feature_map_path = (
            Path("data/processed") / dataset_name / version_id / "preprocessed" / "feature_map.json"
        )
        if feature_map_path.exists():
            with open(feature_map_path) as f:
                feature_map = json.load(f)
            feature_names = feature_map.get("output_features", [])
            image_shape = feature_map.get("image_shape")
            normalization_stats = feature_map.get("normalization_stats")
            index_to_class = feature_map.get("index_to_class")
            _logger.info(
                "Feature map loaded for '%s': image_shape=%s features=%d",
                model_name, image_shape, len(feature_names),
            )
        else:
            _logger.warning(
                "feature_map.json not found at %s — input validation will be skipped.",
                feature_map_path,
            )
    except Exception:
        _logger.warning(
            "Could not recover feature map for '%s' — input validation will be skipped.",
            model_name,
        )

    return ProductionModelInfo(
        model=loaded_model,
        model_name=model_name,
        model_version=str(prod_version.version),
        run_id=prod_version.run_id,
        stage="Production",
        algorithm=tags.get("lineage.algorithm", "unknown"),
        model_format=model_format,
        task_type=tags.get("lineage.task_type", "unknown"),
        trained_at=tags.get("lineage.trained_at", "unknown"),
        dataset_version_id=tags.get("lineage.dataset_version_id", "unknown"),
        dataset_name=dataset_name,
        promotion_outcome=tags.get("lineage.promotion_outcome", "unknown"),
        feature_names=feature_names,
        image_shape=image_shape,
        normalization_stats=normalization_stats,
        index_to_class=index_to_class,
    )


def load_all_production_models(
    config: PipelineConfig,
    deploy_config: DeploymentConfig,
) -> dict[str, ProductionModelInfo]:
    """Load all Production models from the MLflow registry.

    Returns a dict keyed by MLflow model name. Models that fail to load are
    logged as errors and skipped — the API will start as long as at least
    one model loads successfully (when ``require_production_model`` is True).

    When ``require_production_model`` is False, the API is allowed to start
    with zero loaded models.

    Raises:
        RuntimeError: If ``require_production_model`` is True and no models
            could be loaded.
    """
    allowed_stage = deploy_config.model.allowed_stage
    require = deploy_config.model.require_production_model
    client = get_mlflow_client(config)

    try:
        prod_versions = _find_all_production_versions(
            client, config, allowed_stage=allowed_stage,
        )
    except RuntimeError:
        if not require:
            _logger.warning(
                "No %s models found, but require_production_model=False — "
                "starting with zero models.",
                allowed_stage,
            )
            return {}
        raise

    models: dict[str, ProductionModelInfo] = {}
    for model_name, prod_version in prod_versions:
        try:
            info = _load_single_production_model(client, model_name, prod_version)
            models[model_name] = info
            _logger.info(
                "Loaded %s model: '%s' version=%s algorithm=%s",
                allowed_stage, model_name, info.model_version, info.algorithm,
            )
        except Exception as exc:
            _logger.error("Failed to load model '%s': %s", model_name, exc)

    if not models and require:
        raise RuntimeError(
            f"No {allowed_stage} models could be loaded. "
            "Check that model artifacts exist and that run tags are complete."
        )

    if not models:
        _logger.warning("No models loaded — API will start without prediction capability.")

    return models


def validate_environment() -> tuple[PipelineConfig, DeploymentConfig]:
    """Validate that the runtime environment is correctly configured.

    Checks that the pipeline config and deployment config can be loaded
    and that the MLflow tracking store is accessible.

    Returns:
        A ``(PipelineConfig, DeploymentConfig)`` tuple for use by the
        application lifespan and ``load_all_production_models``.

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

    deploy_config_path = Path(config.configs.deployment)
    if not deploy_config_path.exists():
        raise RuntimeError(
            f"Deployment config not found at '{deploy_config_path}'."
        )

    try:
        deploy_config = load_deployment_config(deploy_config_path)
    except (FileNotFoundError, ValueError) as exc:
        raise RuntimeError(f"Failed to load deployment config: {exc}") from exc

    tracking_uri = resolve_tracking_uri(config)
    _logger.info("MLflow tracking URI: %s", tracking_uri)

    if deploy_config.model.require_production_model:
        _logger.info(
            "Production model required at startup (allowed_stage=%s).",
            deploy_config.model.allowed_stage,
        )
    else:
        _logger.info(
            "Production model NOT required at startup — API may start with zero models.",
        )

    return config, deploy_config
