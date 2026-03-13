# =============================================================================
# src/registry/model_registry.py — MLflow Model Registry integration
# =============================================================================
# Responsibility: Register approved models in the MLflow Model Registry,
# manage Production stage transitions, attach lineage tags, and look up
# current Production model metrics for comparison.
# =============================================================================

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.config.loader import PipelineConfig

_logger = logging.getLogger(__name__)

_CLASSIFICATION_EVAL_KEYS: frozenset[str] = frozenset({"accuracy", "precision", "recall", "f1_score"})
_REGRESSION_EVAL_KEYS: frozenset[str] = frozenset({"r2", "mae", "mse", "rmse"})

_EVAL_KEYS_BY_TASK: dict[str, frozenset[str]] = {
    "classification": _CLASSIFICATION_EVAL_KEYS,
    "regression": _REGRESSION_EVAL_KEYS,
}

# Shared artifact path constants.  Import these in callers (e.g. steps.py) so
# the coupling between where artifacts are logged and where they are referenced
# is enforced by the import graph, not by a matching string.
MODEL_ARTIFACT_SUBPATH = "model"
PROMOTION_ARTIFACT_SUBDIR = "promotion"
PROMOTION_DECISION_ARTIFACT_PATH = f"{PROMOTION_ARTIFACT_SUBDIR}/promotion_decision.json"

_REQUIRED_LINEAGE_TAGS: frozenset[str] = frozenset({
    "lineage.run_id",
    "lineage.config_hash",
    "lineage.dataset_version_id",
    "lineage.task_type",
    "lineage.algorithm",
    "lineage.trained_at",
    "lineage.promotion_outcome",
})


def resolve_tracking_uri(config: PipelineConfig) -> str:
    from urllib.parse import urlparse

    def _to_uri(raw: str) -> str:
        parsed = urlparse(raw)
        if not parsed.scheme or len(parsed.scheme) == 1:
            return Path(raw).resolve().as_uri()
        return raw

    raw = config.mlflow.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if raw:
        return _to_uri(raw)
    return Path("mlruns").resolve().as_uri()


def resolve_model_name(config: PipelineConfig) -> str:
    if config.mlflow.registry_model_name:
        return config.mlflow.registry_model_name
    raw_name = f"{config.project.name}-{config.task_type}"
    # MLflow model names must contain only alphanumerics, dots, hyphens, underscores.
    return re.sub(r"[^a-zA-Z0-9._\-]", "-", raw_name)


def get_mlflow_client(config: PipelineConfig) -> MlflowClient:
    return MlflowClient(tracking_uri=resolve_tracking_uri(config))


def is_run_already_registered(client: MlflowClient, run_id: str, model_name: str) -> bool:
    # MlflowException propagates to the caller — do NOT mask registry errors as False.
    # A connectivity failure must not be silently treated as "not registered".
    versions = client.search_model_versions(f"run_id='{run_id}' AND name='{model_name}'")
    return len(versions) > 0


def register_approved_model(config: PipelineConfig, run_id: str):
    """Register the model artifact from run_id in the MLflow Model Registry.

    Returns the MLflow ModelVersion object (existing or newly created).
    Uses a single search call to avoid the check-then-act race condition that
    would arise from a separate is_run_already_registered() guard.
    """
    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)

    # Single search — reuse the result for both the guard and the return value.
    existing = client.search_model_versions(f"run_id='{run_id}' AND name='{model_name}'")
    if existing:
        if len(existing) > 1:
            existing.sort(key=lambda v: int(v.version), reverse=True)
            _logger.warning(
                "run_id '%s' has %d versions under model '%s' — using highest version (%s).",
                run_id, len(existing), model_name, existing[0].version,
            )
        _logger.warning(
            "Run '%s' is already registered under model '%s' (version %s) — skipping duplicate registration.",
            run_id, model_name, existing[0].version,
        )
        return existing[0]

    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_SUBPATH}"
    _logger.info("Registering model: uri=%s name=%s", model_uri, model_name)
    model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
    _logger.info("Model registered: name=%s version=%s", model_name, model_version.version)
    return model_version


def promote_to_production(
    config: PipelineConfig,
    version_number: int | str,
    mlflow_run_id: str,
    *,
    promoted_by: str = "pipeline",
    archive_existing: bool = True,
) -> None:
    """Transition a registered model version to the Production stage.

    Args:
        promoted_by:     Identity recorded in the promotion.promoted_by tag.
                         Defaults to "pipeline"; pass a username or CI actor for traceability.
        archive_existing: When True (default), existing Production versions are archived.
                          Set False for canary or blue/green deployments.
    """
    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    version_str = str(version_number)
    _logger.info("Promoting model '%s' version %s to Production", model_name, version_str)
    client.transition_model_version_stage(
        name=model_name,
        version=version_str,
        stage="Production",
        archive_existing_versions=archive_existing,
    )
    now = datetime.now(timezone.utc).isoformat()
    client.set_model_version_tag(model_name, version_str, "promotion.promoted_at", now)
    client.set_model_version_tag(model_name, version_str, "promotion.promoted_by", promoted_by)
    client.set_model_version_tag(model_name, version_str, "promotion.decision_run_id", mlflow_run_id)
    _logger.info("Model '%s' version %s is now Production", model_name, version_str)


def _build_lineage_tags(config: PipelineConfig, run: Run, report: dict, decision: dict) -> dict[str, str]:
    """Assemble all lineage tags for a registered model version.

    Raises:
        ValueError: If task_type is unsupported, or if any required lineage tag
                    is missing or empty (hard fail to protect audit trail integrity).
    """
    run_tags = run.data.tags
    run_params = run.data.params

    # Strip "eval." prefix from run metrics to get plain metric names.
    run_metrics = {
        k[5:]: v
        for k, v in run.data.metrics.items()
        if k.startswith("eval.")
    }

    task_type = config.task_type
    eval_keys = _EVAL_KEYS_BY_TASK.get(task_type)
    if eval_keys is None:
        raise ValueError(
            f"Unsupported task_type for lineage tagging: {task_type!r}. "
            f"Supported types: {sorted(_EVAL_KEYS_BY_TASK)}"
        )

    eval_tags = {
        f"lineage.eval.{k}": str(round(v, 6))
        for k, v in run_metrics.items()
        if k in eval_keys
    }

    tags: dict[str, str] = {
        # Identity
        "lineage.run_id":                run.info.run_id,
        "lineage.pipeline_execution_id": run_tags.get("pipeline.execution_id", ""),
        "lineage.config_hash":           run_tags.get("pipeline.config_hash", ""),
        # Data
        "lineage.dataset_name":          config.dataset,
        "lineage.dataset_version_id":    run_tags.get("pipeline.dataset_version_id", ""),
        # Training
        "lineage.task_type":             task_type,
        "lineage.algorithm":             run_params.get("algorithm", ""),
        "lineage.train_rows":            run_params.get("train_rows", ""),
        "lineage.trained_at":            run_params.get("trained_at", ""),
        # Decision
        "lineage.promotion_outcome":     decision.get("outcome", ""),
        "lineage.comparison_verdict":    report.get("comparison", {}).get("overall_verdict", ""),
        "lineage.decision_artifact":     PROMOTION_DECISION_ARTIFACT_PATH,
    }
    tags.update(eval_tags)

    # Hard fail if any required tag is missing or empty — a model must not reach
    # Production with an incomplete audit trail.
    missing = [k for k in _REQUIRED_LINEAGE_TAGS if not tags.get(k)]
    if missing:
        raise ValueError(
            f"Cannot attach lineage: required tags are missing or empty: {missing}. "
            "Check that the training run logged the expected params and tags."
        )

    return {k: v for k, v in tags.items() if v}  # drop remaining optional empty strings


def attach_lineage_tags(
    config: PipelineConfig,
    version_number: int | str,
    run: Run,
    report: dict,
    decision: dict,
) -> None:
    """Attach lineage tags to a registered model version.

    Tags are written one at a time (MLflow API limitation). If the write loop is
    interrupted, the error is logged with full details of which tags were and were
    not written before re-raising.
    """
    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    version_str = str(version_number)
    tags = _build_lineage_tags(config, run, report, decision)
    written: list[str] = []
    try:
        for key, value in tags.items():
            client.set_model_version_tag(model_name, version_str, key, value)
            written.append(key)
    except Exception:
        not_written = [k for k in tags if k not in written]
        _logger.error(
            "Lineage tagging interrupted after %d/%d tags written. Not written: %s",
            len(written), len(tags), not_written,
        )
        raise
    _logger.info(
        "Attached %d lineage tags to model '%s' version %s",
        len(tags), model_name, version_str,
    )


def get_production_model_metrics(config: PipelineConfig) -> dict | None:
    """Return eval metrics of the current Production model version, or None if no Production exists.

    Returns None for four distinct states — use the log level to distinguish them:
      INFO:    No Production model registered (valid first-run / bootstrap scenario).
      ERROR:   Registry unreachable, or the Production run record is missing.
      WARNING: Production run exists but has no 'eval.*' metrics.

    IMPORTANT: This function assumes metrics are logged with the "eval." prefix
    by log_evaluation_to_mlflow() in src/pipeline/mlflow_logger.py.
    If that prefix changes, update the stripping logic here accordingly.
    """
    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
    except Exception as e:
        _logger.error("Could not query Production model from registry: %s", e)
        return None
    if not versions:
        _logger.info("No Production model found in registry for '%s'", model_name)
        return None
    if len(versions) > 1:
        _logger.warning(
            "Found %d Production versions for '%s' — expected 1. Using version %s.",
            len(versions), model_name, versions[0].version,
        )
    prod_run_id = versions[0].run_id
    try:
        run = client.get_run(prod_run_id)
    except Exception as e:
        _logger.error("Could not fetch Production model run '%s': %s", prod_run_id, e)
        return None
    metrics = {
        k[5:]: v
        for k, v in run.data.metrics.items()
        if k.startswith("eval.")
    }
    if not metrics:
        _logger.warning(
            "Production model run '%s' has no 'eval.*' metrics — falling back to no_baseline.",
            prod_run_id,
        )
        return None
    _logger.info(
        "Found Production model metrics from run '%s': %s",
        prod_run_id, list(metrics.keys()),
    )
    return metrics
