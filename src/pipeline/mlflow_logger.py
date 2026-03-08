import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import mlflow

from src.config.loader import PipelineConfig

_logger = logging.getLogger(__name__)


def _to_mlflow_uri(raw_uri: str) -> str:
    """Convert a raw URI or local path to a form MLflow accepts.

    On Windows, absolute paths like C:\\... have scheme 'c' which MLflow
    does not recognise. This converts them to proper file:// URIs.
    """
    parsed = urlparse(raw_uri)
    # Single-letter scheme = Windows drive letter, or no scheme = local path
    if not parsed.scheme or len(parsed.scheme) == 1:
        return Path(raw_uri).resolve().as_uri()
    return raw_uri


def configure_mlflow(
    config: PipelineConfig,
    pipeline_execution_id: str,
    config_hash: str,
) -> None:
    """Set tracking URI, experiment, and start the pipeline-level MLflow run.

    Uses thread-local active run — callers use mlflow.log_* without a run handle.
    Caller is responsible for calling mlflow.end_run() in a finally block.
    """
    if mlflow.active_run() is not None:
        _logger.warning(
            "An MLflow run was already active when configure_mlflow() was called — "
            "ending it before starting the pipeline run. This may indicate a programming error."
        )
        mlflow.end_run()

    raw_uri = config.mlflow.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
    if raw_uri:
        mlflow.set_tracking_uri(_to_mlflow_uri(raw_uri))
    else:
        # Resolve to absolute path so mlruns/ is stable regardless of cwd
        mlruns_path = Path("mlruns").resolve()
        try:
            mlruns_path.mkdir(parents=True, exist_ok=True)
            # Quick write-permission check
            probe = mlruns_path / ".write_test"
            probe.write_text("ok")
            probe.unlink()
        except OSError as e:
            raise RuntimeError(
                f"MLflow tracking fallback failed: cannot write to local tracking directory "
                f"'{mlruns_path}'. "
                f"Set config.mlflow.tracking_uri or MLFLOW_TRACKING_URI to a writable location."
            ) from e

        mlflow.set_tracking_uri(mlruns_path.as_uri())

    experiment_name = config.mlflow.experiment_name or config.project.name
    mlflow.set_experiment(experiment_name)

    run_name = f"{config.project.name}-{config.dataset}-{pipeline_execution_id[:8]}"
    mlflow.start_run(run_name=run_name)

    mlflow.set_tags({
        "pipeline.execution_id": pipeline_execution_id,
        "pipeline.project_name": config.project.name,
        "pipeline.project_version": config.project.version,
        "pipeline.task_type": config.task_type,
        "pipeline.config_hash": config_hash,
        "pipeline.dataset": config.dataset,
        "pipeline.stages": ",".join(config.pipeline_stages),
    })
    _logger.info("MLflow run started: %s (execution_id=%s)", run_name, pipeline_execution_id)


def log_training_to_mlflow(result) -> None:
    """Log training params and tags to the active MLflow run."""
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping training param/tag logging.")
        return
    params: dict[str, str] = {
        "algorithm": result.algorithm,
        "dataset_version_id": result.dataset_version_id,
        "random_seed": str(result.random_seed),
        "train_rows": str(result.train_rows),
        "trained_at": result.trained_at,
    }
    for key, val in result.hyperparameters.items():
        params[f"hp.{key}"] = "null" if val is None else str(val)

    mlflow.log_params(params)
    mlflow.set_tags({
        "training.algorithm": result.algorithm,
        "training.trained_at": result.trained_at,
    })


def log_training_artifacts_to_mlflow(model_dir: Path, preprocessed_dir: Path) -> None:
    """Log model and preprocessing artifacts to the active MLflow run."""
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping training artifact logging.")
        return
    mlflow.log_artifacts(str(model_dir), artifact_path="model")
    for fname in ("feature_map.json", "metadata.json"):
        fpath = preprocessed_dir / fname
        if fpath.exists():
            mlflow.log_artifact(str(fpath), artifact_path="preprocessing")
        else:
            _logger.warning("Preprocessing artifact not found, skipping: %s", fpath)


def log_evaluation_to_mlflow(result: dict) -> None:
    """Log evaluation metrics and tags to the active MLflow run.

    Note: pipeline_execution_id and task_type are not re-set here because this
    integration uses a single MLflow run for the entire pipeline invocation.
    Those tags are set once in configure_mlflow() and apply to all stages.
    Non-numeric metric values (e.g. "averaging": "weighted") are silently skipped.
    """
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping evaluation metric logging.")
        return
    metrics = {f"eval.{k}": float(v) for k, v in result["metrics"].items() if isinstance(v, (int, float))}
    mlflow.log_metrics(metrics)
    mlflow.set_tags({
        "eval.evaluated_at": result["evaluated_at"],
        "eval.task_type": result["task_type"],
    })


def log_comparison_to_mlflow(result: dict, output_dir: Path) -> None:
    """Log comparison tags, delta metrics, and comparison.json artifact.

    Reads the structured comparison block produced by comparator.py:
        comparison.has_production_model  — bool
        comparison.overall_verdict       — "better" / "worse" / "equal" / "mixed" / "no_baseline"
        comparison.is_better             — bool | None
        comparison.per_metric.<metric>.delta — float delta per metric
    """
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping comparison logging.")
        return
    comparison = result.get("comparison") or {}
    has_prod = comparison.get("has_production_model", False)
    is_better = comparison.get("is_better")
    overall_verdict = comparison.get("overall_verdict", "no_baseline")

    if not has_prod:
        mlflow.set_tag("is_better_than_production", "no_production_model")
    else:
        mlflow.set_tag("is_better_than_production", "true" if is_better else "false")

    mlflow.set_tag("comparison.overall_verdict", overall_verdict)

    # Log per-metric deltas from the nested per_metric structure
    per_metric = comparison.get("per_metric") or {}
    for metric_name, metric_data in per_metric.items():
        delta = metric_data.get("delta")
        if delta is not None:
            mlflow.log_metric(f"comparison.delta.{metric_name}", float(delta))

    comparison_path = Path(output_dir) / "comparison.json"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.write_text(json.dumps(comparison, indent=2))
    mlflow.log_artifact(str(comparison_path), artifact_path="evaluation")


def log_promotion_decision_to_mlflow(decision: dict) -> None:
    """Log promotion decision tags to the active MLflow run."""
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping promotion decision logging.")
        return
    mlflow.set_tag("promotion.outcome", decision["outcome"])
    mlflow.set_tag("promotion.run_id", decision["run_id"])
    if decision.get("reason"):
        mlflow.set_tag("promotion.reason", decision["reason"])


def log_drift_artifacts(drift_dir: Path) -> None:
    """Log drift report HTML/JSON to the active MLflow run if they exist.

    Best-effort: sets has_drift_report=false and logs a warning if absent.
    Path.__truediv__ works on non-existent directories, so no explicit
    drift_dir.exists() check is needed — the per-file f.exists() call handles
    both missing directories and missing files correctly.
    """
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping drift artifact logging.")
        return
    candidates = [
        drift_dir / "drift_report.html",
        drift_dir / "drift_report.json",
    ]
    found = [f for f in candidates if f.exists()]

    if not found:
        _logger.warning(
            "No drift reports found at %s — skipping drift artifact logging.", drift_dir
        )
        mlflow.set_tag("has_drift_report", "false")
        return

    for f in found:
        mlflow.log_artifact(str(f), artifact_path="drift")
    mlflow.set_tag("has_drift_report", "true")
    _logger.info("Logged %d drift artifact(s) from %s", len(found), drift_dir)
