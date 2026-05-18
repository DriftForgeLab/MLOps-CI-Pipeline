# =============================================================================
# src/pipeline/mlflow_logger.py — MLflow logging helpers
# =============================================================================
# Responsibility: Centralise every MLflow write so the orchestrator and the
# individual stages do not each re-implement tracking calls.
#
# MLflow is the system of record for this project: it stores run parameters,
# metrics, artifacts, lineage tags, and the model registry. Key helpers:
#   configure_mlflow()      — resolve the tracking URI, select the experiment,
#                             and start the single pipeline-level run that all
#                             stages log into.
#   log_training_* / etc.   — record each stage's outputs as params, metrics,
#                             and artifacts.
#
# The helpers are intentionally fault-tolerant: a failure to reach MLflow is
# logged but never aborts the pipeline, because experiment tracking is
# observability, not a correctness requirement of the run itself.
# =============================================================================

import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import mlflow

from src.config.loader import PipelineConfig
from src.registry.model_registry import MODEL_ARTIFACT_SUBPATH
from src.training import TrainingResult

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


def log_training_to_mlflow(result: TrainingResult) -> None:
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
    mlflow.log_artifacts(str(model_dir), artifact_path=MODEL_ARTIFACT_SUBPATH)
    # isp_config.json is only present for raw-image runs; skip silently if absent.
    for fname in ("feature_map.json", "metadata.json", "isp_config.json"):
        fpath = preprocessed_dir / fname
        if fpath.exists():
            mlflow.log_artifact(str(fpath), artifact_path="preprocessing")
        elif fname != "isp_config.json":
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


def log_drift_metrics_to_mlflow(drift_result: dict) -> None:
    """Log per-feature drift scores and overall drift tags to the active MLflow run.

    This function logs:
      - Per-feature metric ``drift.{feature}.score`` (one per feature in ``drift_result["features"]``)
      - Tags: ``drift.overall_severity``, ``drift.dataset_drift_detected``,
        ``drift.drifted_feature_count``, ``drift.drift_share``

    Args:
        drift_result: Standard drift result dict as returned by
                      ``src.drift.interpret.build_drift_result``.
    """
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping drift metric logging.")
        return

    features = drift_result.get("features") or {}
    per_feature_metrics = {
        f"drift.{name}.score": float(data["drift_score"])
        for name, data in features.items()
        if isinstance(data.get("drift_score"), (int, float))
    }
    if per_feature_metrics:
        mlflow.log_metrics(per_feature_metrics)

    overall = drift_result.get("overall") or {}
    mlflow.set_tags({
        "drift.overall_severity": str(overall.get("severity", "")),
        "drift.dataset_drift_detected": "true" if overall.get("dataset_drift_detected") else "false",
        "drift.drifted_feature_count": str(overall.get("drifted_feature_count", 0)),
        "drift.drift_share": str(overall.get("drift_share", 0.0)),
    })


def log_isp_versioning_to_mlflow(metadata: dict, isp_config) -> None:
    """Log ISP pipeline version and key config parameters to the active MLflow run.

    Called from the preprocessing stage for raw-image tasks (raw_input=True).
    Logs the pipeline version and preprocessing hash as params (stable,
    queryable identifiers) and key ISP settings as tags (for UI discoverability).

    Args:
        metadata:   The metadata.json dict written by run_image_preprocessing().
                    Must contain "pipeline_version" and "preprocess_hash".
        isp_config: ISPConfig dataclass from the preprocessing config, or None.
    """
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping ISP versioning logging.")
        return

    params: dict[str, str] = {}
    if "pipeline_version" in metadata:
        params["isp.pipeline_version"] = str(metadata["pipeline_version"])
    if "preprocess_hash" in metadata:
        params["preprocessing.hash"] = str(metadata["preprocess_hash"])
    if params:
        mlflow.log_params(params)

    tags: dict[str, str] = {"isp.raw_input": "true"}
    if isp_config is not None:
        tags["isp.demosaicing_algorithm"] = str(isp_config.demosaicing.algorithm)
        tags["isp.denoising_algorithm"] = str(isp_config.denoising.algorithm)
        tags["isp.sharpening_algorithm"] = str(isp_config.sharpening.algorithm)
        tags["isp.gamma"] = str(isp_config.gamma_correction.gamma)
    mlflow.set_tags(tags)


def log_isp_scenario_artifacts(scenario_dir: Path) -> None:
    """Log model analysis report HTML/JSON to the active MLflow run.

    Handles both report types depending on pipeline:
      - Raw DNG pipelines:      sensitivity_report.{json,html}
      - Standard image pipelines: robustness_report.{json,html}

    Runtime drift detection results are handled separately by the monitor CLIs
    and logged via `log_drift_metrics_to_mlflow`.
    """
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping model analysis artifact logging.")
        return

    candidates = [
        scenario_dir / "sensitivity_report.html",
        scenario_dir / "sensitivity_report.json",
        scenario_dir / "robustness_report.html",
        scenario_dir / "robustness_report.json",
    ]
    found = [f for f in candidates if f.exists()]

    if not found:
        _logger.info(
            "No model analysis reports found at %s — model analysis was likely skipped.", scenario_dir
        )
        mlflow.set_tag("has_model_analysis", "false")
        return

    for f in found:
        mlflow.log_artifact(str(f), artifact_path="model_analysis")
    mlflow.set_tag("has_model_analysis", "true")
    _logger.info("Logged %d model analysis artifact(s) from %s", len(found), scenario_dir)


def log_image_drift_metrics_to_mlflow(drift_result: dict) -> None:
    """Log image drift detection results from monitor_image_batch() to MLflow.

    Logs per-channel Wasserstein scores and the overall drift score as metrics,
    and overall severity / drift type as tags. If a scenario match is present,
    the matched scenario name, confidence, and estimated accuracy drop are also
    logged for traceability.

    Args:
        drift_result: Dict as returned by monitor_image_batch() (image_statistical
                      or image_embedding).
    """
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping image drift metric logging.")
        return

    overall = drift_result.get("overall", {})
    metrics: dict[str, float] = {}
    if "drift_score" in overall:
        metrics["drift.image.overall_score"] = float(overall["drift_score"])

    for ch, data in (drift_result.get("channels") or {}).items():
        if "drift_score" in data:
            metrics[f"drift.image.{ch}_score"] = float(data["drift_score"])

    if metrics:
        mlflow.log_metrics(metrics)

    mlflow.set_tags({
        "drift.image.method":          str(drift_result.get("method", "")),
        "drift.image.severity":        str(overall.get("severity", "")),
        "drift.image.drift_detected":  "true" if overall.get("dataset_drift_detected") else "false",
    })

    scenario_match = drift_result.get("scenario_match")
    if scenario_match:
        mlflow.set_tags({
            "drift.image.matched_scenario": str(scenario_match.get("matched_scenario", "")),
            "drift.image.match_confidence": str(scenario_match.get("confidence", "")),
        })
        drop = scenario_match.get("estimated_accuracy_drop")
        if drop is not None:
            mlflow.log_metric("drift.image.estimated_accuracy_drop", float(drop))


def log_drift_decision_to_mlflow(decision_dict: dict) -> None:
    """Log a drift response decision to the active MLflow run.

    Records the decision option and severity as MLflow params (immutable,
    queryable across runs) and the free-text reason as a tag (long text).
    Also marks the run so it is discoverable by drift decision queries.

    Args:
        decision_dict: Dict as returned by DriftDecision.to_dict(). Must
                       contain at minimum "option", "drift_severity",
                       "decided_at", and "reason".
    """
    if not mlflow.active_run():
        _logger.warning("No active MLflow run — skipping drift decision logging.")
        return

    mlflow.log_params({
        "drift.decision.option":    str(decision_dict.get("option", "")),
        "drift.decision.severity":  str(decision_dict.get("drift_severity", "")),
        "drift.decision.decided_at": str(decision_dict.get("decided_at", "")),
    })
    mlflow.set_tags({
        "drift.decision.reason":       str(decision_dict.get("reason", "")),
        "drift.decision.drift_type":   str(decision_dict.get("drift_type", "")),
        "drift.decision.report_linked": str(decision_dict.get("drift_report_linked", "")),
        "has_drift_decision": "true",
    })
