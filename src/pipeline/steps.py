# =============================================================================
# src/pipeline/steps.py — Pipeline stage execution
# =============================================================================
# Responsibility: Execute individual pipeline stages with structured logging
# and timing. Each stage is wrapped with START/END markers for observability.
#
# Sprint 1 scope: Stages are placeholders (no-ops). The execution framework
# is real — START/END markers, timing, status tracking, error handling.
# Later sprints will register actual stage implementations (training, etc.)
#
# Design: Each stage returns a StageResult dataclass capturing what happened.
# This feeds into the run report for traceability.
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config.loader import PipelineConfig, load_promotion_config
from src.data.preprocess import run_preprocessing

from src.training.classification.train import run_training as run_classification_training
from src.training.regression.train import run_training as run_regression_training
from src.registry.metadata import save_model_artifact

from src.evaluation.evaluate import evaluate
from src.common.io import atomic_write_json
from src.pipeline.mlflow_logger import (
    log_training_to_mlflow,
    log_training_artifacts_to_mlflow,
    log_evaluation_to_mlflow,
    log_comparison_to_mlflow,
    log_promotion_decision_to_mlflow
)

from src.promotion.rules import run_promotion_rules
from src.promotion.approval import request_approval
import json
import mlflow

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    """_summary_
    Captures the result of executing a single pipeline stage.

    Attributes:
        stage:            Name of the stage that was executed (ex. "training")
        status:           Execution status ("completed" or "failed")
        started_at:       ISO 8601 timestamp when stage began
        ended_at:         ISO 8601 timestamp when stage ended
        duration_seconds: Wall-clock time elapsed during execution
        error:            Exception message if status is "failed", else None
    """
    stage: str
    status: str
    started_at: str
    ended_at: str
    duration_seconds: float
    error: str | None = None

def _placeholder_stage(config: PipelineConfig, version_id: str) -> None:
    logger.info("  (no-op placeholder — Sprint 1)")


def _preprocessing_stage(config: PipelineConfig, version_id: str) -> None:
    run_preprocessing(
        dataset_name=config.dataset,
        version_id=version_id,
        prep_config_path=Path(config.configs.preprocessing),
        processed_dir=Path(config.data.processed),
    )


def _training_stage(config: PipelineConfig, version_id: str) -> None:
    if config.task_type == "classification":
        result = run_classification_training(config, version_id)
    elif config.task_type == "regression":
        result = run_regression_training(config, version_id)
    else:
        raise ValueError(f"Unsupported task_type: '{config.task_type}'")

    artifact_path = save_model_artifact(result, run_id=version_id)
    log_training_to_mlflow(result)
    preprocessed_dir = (
        Path(config.data.processed) / config.dataset / version_id / "preprocessed"
    )
    log_training_artifacts_to_mlflow(model_dir=artifact_path, preprocessed_dir=preprocessed_dir)
    
    
def _evaluation_stage(config: PipelineConfig, version_id: str) -> None:
    report = evaluate(config, version_id)
    report_path = Path(config.output_dir) / "evaluation_report.json"
    atomic_write_json(report_path, report)
    logger.info("  Evaluation report written to: %s", report_path.resolve())
    log_evaluation_to_mlflow(report)
    log_comparison_to_mlflow(report, output_dir=Path(config.output_dir))
    

def _promotion_stage(config: PipelineConfig, version_id: str) -> None:
    # Load evaluation report
    report_path = Path(config.output_dir) / "evaluation_report.json"
    if not report_path.exists():
        raise FileNotFoundError(
            f"Evaluation report not found at '{report_path}'. "
            "Run the evaluation stage before promotion."
        )
    with open(report_path) as f:
        report = json.load(f)

    # Run threshold rules
    promotion_config = load_promotion_config(Path(config.configs.promotion))
    violations = run_promotion_rules(report["metrics"], config.task_type, promotion_config)

    if violations:
        violation_lines = "\n  ".join(
            f"[{v['rule_id']}] {v['metric']}={v['observed']} "
            f"(required {v['operator']} {v['threshold']}): {v['description']}"
            for v in violations
        )
        raise ValueError(
            f"Promotion blocked — {len(violations)} rule(s) failed:\n  {violation_lines}"
        )

    # Log comparison summary
    comparison = report.get("comparison", {})
    overall_verdict = comparison.get("overall_verdict", "no_baseline")
    logger.info(
        "  Comparison verdict: %s (has_production_model=%s)",
        overall_verdict.upper(),
        comparison.get("has_production_model", False),
    )

    logger.info("  All promotion rules passed — candidate is eligible for promotion.")

    result = request_approval(report)

    mlflow_run_id = mlflow.active_run().info.run_id if mlflow.active_run() else None
    decision = {
        "outcome":            "approved" if result.approved else "rejected",
        "reason":             result.reason,
        "run_id":             mlflow_run_id,
        "dataset_version_id": version_id,
        "metrics":            report.get("metrics", {}),
        "comparison":         report.get("comparison", {}),
    }

    decision_path = Path(config.output_dir) / "promotion_decision.json"
    atomic_write_json(decision_path, decision)
    logger.info("  Promotion decision written to: %s", decision_path.resolve())

    log_promotion_decision_to_mlflow(decision)

    if mlflow.active_run():
        mlflow.log_artifact(str(decision_path), artifact_path="promotion")

    if not result.approved:
        reason_msg = f" Reason: {result.reason}" if result.reason else " No reason provided."
        raise ValueError(f"Promotion rejected by user.{reason_msg}")

    # Production model registration deferred to ID 9 (MLflow Model Registry).
    # ID 9 will register approved models in MLflow and manage production state. (remove comments when done)


_STAGE_REGISTRY: dict[str, callable] = {
    "preprocessing": _preprocessing_stage,
    "training":      _training_stage,
    "evaluation":    _evaluation_stage,
    "promotion":     _promotion_stage,
    "deployment":    _placeholder_stage,
}


def execute_stage(stage_name: str, config: PipelineConfig, version_id: str) -> StageResult:
    """
    Execute a single pipeline stage with START/END markers and timing.
    
    Looks up the stage function in _STAGE_REGISTRY, calls it with config,
    and wraps the call in structured logging and timing. Exceptions are
    caught and recorded — the END marker is always logged.
    
    Args:
        stage_name: Must be a key in _STAGE_REGISTRY (e.g. "training")
        config:     Validated, immutable PipelineConfig
        
    Returns:
        StageResult capturing timing, status, and optional error info.
        
    Raises:
        KeyError: If stage_name is not found in _STAGE_REGISTRY.
    """
    start = datetime.now(timezone.utc)
    logger.info("===== START: %s =====", stage_name)
    
    try:
        stage_fn = _STAGE_REGISTRY[stage_name]
        stage_fn(config, version_id)
        status = "completed"
        error = None
    except KeyError:
        raise KeyError(
            f"Unknown pipeline stage: '{stage_name}'. "
            f"Registered stages: {', '.join(sorted(_STAGE_REGISTRY))}"
        )
    except Exception as e:
        status = "failed"
        error = str(e)
        logger.error("  Stage '%s' failed: %s", stage_name, e)
        
    end = datetime.now(timezone.utc)
    duration = (end - start).total_seconds()
    
    logger.info("===== END: %s (%s, %.3fs) =====", stage_name, status, duration)
    
    return StageResult(
        stage=stage_name,
        status=status,
        started_at=start.isoformat(),
        ended_at=end.isoformat(),
        duration_seconds=duration,
        error=error,
    )
    