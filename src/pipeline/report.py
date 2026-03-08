# =============================================================================
# src/pipeline/report.py — Run report generation
# =============================================================================
# Responsibility: Build and write the run_report.json artifact that captures
# what happened during a pipeline execution. This is the primary traceability
# artifact for Sprint 1 and will be extended in later sprints.
#
# This module is separate from run_pipeline.py because report generation is
# a distinct concern that will grow (MLflow integration, drift results,
# promotion decisions, etc.)
#
# Design: Pure functions for building the report dict, one I/O function for
# writing it. This separation makes the logic easy to test without touching
# the filesystem.
# =============================================================================

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.pipeline.steps import StageResult

logger = logging.getLogger(__name__)


def compute_config_hash(config_path: Path) -> str:
    """
    Compute a SHA-256 hash of the raw config file content.

    Hashing the raw file bytes (not the parsed Python object) ensures a
    deterministic fingerprint that is stable across Python versions,
    dataclass changes, and dict ordering differences.

    Args:
        config_path: Resolved, absolute path to the YAML config file.

    Returns:
        String on the form "sha256:<64-char hex digest>".
    """
    raw_bytes = config_path.read_bytes()
    hex_digest = hashlib.sha256(raw_bytes).hexdigest()
    return f"sha256:{hex_digest}"


def build_run_report(
    project_name: str,
    project_version: str,
    config_hash: str,
    task_type: str,
    random_seed: int,
    dataset_version_id: str,
    stage_results: list[StageResult],
    artifact_path: str | None = None,
    pipeline_execution_id: str | None = None,
    mlflow_run_id: str | None = None,
) -> dict:
    """
    Build the run report as a plain dict ready for JSON serialization.

    Pure function — no I/O, no side effects, no logging. Takes primitive
    values and StageResult objects, returns a dict. Easy to test by
    asserting on the returned dict without mocking anything.
    
    Returns:
        Dict matching the run_report.json schema.
    """
    all_completed = all(r.status == "completed" for r in stage_results)

    return {
        "project_id": f"{project_name}-v{project_version}",
        "config_hash": config_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type,
        "random_seed": random_seed,
        "dataset_version_id": dataset_version_id,
        "artifact_path": artifact_path,
        "executed_stages": [asdict(r) for r in stage_results],
        "overall_status": "completed" if all_completed else "failed",
        "pipeline_execution_id": pipeline_execution_id,
        "mlflow_run_id": mlflow_run_id,
    }


def write_run_report(report: dict, output_dir: str) -> Path:
    """
    Write the run report dict as JSON to output_dir/run_report.json.

    Creates output_dir (and any parent directories) if they do not exist.
    Overwrites any previous run_report.json in the same directory.

    Args:
        report:     Dict from build_run_report().
        output_dir: Directory path string (from config.output_dir).

    Returns:
        Resolved Path to the written run_report.json file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    report_file = out_path / "run_report.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("Run report written to: %s", report_file.resolve())
    return report_file.resolve()