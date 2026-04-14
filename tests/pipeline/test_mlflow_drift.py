import mlflow
import pytest
from pathlib import Path
from src.pipeline.mlflow_logger import log_isp_scenario_artifacts, log_drift_metrics_to_mlflow


# ---------------------------------------------------------------------------
# Helper: build a minimal drift_result dict conforming to the standard schema.
# ---------------------------------------------------------------------------

def _drift_result(
    *,
    overall_severity: str = "high",
    dataset_drift_detected: bool = True,
    drift_share: float = 0.75,
    drifted_count: int = 3,
    total_count: int = 4,
    features: dict | None = None,
) -> dict:
    if features is None:
        features = {
            "sepal_length": {
                "column_type": "num",
                "drift_detected": True,
                "drift_score": 0.0022,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "high",
            },
            "sepal_width": {
                "column_type": "num",
                "drift_detected": False,
                "drift_score": 0.3200,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "low",
            },
        }
    return {
        "schema_version": "1.0.0",
        "drift_type": "tabular",
        "generated_at": "2026-04-05T12:00:00+00:00",
        "pipeline_execution_id": "exec-abc123",
        "dataset_version_id": "dv-a1b2c3",
        "task_type": "classification",
        "reference_dataset": {"source": "train", "path": "", "row_count": 105, "feature_count": 4},
        "current_dataset": {"source": "val", "path": "", "row_count": 45, "feature_count": 4},
        "overall": {
            "dataset_drift_detected": dataset_drift_detected,
            "drift_share": drift_share,
            "drifted_feature_count": drifted_count,
            "total_feature_count": total_count,
            "severity": overall_severity,
        },
        "features": features,
        "artifacts": {},
        "config_snapshot": {},
    }


def test_logs_isp_scenario_files_when_present(tmp_path):
    scenario_dir = tmp_path / "drift_scenarios"
    scenario_dir.mkdir()
    (scenario_dir / "sensitivity_report.html").write_text("<html>sens</html>")
    (scenario_dir / "sensitivity_report.json").write_text('{"scenarios": []}')

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_isp_scenario_artifacts(scenario_dir)
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags["has_isp_scenarios"] == "true"
    client = mlflow.tracking.MlflowClient()
    top = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "isp_scenarios" in top, f"Expected 'isp_scenarios' dir in artifacts, got {top}"


def test_noop_when_scenario_directory_absent(tmp_path):
    scenario_dir = tmp_path / "drift-does-not-exist"   # intentionally missing

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_isp_scenario_artifacts(scenario_dir)   # must not raise
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags.get("has_isp_scenarios") == "false"


def test_noop_when_scenario_files_absent(tmp_path):
    scenario_dir = tmp_path / "drift_scenarios"
    scenario_dir.mkdir()   # dir exists but no report files

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_isp_scenario_artifacts(scenario_dir)
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags.get("has_isp_scenarios") == "false"


# ---------------------------------------------------------------------------
# log_drift_metrics_to_mlflow
# ---------------------------------------------------------------------------


def test_log_drift_metrics_logs_per_feature_scores(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_drift_metrics_to_mlflow(_drift_result())
    data = mlflow.get_run(run.info.run_id).data
    assert data.metrics["drift.sepal_length.score"] == pytest.approx(0.0022)
    assert data.metrics["drift.sepal_width.score"] == pytest.approx(0.3200)


def test_log_drift_metrics_sets_overall_tags(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_drift_metrics_to_mlflow(_drift_result())
    tags = mlflow.get_run(run.info.run_id).data.tags
    assert tags["drift.overall_severity"] == "high"
    assert tags["drift.dataset_drift_detected"] == "true"
    assert tags["drift.drifted_feature_count"] == "3"
    assert tags["drift.drift_share"] == "0.75"


def test_log_drift_metrics_false_when_no_drift(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    result = _drift_result(
        dataset_drift_detected=False,
        overall_severity="low",
        drift_share=0.0,
        drifted_count=0,
    )
    with mlflow.start_run() as run:
        log_drift_metrics_to_mlflow(result)
    tags = mlflow.get_run(run.info.run_id).data.tags
    assert tags["drift.dataset_drift_detected"] == "false"
    assert tags["drift.overall_severity"] == "low"


def test_log_drift_metrics_noop_when_no_active_run(tmp_path):
    """Must not raise if called with no active run."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    # No mlflow.start_run() — should warn and return cleanly.
    log_drift_metrics_to_mlflow(_drift_result())


def test_log_drift_metrics_handles_empty_features(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    result = _drift_result(features={})
    with mlflow.start_run() as run:
        log_drift_metrics_to_mlflow(result)
    # Should still set tags; no per-feature metrics.
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags["drift.overall_severity"] == "high"
    assert not any(k.startswith("drift.") and k.endswith(".score") for k in data.metrics)
