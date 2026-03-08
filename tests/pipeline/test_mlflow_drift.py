import mlflow
import pytest
from pathlib import Path
from src.pipeline.mlflow_logger import log_drift_artifacts


def test_logs_drift_files_when_present(tmp_path):
    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()
    (drift_dir / "drift_report.html").write_text("<html>drift</html>")
    (drift_dir / "drift_report.json").write_text('{"drift": true}')

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_drift_artifacts(drift_dir)
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags["has_drift_report"] == "true"
    client = mlflow.tracking.MlflowClient()
    top = [a.path for a in client.list_artifacts(run.info.run_id)]
    assert "drift" in top, f"Expected 'drift' dir in artifacts, got {top}"


def test_noop_when_drift_directory_absent(tmp_path):
    drift_dir = tmp_path / "drift-does-not-exist"   # intentionally missing

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_drift_artifacts(drift_dir)   # must not raise
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags.get("has_drift_report") == "false"


def test_noop_when_drift_files_absent(tmp_path):
    drift_dir = tmp_path / "drift"
    drift_dir.mkdir()   # dir exists but no report files

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_drift_artifacts(drift_dir)
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags.get("has_drift_report") == "false"
