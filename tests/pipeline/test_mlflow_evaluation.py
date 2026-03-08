import json

import pytest
import mlflow

from src.pipeline.mlflow_logger import log_evaluation_to_mlflow, log_comparison_to_mlflow

_CLASSIFICATION = {
    "model_version": "d32f10723d05", "dataset_version": "d32f10723d05",
    "task_type": "classification", "evaluated_at": "2026-03-04T12:00:00+00:00",
    "metrics": {"accuracy": 0.95, "precision": 0.94, "recall": 0.93, "f1_score": 0.935},
    "comparison": {"has_production_model": False, "is_better": None, "delta": None},
}
_REGRESSION = {
    "model_version": "d32f10723d05", "dataset_version": "d32f10723d05",
    "task_type": "regression", "evaluated_at": "2026-03-04T12:00:00+00:00",
    "metrics": {"mae": 0.5, "mse": 0.25, "rmse": 0.5, "r2": 0.88},
    "comparison": {"has_production_model": False, "is_better": None, "delta": None},
}
_WITH_COMPARISON = {
    **{k: v for k, v in _CLASSIFICATION.items() if k != "comparison"},
    "comparison": {
        "has_production_model": True,
        "is_better": True,
        "delta": {"accuracy": 0.03, "precision": 0.02, "recall": 0.01, "f1_score": 0.025},
    },
}
_NO_PRODUCTION = {
    **{k: v for k, v in _CLASSIFICATION.items() if k != "comparison"},
    "comparison": {"has_production_model": False, "is_better": None, "delta": None},
}


def test_classification_metrics_logged(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_evaluation_to_mlflow(_CLASSIFICATION)
    data = mlflow.get_run(run.info.run_id).data
    assert data.metrics["eval.accuracy"] == pytest.approx(0.95)
    assert data.metrics["eval.f1_score"] == pytest.approx(0.935)


def test_regression_metrics_logged(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_evaluation_to_mlflow(_REGRESSION)
    data = mlflow.get_run(run.info.run_id).data
    assert data.metrics["eval.mae"] == pytest.approx(0.5)
    assert data.metrics["eval.r2"] == pytest.approx(0.88)


def test_evaluation_sets_evaluated_at_tag(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_evaluation_to_mlflow(_CLASSIFICATION)
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags.get("eval.evaluated_at") == "2026-03-04T12:00:00+00:00"


def test_comparison_with_production_model(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_comparison_to_mlflow(_WITH_COMPARISON, output_dir=tmp_path)
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags["is_better_than_production"] == "true"
    assert data.metrics["comparison.delta.accuracy"] == pytest.approx(0.03)
    client = mlflow.tracking.MlflowClient()
    artifact_paths = [a.path for a in client.list_artifacts(run.info.run_id, "evaluation")]
    assert "evaluation/comparison.json" in artifact_paths


def test_comparison_no_production_model(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.set_experiment("test")
    with mlflow.start_run() as run:
        log_comparison_to_mlflow(_NO_PRODUCTION, output_dir=tmp_path)
    data = mlflow.get_run(run.info.run_id).data
    assert data.tags["is_better_than_production"] == "no_production_model"
