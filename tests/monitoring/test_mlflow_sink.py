"""Tests for src.monitoring.mlflow_sink.log_runtime_drift_to_mlflow."""
from __future__ import annotations

from pathlib import Path

import mlflow
import pytest

from src.monitoring.mlflow_sink import log_runtime_drift_to_mlflow


def _tabular_result(severity: str = "high") -> dict:
    return {
        "schema_version": "1.0.0",
        "drift_type": "tabular",
        "generated_at": "2026-04-14T00:00:00+00:00",
        "pipeline_execution_id": "exec-abc123",
        "dataset_version_id": "dv-xyz",
        "task_type": "classification",
        "overall": {
            "dataset_drift_detected": severity != "low",
            "drift_share": 0.5,
            "drifted_feature_count": 2,
            "total_feature_count": 4,
            "severity": severity,
        },
        "features": {
            "sepal_length": {
                "column_type": "num",
                "drift_detected": True,
                "drift_score": 0.004,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "high",
            },
            "sepal_width": {
                "column_type": "num",
                "drift_detected": False,
                "drift_score": 0.42,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "low",
            },
        },
    }


def _image_result(severity: str = "medium") -> dict:
    return {
        "schema_version": "1.0.0",
        "drift_type": "image_statistical",
        "method": "statistical",
        "generated_at": "2026-04-14T00:00:00+00:00",
        "task_type": "image_classification_cnn",
        "dataset_version_id": "dv-img-1",
        "overall": {
            "drift_score": 0.18,
            "dataset_drift_detected": True,
            "severity": severity,
        },
        "channels": {
            "R": {"drift_score": 0.22, "severity": "medium"},
            "G": {"drift_score": 0.10, "severity": "low"},
            "B": {"drift_score": 0.05, "severity": "low"},
        },
    }


def _read_run(tmp_path: Path, run_id: str):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    return mlflow.get_run(run_id)


class TestRuntimeDriftSink:
    def test_returns_run_id_and_creates_run(self, tmp_path: Path):
        run_id = log_runtime_drift_to_mlflow(
            model_name="iris_rf",
            drift_result=_tabular_result(),
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
        )
        assert isinstance(run_id, str) and run_id
        run = _read_run(tmp_path, run_id)
        assert run.info.run_id == run_id

    def test_run_name_starts_with_drift_prefix(self, tmp_path: Path):
        run_id = log_runtime_drift_to_mlflow(
            model_name="iris_rf",
            drift_result=_tabular_result(),
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
        )
        run = _read_run(tmp_path, run_id)
        assert run.info.run_name.startswith("drift-")

    def test_sets_runtime_and_model_tags(self, tmp_path: Path):
        run_id = log_runtime_drift_to_mlflow(
            model_name="iris_rf",
            drift_result=_tabular_result(),
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
        )
        tags = _read_run(tmp_path, run_id).data.tags
        assert tags["drift.runtime"] == "true"
        assert tags["drift.model"] == "iris_rf"
        assert tags["drift.type"] == "tabular"
        assert tags["drift.overall_severity"] == "high"

    def test_tabular_logs_per_feature_scores(self, tmp_path: Path):
        run_id = log_runtime_drift_to_mlflow(
            model_name="iris_rf",
            drift_result=_tabular_result(),
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
        )
        metrics = _read_run(tmp_path, run_id).data.metrics
        assert metrics["drift.sepal_length.score"] == pytest.approx(0.004)
        assert metrics["drift.sepal_width.score"] == pytest.approx(0.42)

    def test_image_logs_per_channel_scores(self, tmp_path: Path):
        run_id = log_runtime_drift_to_mlflow(
            model_name="images",
            drift_result=_image_result(),
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
        )
        run = _read_run(tmp_path, run_id)
        metrics = run.data.metrics
        assert metrics["drift.image.overall_score"] == pytest.approx(0.18)
        assert metrics["drift.image.R_score"] == pytest.approx(0.22)
        assert metrics["drift.image.G_score"] == pytest.approx(0.10)
        assert metrics["drift.image.B_score"] == pytest.approx(0.05)
        assert run.data.tags["drift.type"] == "image_statistical"

    def test_uses_drift_namespace_only_no_analysis_namespace(
        self, tmp_path: Path
    ):
        run_id = log_runtime_drift_to_mlflow(
            model_name="iris_rf",
            drift_result=_tabular_result(),
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
        )
        run = _read_run(tmp_path, run_id)
        # All drift keys live under drift.* — never analysis.*
        for k in list(run.data.metrics) + list(run.data.tags):
            assert not k.startswith("analysis."), (
                f"Runtime drift should not use analysis.* namespace, found: {k}"
            )

    def test_does_not_attach_to_active_run(self, tmp_path: Path):
        """Runtime drift must not nest inside the caller's active run."""
        mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
        mlflow.set_experiment("outer_exp")
        with mlflow.start_run(run_name="outer-training") as outer:
            run_id = log_runtime_drift_to_mlflow(
                model_name="iris_rf",
                drift_result=_tabular_result(),
                tracking_uri=(tmp_path / "mlruns").as_uri(),
                experiment_name="test_exp",
            )
            # Outer run must still be active after sink returns
            assert mlflow.active_run().info.run_id == outer.info.run_id
        # Drift run is distinct and not a child of outer
        drift_run = mlflow.get_run(run_id)
        assert drift_run.info.run_id != outer.info.run_id
        assert drift_run.data.tags.get("mlflow.parentRunId") is None

    def test_returns_none_on_mlflow_failure(
        self, tmp_path: Path, monkeypatch
    ):
        """Broken MLflow must never crash the CLI — log warning, return None."""
        def _boom(*_a, **_kw):
            raise RuntimeError("mlflow down")

        from mlflow.tracking import MlflowClient
        monkeypatch.setattr(MlflowClient, "create_run", _boom)
        result = log_runtime_drift_to_mlflow(
            model_name="iris_rf",
            drift_result=_tabular_result(),
            tracking_uri=(tmp_path / "mlruns").as_uri(),
            experiment_name="test_exp",
        )
        assert result is None
