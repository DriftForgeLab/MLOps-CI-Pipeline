"""Tests for the deployment stage that emits outputs/deployment_manifest.json.

The deployment stage is a deterministic, side-effect-free artifact producer.
It reads the promotion decision, the active MLflow run, and the model
registry's current Production version, and writes a structured manifest
describing the deployable state. It never raises under normal operation.
"""

import inspect
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.steps import _deployment_stage, execute_stage


VERSION_ID = "b6b8efc86732"

# Two literal disclaimer strings that MUST appear in manifest["notes"].
DISCLAIMER_NOTES = [
    "This manifest declares deployment metadata only. It does not provision infrastructure, build or push container images, start Docker containers, or call external services.",
    "Live serving is performed out-of-band by the FastAPI application started via 'run-api' or 'docker compose -f docker/docker-compose.yml up --build'.",
]


def _make_config(tmp_path: Path) -> MagicMock:
    """Build a minimal MagicMock standing in for a PipelineConfig.

    Only attributes actually read by the deployment stage are populated.
    """
    config = MagicMock()
    config.output_dir = str(tmp_path / "outputs")
    config.task_type = "classification"
    config.dataset = "breast_cancer"
    config.project.name = "lightweight-mlops-pipeline"
    config.configs.deployment = "src/config/deployment.yaml"
    config.mlflow.tracking_uri = None
    config.mlflow.registry_model_name = None
    return config


def _write_promotion_decision(output_dir: Path, outcome: str = "approved") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "outcome": outcome,
        "reason": None,
        "run_id": "eac1bf9ea7074c61a92f680028a2b2f3",
        "dataset_version_id": VERSION_ID,
        "metrics": {"accuracy": 0.913, "f1_score": 0.9117},
        "comparison": {"overall_verdict": "no_baseline", "has_production_model": False},
    }
    (output_dir / "promotion_decision.json").write_text(json.dumps(record, indent=2))
    return record


def _registry_version_mock(run_id: str = "eac1bf9ea7074c61a92f680028a2b2f3", version: str = "4") -> MagicMock:
    mv = MagicMock()
    mv.version = version
    mv.run_id = run_id
    return mv


def _run_with_mocks(config, version_id=VERSION_ID, *, versions=None, registry_raises=None, active_run=None):
    """Run _deployment_stage with MlflowClient.get_latest_versions and active_run patched."""
    client = MagicMock()
    if registry_raises is not None:
        client.get_latest_versions.side_effect = registry_raises
    else:
        client.get_latest_versions.return_value = versions or []
    run = MagicMock()
    run.data.tags = {
        "lineage.algorithm": "RandomForestClassifier",
        "lineage.trained_at": "2026-05-13T14:21:44.118000+00:00",
        "lineage.dataset_version_id": VERSION_ID,
    }
    client.get_run.return_value = run

    with patch("src.pipeline.steps.mlflow") as mock_mlflow:
        mock_mlflow.tracking.MlflowClient.return_value = client
        mock_mlflow.active_run.return_value = active_run
        if active_run is not None:
            mock_mlflow.get_experiment.return_value = MagicMock(name="lightweight-mlops-pipeline")
        _deployment_stage(config, version_id)


def _read_manifest(config) -> dict:
    return json.loads((Path(config.output_dir) / "deployment_manifest.json").read_text())


# ── Schema / required-fields ────────────────────────────────────────────────

class TestManifestSchema:
    def test_deployment_manifest_has_required_fields(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_promotion_decision(Path(config.output_dir))
        _run_with_mocks(config, versions=[_registry_version_mock()])
        manifest = _read_manifest(config)
        assert manifest["schema_version"] == "1.0"
        for top in [
            "generated_at",
            "pipeline",
            "model",
            "promotion",
            "service",
            "container",
            "env_template",
            "outputs",
            "readiness",
            "notes",
        ]:
            assert top in manifest, f"Missing top-level field: {top}"
        # Required sub-fields
        for key in ["task_type", "dataset_version_id", "config_dataset", "output_dir"]:
            assert key in manifest["pipeline"]
        for key in ["registry_name"]:
            assert key in manifest["model"]
        for key in [
            "app_import",
            "entrypoint_cli",
            "endpoints",
            "host",
            "port",
            "log_level",
            "allowed_stage",
            "require_production_model",
            "startup_timeout_seconds",
        ]:
            assert key in manifest["service"]
        for key in [
            "dockerfile",
            "compose_file",
            "compose_service",
            "image_name",
            "suggested_image_tag",
            "exposed_port",
        ]:
            assert key in manifest["container"]
        assert "status" in manifest["readiness"]
        assert "warnings" in manifest["readiness"]

    def test_deployment_manifest_is_valid_json(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _run_with_mocks(config, versions=[])
        manifest_path = Path(config.output_dir) / "deployment_manifest.json"
        text = manifest_path.read_text()
        json.loads(text)
        # atomic_write_json uses indent=2
        assert "\n  " in text


# ── Readiness state machine ─────────────────────────────────────────────────

class TestReadinessStateMachine:
    def test_ready_when_promotion_approved_and_registry_has_production(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_promotion_decision(Path(config.output_dir), outcome="approved")
        _run_with_mocks(config, versions=[_registry_version_mock()])
        manifest = _read_manifest(config)
        assert manifest["readiness"]["status"] == "ready"
        # warnings may include local_artifact_missing (no artifacts/runs/... on disk in tmp)
        assert "mlflow_registry_unreachable" not in manifest["readiness"]["warnings"]
        assert manifest["model"]["registry_version"] == "4"

    def test_no_production_model(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_promotion_decision(Path(config.output_dir), outcome="approved")
        _run_with_mocks(config, versions=[])
        manifest = _read_manifest(config)
        assert manifest["readiness"]["status"] == "no_production_model"
        assert manifest["model"]["registry_version"] is None
        assert manifest["model"]["stage"] is None

    def test_promotion_not_approved(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_promotion_decision(Path(config.output_dir), outcome="rejected")
        _run_with_mocks(config, versions=[_registry_version_mock()])
        manifest = _read_manifest(config)
        assert manifest["readiness"]["status"] == "promotion_not_approved"

    def test_registry_unreachable_uses_safe_defaults(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_promotion_decision(Path(config.output_dir), outcome="approved")
        _run_with_mocks(config, registry_raises=ConnectionError("registry down"))
        manifest = _read_manifest(config)
        assert manifest["readiness"]["status"] == "mlflow_unavailable"
        assert "mlflow_registry_unreachable" in manifest["readiness"]["warnings"]
        assert manifest["model"]["registry_version"] is None


# ── Promotion handling ──────────────────────────────────────────────────────

class TestPromotionHandling:
    def test_missing_promotion_decision_uses_safe_defaults(self, tmp_path: Path):
        config = _make_config(tmp_path)
        # No promotion_decision.json written
        _run_with_mocks(config, versions=[])
        manifest = _read_manifest(config)
        assert manifest["promotion"]["outcome"] is None
        assert manifest["promotion"]["run_id"] is None
        # Stage did not raise — manifest exists
        assert (Path(config.output_dir) / "deployment_manifest.json").exists()


# ── Disclaimer note ─────────────────────────────────────────────────────────

class TestDisclaimerNote:
    def test_contains_explicit_no_live_deployment_note(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _run_with_mocks(config, versions=[])
        manifest = _read_manifest(config)
        for required_note in DISCLAIMER_NOTES:
            assert required_note in manifest["notes"]


# ── Dispatcher contract ─────────────────────────────────────────────────────

class TestDispatcherContract:
    def test_signature_matches_dispatcher_contract(self):
        sig = inspect.signature(_deployment_stage)
        params = list(sig.parameters.keys())
        assert params == ["config", "version_id"]

    def test_execute_stage_deployment_returns_completed(self, tmp_path: Path):
        config = _make_config(tmp_path)
        _write_promotion_decision(Path(config.output_dir), outcome="approved")
        client = MagicMock()
        client.get_latest_versions.return_value = [_registry_version_mock()]
        client.get_run.return_value = MagicMock(data=MagicMock(tags={}))
        with patch("src.pipeline.steps.mlflow") as mock_mlflow:
            mock_mlflow.tracking.MlflowClient.return_value = client
            mock_mlflow.active_run.return_value = None
            result = execute_stage("deployment", config, version_id=VERSION_ID)
        assert result.status == "completed"
        assert (Path(config.output_dir) / "deployment_manifest.json").exists()
