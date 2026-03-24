import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.deployment.startup_checks import (
    ProductionModelInfo,
    _find_all_production_versions,
    _load_single_production_model,
    load_all_production_models,
    validate_environment,
)


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_config(deployment_path="src/config/deployment.yaml"):
    """Return a minimal PipelineConfig-like object."""
    return SimpleNamespace(
        configs=SimpleNamespace(deployment=deployment_path),
    )


def _make_deploy_config(require_production=True, allowed_stage="Production"):
    return SimpleNamespace(
        model=SimpleNamespace(
            require_production_model=require_production,
            allowed_stage=allowed_stage,
        ),
    )


def _make_prod_version(run_id="run-abc", version="3", tags=None):
    """Return a mock MLflow model version."""
    mv = SimpleNamespace(
        run_id=run_id,
        version=version,
        tags=tags or {
            "lineage.algorithm": "random_forest",
            "lineage.task_type": "classification",
            "lineage.trained_at": "2026-03-15",
            "lineage.dataset_version_id": "v42",
            "lineage.promotion_outcome": "approved",
        },
    )
    return mv


def _make_run(dataset_version_id="v42", dataset_name="iris"):
    """Return a mock MLflow run with standard tags."""
    return SimpleNamespace(
        data=SimpleNamespace(
            tags={
                "pipeline.dataset_version_id": dataset_version_id,
                "pipeline.dataset": dataset_name,
            }
        )
    )


def _make_registered_model(name):
    return SimpleNamespace(name=name)


# ── validate_environment ────────────────────────────────────────────────────

class TestValidateEnvironment:
    def test_config_file_missing_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(tmp_path / "nonexistent.yaml"))
        with pytest.raises(RuntimeError, match="Pipeline config not found"):
            validate_environment()

    @patch("src.deployment.startup_checks.load_config")
    def test_config_file_invalid_raises(self, mock_load, monkeypatch, tmp_path):
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text("bad")
        monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(cfg_path))
        mock_load.side_effect = ValueError("parse error")
        with pytest.raises(RuntimeError, match="Failed to load pipeline config"):
            validate_environment()

    @patch("src.deployment.startup_checks.resolve_tracking_uri", return_value="file:///tmp")
    @patch("src.deployment.startup_checks.load_deployment_config")
    @patch("src.deployment.startup_checks.load_config")
    def test_deployment_config_missing_raises(
        self, mock_load_cfg, mock_load_deploy, mock_uri, monkeypatch, tmp_path
    ):
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text("ok")
        monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(cfg_path))
        mock_load_cfg.return_value = _make_config(
            deployment_path=str(tmp_path / "no_deploy.yaml")
        )
        with pytest.raises(RuntimeError, match="Deployment config not found"):
            validate_environment()

    @patch("src.deployment.startup_checks.resolve_tracking_uri", return_value="file:///tmp")
    @patch("src.deployment.startup_checks.load_deployment_config")
    @patch("src.deployment.startup_checks.load_config")
    def test_valid_environment_returns_config(
        self, mock_load_cfg, mock_load_deploy, mock_uri, monkeypatch, tmp_path
    ):
        cfg_path = tmp_path / "pipeline.yaml"
        cfg_path.write_text("ok")
        deploy_path = tmp_path / "deployment.yaml"
        deploy_path.write_text("ok")
        monkeypatch.setenv("PIPELINE_CONFIG_PATH", str(cfg_path))

        config = _make_config(deployment_path=str(deploy_path))
        mock_load_cfg.return_value = config
        mock_load_deploy.return_value = _make_deploy_config()

        result_config, result_deploy = validate_environment()
        assert result_config is config
        assert result_deploy is mock_load_deploy.return_value


# ── _find_all_production_versions ───────────────────────────────────────────

class TestFindAllProductionVersions:
    def test_explicit_model_name_found(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_MODEL_NAME", "my-model")
        client = MagicMock()
        pv = _make_prod_version()
        client.get_latest_versions.return_value = [pv]

        result = _find_all_production_versions(client, _make_config())
        assert result == [("my-model", pv)]
        client.get_latest_versions.assert_called_once_with("my-model", stages=["Production"])

    def test_explicit_model_name_not_found_raises(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_MODEL_NAME", "missing-model")
        client = MagicMock()
        client.get_latest_versions.return_value = []

        with pytest.raises(RuntimeError, match="No Production model found"):
            _find_all_production_versions(client, _make_config())

    def test_scan_finds_production_models(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
        client = MagicMock()
        rm1 = _make_registered_model("model-a")
        rm2 = _make_registered_model("model-b")
        client.search_registered_models.return_value = [rm1, rm2]
        pv1 = _make_prod_version(run_id="r1")
        pv2 = _make_prod_version(run_id="r2")
        client.get_latest_versions.side_effect = [[pv1], [pv2]]

        result = _find_all_production_versions(client, _make_config())
        assert len(result) == 2
        assert result[0] == ("model-a", pv1)
        assert result[1] == ("model-b", pv2)

    def test_scan_finds_nothing_raises(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
        client = MagicMock()
        client.search_registered_models.return_value = [_make_registered_model("m")]
        client.get_latest_versions.return_value = []

        with pytest.raises(RuntimeError, match="No Production model found anywhere"):
            _find_all_production_versions(client, _make_config())

    def test_registry_unreachable_raises(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_MODEL_NAME", raising=False)
        client = MagicMock()
        client.search_registered_models.side_effect = ConnectionError("offline")

        with pytest.raises(RuntimeError, match="Cannot search MLflow"):
            _find_all_production_versions(client, _make_config())


# ── _load_single_production_model ───────────────────────────────────────────

class TestLoadSingleProductionModel:
    def test_loads_joblib_model(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        version_id = "v42"
        dataset_name = "iris"

        # Create model artifact
        model_dir = tmp_path / "artifacts" / "runs" / version_id / "model"
        model_dir.mkdir(parents=True)
        import joblib
        joblib.dump({"dummy": "model"}, model_dir / "model.joblib")

        # Create feature map
        fm_dir = tmp_path / "data" / "processed" / dataset_name / version_id / "preprocessed"
        fm_dir.mkdir(parents=True)
        (fm_dir / "feature_map.json").write_text(
            json.dumps({"output_features": ["f1", "f2"]})
        )

        client = MagicMock()
        client.get_run.return_value = _make_run(version_id, dataset_name)
        pv = _make_prod_version()

        info = _load_single_production_model(client, "iris-clf", pv)
        assert isinstance(info, ProductionModelInfo)
        assert info.model == {"dummy": "model"}
        assert info.feature_names == ["f1", "f2"]
        assert info.model_name == "iris-clf"
        assert info.stage == "Production"
        assert info.model_format == "sklearn"

    def test_loads_pytorch_model(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        version_id = "v42"
        dataset_name = "iris"

        model_dir = tmp_path / "artifacts" / "runs" / version_id / "model"
        model_dir.mkdir(parents=True)
        import torch
        torch.save({"pt": "model"}, model_dir / "model.pt")

        client = MagicMock()
        client.get_run.return_value = _make_run(version_id, dataset_name)
        pv = _make_prod_version()

        info = _load_single_production_model(client, "img-clf", pv)
        # With the torch stub, torch.load returns None; with real torch, it loads
        assert isinstance(info, ProductionModelInfo)
        assert info.model_format == "pytorch"

    def test_no_model_file_raises(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        version_id = "v42"

        # Create directory but no model files
        model_dir = tmp_path / "artifacts" / "runs" / version_id / "model"
        model_dir.mkdir(parents=True)

        client = MagicMock()
        client.get_run.return_value = _make_run(version_id, "iris")
        pv = _make_prod_version()

        with pytest.raises(RuntimeError, match="Model artifact not found"):
            _load_single_production_model(client, "m", pv)

    def test_missing_dataset_version_id_raises(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        client = MagicMock()
        run = SimpleNamespace(data=SimpleNamespace(tags={}))
        client.get_run.return_value = run
        pv = _make_prod_version()

        with pytest.raises(RuntimeError, match="dataset_version_id"):
            _load_single_production_model(client, "m", pv)

    def test_feature_map_missing_no_crash(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        version_id = "v42"

        model_dir = tmp_path / "artifacts" / "runs" / version_id / "model"
        model_dir.mkdir(parents=True)
        import joblib
        joblib.dump("model", model_dir / "model.joblib")

        # No feature_map.json created
        client = MagicMock()
        client.get_run.return_value = _make_run(version_id, "iris")
        pv = _make_prod_version()

        info = _load_single_production_model(client, "m", pv)
        assert info.feature_names == []
        assert info.image_shape is None

    def test_feature_map_corrupted_no_crash(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        version_id = "v42"
        dataset_name = "iris"

        model_dir = tmp_path / "artifacts" / "runs" / version_id / "model"
        model_dir.mkdir(parents=True)
        import joblib
        joblib.dump("model", model_dir / "model.joblib")

        fm_dir = tmp_path / "data" / "processed" / dataset_name / version_id / "preprocessed"
        fm_dir.mkdir(parents=True)
        (fm_dir / "feature_map.json").write_text("NOT-VALID-JSON{{{")

        client = MagicMock()
        client.get_run.return_value = _make_run(version_id, dataset_name)
        pv = _make_prod_version()

        info = _load_single_production_model(client, "m", pv)
        assert info.feature_names == []


# ── load_all_production_models ──────────────────────────────────────────────

class TestLoadAllProductionModels:
    @patch("src.deployment.startup_checks._load_single_production_model")
    @patch("src.deployment.startup_checks._find_all_production_versions")
    @patch("src.deployment.startup_checks.get_mlflow_client")
    def test_partial_failure_skips_bad_model(
        self, mock_client, mock_find, mock_load
    ):
        mock_client.return_value = MagicMock()
        pv1 = _make_prod_version(run_id="r1")
        pv2 = _make_prod_version(run_id="r2")
        mock_find.return_value = [("good-model", pv1), ("bad-model", pv2)]

        good_info = MagicMock(spec=ProductionModelInfo)
        good_info.model_version = "1"
        good_info.algorithm = "rf"
        mock_load.side_effect = [good_info, RuntimeError("corrupt")]

        result = load_all_production_models(_make_config(), _make_deploy_config())
        assert "good-model" in result
        assert "bad-model" not in result

    @patch("src.deployment.startup_checks._load_single_production_model")
    @patch("src.deployment.startup_checks._find_all_production_versions")
    @patch("src.deployment.startup_checks.get_mlflow_client")
    def test_all_fail_raises(self, mock_client, mock_find, mock_load):
        mock_client.return_value = MagicMock()
        mock_find.return_value = [("m1", _make_prod_version())]
        mock_load.side_effect = RuntimeError("fail")

        with pytest.raises(RuntimeError, match="No Production models could be loaded"):
            load_all_production_models(_make_config(), _make_deploy_config())

    @patch("src.deployment.startup_checks._find_all_production_versions")
    @patch("src.deployment.startup_checks.get_mlflow_client")
    def test_require_false_allows_empty(self, mock_client, mock_find):
        """When require_production_model=False, zero models is not an error."""
        mock_client.return_value = MagicMock()
        mock_find.side_effect = RuntimeError("No Production model found anywhere")

        deploy = _make_deploy_config(require_production=False)
        result = load_all_production_models(_make_config(), deploy)
        assert result == {}
