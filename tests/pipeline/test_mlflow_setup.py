import pytest
from pathlib import Path
from src.config.loader import load_config


def _write_minimal_config(tmp_path: Path, mlflow_block: str = "") -> Path:
    cfg = tmp_path / "pipeline.yaml"
    content = (
        "project:\n"
        "  name: test-project\n"
        "  version: 0.1.0\n"
        "task_type: classification\n"
        "random_seed: 42\n"
        "pipeline_stages: [preprocessing, training]\n"
        "output_dir: outputs\n"
        "dataset: iris\n"
        "data:\n"
        "  raw: data/raw\n"
        "  processed: data/processed\n"
        "  evaluation: data/evaluation\n"
        "  drift_scenarios: data/drift\n"
        "configs:\n"
        "  preprocessing: src/config/preprocessing.yaml\n"
        "  training: src/config/training_classification.yaml\n"
        "  evaluation: src/config/evaluation.yaml\n"
        "  deployment: src/config/deployment.yaml\n"
        "log_level: INFO\n"
    )
    if mlflow_block:
        content += mlflow_block + "\n"
    cfg.write_text(content)
    return cfg


def test_mlflow_config_defaults_to_none(tmp_path):
    cfg = _write_minimal_config(tmp_path)
    config = load_config(cfg)
    assert config.mlflow.tracking_uri is None
    assert config.mlflow.experiment_name is None


def test_mlflow_config_explicit_values(tmp_path):
    block = "mlflow:\n  tracking_uri: http://localhost:5000\n  experiment_name: my-exp"
    cfg = _write_minimal_config(tmp_path, mlflow_block=block)
    config = load_config(cfg)
    assert config.mlflow.tracking_uri == "http://localhost:5000"
    assert config.mlflow.experiment_name == "my-exp"


import mlflow
import os
from src.pipeline.mlflow_logger import configure_mlflow


def test_configure_mlflow_uses_config_uri(tmp_path):
    tracking_dir = str(tmp_path / "mlruns")
    block = f"mlflow:\n  tracking_uri: {tracking_dir}\n  experiment_name: test-exp"
    cfg = _write_minimal_config(tmp_path, mlflow_block=block)
    config = load_config(cfg)
    configure_mlflow(config, "exec-001", "sha256:abc123")
    # configure_mlflow converts local paths to file:// URIs for cross-platform compatibility
    from pathlib import Path as _Path
    assert mlflow.get_tracking_uri() == _Path(tracking_dir).resolve().as_uri()
    mlflow.end_run()


def test_configure_mlflow_falls_back_to_env_var(tmp_path, monkeypatch):
    tracking_dir = str(tmp_path / "mlruns-env")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_dir)
    cfg = _write_minimal_config(tmp_path)
    config = load_config(cfg)
    configure_mlflow(config, "exec-002", "sha256:abc123")
    from pathlib import Path as _Path
    assert mlflow.get_tracking_uri() == _Path(tracking_dir).resolve().as_uri()
    mlflow.end_run()


def test_configure_mlflow_creates_experiment_from_project_name(tmp_path):
    tracking_dir = str(tmp_path / "mlruns")
    mlflow.set_tracking_uri(tracking_dir)
    cfg = _write_minimal_config(tmp_path)
    config = load_config(cfg)
    configure_mlflow(config, "exec-003", "sha256:abc123")
    exp = mlflow.get_experiment_by_name("test-project")
    assert exp is not None, "Experiment should be created with project.name"
    mlflow.end_run()
