"""Tests for src/registry/model_registry.py — MLflow Model Registry integration."""

import pytest
import mlflow
import sklearn.ensemble
import joblib
from pathlib import Path

from src.config.loader import load_config
from src.registry.model_registry import (
    resolve_model_name,
    get_mlflow_client,
    is_run_already_registered,
    register_approved_model,
    promote_to_production,
    _build_lineage_tags,
    attach_lineage_tags,
    get_production_model_metrics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, task_type: str = "classification", registry_model_name: str | None = None) -> Path:
    """Write a minimal pipeline.yaml with the given settings."""
    cfg = tmp_path / "pipeline.yaml"
    mlflow_block = f"  tracking_uri: {(tmp_path / 'mlruns').as_uri()}\n"
    if registry_model_name:
        mlflow_block += f"  registry_model_name: {registry_model_name}\n"
    content = (
        "project:\n"
        "  name: test-project\n"
        "  version: 0.1.0\n"
        f"task_type: {task_type}\n"
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
        "  promotion: src/config/promotion.yaml\n"
        "log_level: INFO\n"
        "mlflow:\n"
        f"{mlflow_block}"
    )
    cfg.write_text(content)
    return cfg


def _log_fake_model_artifact(tmp_path: Path, run_id_ref: list) -> str:
    """Start an MLflow run, log a fake model artifact, end it, return run_id."""
    model_dir = tmp_path / "fake_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    # Create a real (tiny) sklearn model so MLflow doesn't complain
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier(n_estimators=2, random_state=42).fit(X, y)
    joblib.dump(model, model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text('{"algorithm":"random_forest"}')

    with mlflow.start_run() as run:
        mlflow.log_params({
            "algorithm": "random_forest",
            "train_rows": "104",
            "trained_at": "2026-03-10T10:00:00+00:00",
        })
        mlflow.set_tags({
            "pipeline.execution_id": "exec-test-001",
            "pipeline.config_hash": "sha256:testconfighash",
            "pipeline.dataset_version_id": "testversion001",
            "pipeline.task_type": "classification",
        })
        mlflow.log_metrics({
            "eval.accuracy": 0.95,
            "eval.f1_score": 0.94,
            "eval.precision": 0.93,
            "eval.recall": 0.92,
        })
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        run_id_ref.append(run.info.run_id)
    return run_id_ref[0]


# ---------------------------------------------------------------------------
# Unit tests — resolve_model_name
# ---------------------------------------------------------------------------

def test_resolve_model_name_default(tmp_path):
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)
    assert resolve_model_name(config) == "test-project-classification"


def test_resolve_model_name_explicit(tmp_path):
    cfg_path = _write_config(tmp_path, registry_model_name="my-custom-model")
    config = load_config(cfg_path)
    assert resolve_model_name(config) == "my-custom-model"


def test_resolve_model_name_regression(tmp_path):
    cfg_path = _write_config(tmp_path, task_type="regression")
    config = load_config(cfg_path)
    assert resolve_model_name(config) == "test-project-regression"


# ---------------------------------------------------------------------------
# Unit tests — _build_lineage_tags
# ---------------------------------------------------------------------------

def _make_fake_run(the_run_id: str, task_type: str = "classification"):
    """Create a minimal fake run object for unit-testing _build_lineage_tags."""
    class FakeRunData:
        tags = {
            "pipeline.execution_id": "exec-001",
            "pipeline.config_hash": "sha256:abc",
            "pipeline.dataset_version_id": "ver001",
        }
        params = {
            "algorithm": "random_forest",
            "train_rows": "104",
            "trained_at": "2026-03-10T10:00:00+00:00",
        }
        metrics = {
            "eval.accuracy": 0.95,
            "eval.f1_score": 0.94,
            "eval.precision": 0.93,
            "eval.recall": 0.92,
        }

    class FakeRunInfo:
        pass

    info = FakeRunInfo()
    info.run_id = the_run_id

    class FakeRun:
        data = FakeRunData()

    fake_run = FakeRun()
    fake_run.info = info
    return fake_run


def test_build_lineage_tags_classification_keys(tmp_path):
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)
    run = _make_fake_run("test-run-id")
    report = {"comparison": {"overall_verdict": "no_baseline"}}
    decision = {"outcome": "approved"}

    tags = _build_lineage_tags(config, run, report, decision)

    # Required identity tags
    assert tags["lineage.run_id"] == "test-run-id"
    assert tags["lineage.config_hash"] == "sha256:abc"
    assert tags["lineage.dataset_version_id"] == "ver001"
    assert tags["lineage.task_type"] == "classification"
    assert tags["lineage.algorithm"] == "random_forest"
    assert tags["lineage.trained_at"] == "2026-03-10T10:00:00+00:00"
    assert tags["lineage.promotion_outcome"] == "approved"

    # Classification eval keys present
    assert "lineage.eval.accuracy" in tags
    assert "lineage.eval.f1_score" in tags
    assert "lineage.eval.precision" in tags
    assert "lineage.eval.recall" in tags

    # Regression eval keys NOT present
    assert "lineage.eval.r2" not in tags
    assert "lineage.eval.mae" not in tags


def test_build_lineage_tags_regression_keys(tmp_path):
    cfg_path = _write_config(tmp_path, task_type="regression")
    config = load_config(cfg_path)

    class FakeRunData:
        tags = {"pipeline.execution_id": "x", "pipeline.config_hash": "h", "pipeline.dataset_version_id": "v"}
        params = {"algorithm": "linear_regression", "train_rows": "80", "trained_at": "2026-03-10T10:00:00+00:00"}
        metrics = {"eval.r2": 0.88, "eval.mae": 0.12, "eval.mse": 0.02, "eval.rmse": 0.14}
    class FakeRunInfo:
        run_id = "reg-run-id"
    class FakeRun:
        data = FakeRunData()
        info = FakeRunInfo()

    tags = _build_lineage_tags(config, FakeRun(), {}, {"outcome": "approved"})

    assert "lineage.eval.r2" in tags
    assert "lineage.eval.mae" in tags
    # Classification keys NOT present
    assert "lineage.eval.accuracy" not in tags
    assert "lineage.eval.f1_score" not in tags


def test_build_lineage_tags_raises_when_required_tags_missing(tmp_path):
    """Missing required lineage tags must raise ValueError — not silently emit a warning."""
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    class FakeRunData:
        tags = {}   # No pipeline tags -> all derived fields are empty
        params = {}
        metrics = {}
    class FakeRunInfo:
        run_id = "run-001"
    class FakeRun:
        data = FakeRunData()
        info = FakeRunInfo()

    with pytest.raises(ValueError, match="required tags are missing or empty"):
        _build_lineage_tags(config, FakeRun(), {}, {})


def test_build_lineage_tags_excludes_averaging(tmp_path):
    """Ensure the 'averaging' non-numeric metric is never included."""
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    class FakeRunData:
        tags = {"pipeline.execution_id": "x", "pipeline.config_hash": "h", "pipeline.dataset_version_id": "v"}
        params = {"algorithm": "random_forest", "train_rows": "100", "trained_at": "2026-03-10T00:00:00+00:00"}
        # Note: "averaging" is NOT logged with "eval." prefix in mlflow_logger.py — safe by design
        metrics = {"eval.accuracy": 0.9, "eval.f1_score": 0.88}
    class FakeRunInfo:
        run_id = "no-averaging-run"
    class FakeRun:
        data = FakeRunData()
        info = FakeRunInfo()

    tags = _build_lineage_tags(config, FakeRun(), {}, {"outcome": "approved"})
    assert "lineage.eval.averaging" not in tags


def test_build_lineage_tags_raises_on_unknown_task_type(tmp_path):
    """An unsupported task_type must raise ValueError — not silently use regression keys."""
    cfg_path = _write_config(tmp_path, task_type="classification")
    config = load_config(cfg_path)

    # Patch config with a fake task_type by using a minimal stand-in
    from dataclasses import replace
    bad_config = replace(config, task_type="anomaly_detection")

    run = _make_fake_run("run-unknown-task")
    with pytest.raises(ValueError, match="Unsupported task_type"):
        _build_lineage_tags(bad_config, run, {}, {"outcome": "approved"})


# ---------------------------------------------------------------------------
# Integration tests — require real MLflow tracking store
# ---------------------------------------------------------------------------

def test_is_run_already_registered_false(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)
    client = get_mlflow_client(config)
    # Fresh store — no registrations
    result = is_run_already_registered(client, "nonexistent-run-id", "test-model")
    assert result is False


def test_register_approved_model_creates_version(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    run_id_ref = []
    run_id = _log_fake_model_artifact(tmp_path, run_id_ref)

    model_version = register_approved_model(config, run_id)

    assert model_version is not None
    assert model_version.run_id == run_id
    model_name = resolve_model_name(config)
    assert model_version.name == model_name
    assert str(model_version.version) == "1"


def test_register_approved_model_idempotent(tmp_path):
    """Calling register_approved_model twice for the same run returns existing version, no duplicate."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    run_id_ref = []
    run_id = _log_fake_model_artifact(tmp_path, run_id_ref)

    v1 = register_approved_model(config, run_id)
    v2 = register_approved_model(config, run_id)

    # Same version number — no duplicate created
    assert v1.version == v2.version

    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    all_versions = client.search_model_versions(f"name='{model_name}'")
    assert len(all_versions) == 1


def test_promote_to_production_first_model(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    run_id_ref = []
    run_id = _log_fake_model_artifact(tmp_path, run_id_ref)
    model_version = register_approved_model(config, run_id)

    promote_to_production(config, model_version.version, run_id)

    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    assert len(production_versions) == 1
    assert production_versions[0].version == model_version.version


def test_promote_to_production_archives_previous(tmp_path):
    """After second promotion, exactly one Production and one Archived version exist."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    # Register and promote first model
    run_id_ref_1 = []
    run_id_1 = _log_fake_model_artifact(tmp_path, run_id_ref_1)
    v1 = register_approved_model(config, run_id_1)
    promote_to_production(config, v1.version, run_id_1)

    # Register and promote second model
    run_id_ref_2 = []
    run_id_2 = _log_fake_model_artifact(tmp_path, run_id_ref_2)
    v2 = register_approved_model(config, run_id_2)
    promote_to_production(config, v2.version, run_id_2)

    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)

    production_versions = client.get_latest_versions(model_name, stages=["Production"])
    archived_versions = client.get_latest_versions(model_name, stages=["Archived"])

    assert len(production_versions) == 1, f"Expected 1 Production, got {len(production_versions)}"
    assert production_versions[0].version == v2.version, "v2 should be Production"
    assert len(archived_versions) == 1, f"Expected 1 Archived, got {len(archived_versions)}"
    assert archived_versions[0].version == v1.version, "v1 should be Archived"


def test_promote_to_production_sets_tags(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    run_id_ref = []
    run_id = _log_fake_model_artifact(tmp_path, run_id_ref)
    model_version = register_approved_model(config, run_id)
    promote_to_production(config, model_version.version, run_id)

    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    version = client.get_model_version(model_name, model_version.version)

    assert "promotion.promoted_at" in version.tags
    assert version.tags["promotion.promoted_by"] == "pipeline"
    assert version.tags["promotion.decision_run_id"] == run_id


def test_promote_to_production_custom_promoted_by(tmp_path):
    """promoted_by parameter should appear in the tag rather than the hardcoded 'pipeline'."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    run_id_ref = []
    run_id = _log_fake_model_artifact(tmp_path, run_id_ref)
    model_version = register_approved_model(config, run_id)
    promote_to_production(config, model_version.version, run_id, promoted_by="ci-runner")

    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    version = client.get_model_version(model_name, model_version.version)
    assert version.tags["promotion.promoted_by"] == "ci-runner"


def test_attach_lineage_tags_all_required_keys(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    run_id_ref = []
    run_id = _log_fake_model_artifact(tmp_path, run_id_ref)
    model_version = register_approved_model(config, run_id)

    client = get_mlflow_client(config)
    run = client.get_run(run_id)
    report = {"comparison": {"overall_verdict": "no_baseline"}}
    decision = {"outcome": "approved"}

    attach_lineage_tags(config, model_version.version, run, report, decision)

    model_name = resolve_model_name(config)
    version = client.get_model_version(model_name, model_version.version)
    tags = version.tags

    required = [
        "lineage.run_id",
        "lineage.config_hash",
        "lineage.dataset_version_id",
        "lineage.task_type",
        "lineage.algorithm",
        "lineage.trained_at",
        "lineage.promotion_outcome",
    ]
    for key in required:
        assert key in tags, f"Required lineage tag missing: {key}"


def test_get_production_model_metrics_no_production(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    result = get_production_model_metrics(config)
    assert result is None


def test_get_production_model_metrics_returns_metrics(tmp_path):
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    run_id_ref = []
    run_id = _log_fake_model_artifact(tmp_path, run_id_ref)
    model_version = register_approved_model(config, run_id)
    promote_to_production(config, model_version.version, run_id)

    metrics = get_production_model_metrics(config)

    assert metrics is not None
    assert "accuracy" in metrics
    assert "f1_score" in metrics
    assert abs(metrics["accuracy"] - 0.95) < 1e-6
