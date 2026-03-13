"""Tests for src/registry/rollback.py — Rollback governance CLI."""

import json
import pytest
import mlflow
import mlflow.tracking.fluent
import joblib
from pathlib import Path

from src.config.loader import load_config
from src.registry.model_registry import (
    register_approved_model,
    promote_to_production,
    get_mlflow_client,
    resolve_model_name,
)
from src.registry.rollback import (
    _execute_rollback,
    _find_current_production,
    _get_version_tag,
    _write_rollback_intent,
    _complete_rollback_audit,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, task_type: str = "classification", registry_model_name: str | None = None) -> Path:
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


def _setup_two_versions(tmp_path: Path, config):
    """Register and promote two versions; v1 gets archived, v2 is Production."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    # Reset the active experiment so start_run() uses the Default experiment in this store.
    mlflow.tracking.fluent._active_experiment_id = None

    def _make_run():
        model_dir = tmp_path / f"model_{len(list(tmp_path.iterdir()))}"
        model_dir.mkdir(parents=True, exist_ok=True)
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(n_estimators=2, random_state=42).fit(X, y)
        joblib.dump(model, model_dir / "model.joblib")
        (model_dir / "metadata.json").write_text('{}')
        with mlflow.start_run() as run:
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            mlflow.log_params({
                "algorithm": "random_forest",
                "train_rows": "100",
                "trained_at": "2026-03-10T00:00:00+00:00",
            })
        return run.info.run_id

    run_id_1 = _make_run()
    v1 = register_approved_model(config, run_id_1)
    promote_to_production(config, v1.version, run_id_1)

    run_id_2 = _make_run()
    v2 = register_approved_model(config, run_id_2)
    promote_to_production(config, v2.version, run_id_2)

    return v1, v2


# ---------------------------------------------------------------------------
# Unit tests — _find_current_production
# ---------------------------------------------------------------------------

def test_find_current_production_with_production(tmp_path):
    """_find_current_production returns the Production-stage version."""
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    v1, v2 = _setup_two_versions(tmp_path, config)

    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    all_versions = client.search_model_versions(f"name='{model_name}'")

    result = _find_current_production(all_versions)

    assert result is not None
    assert int(result.version) == int(v2.version)
    assert result.current_stage == "Production"


def test_find_current_production_no_production():
    """_find_current_production returns None when no version is in Production."""
    class FakeVersion:
        current_stage = None

    result = _find_current_production([FakeVersion()])
    assert result is None


# ---------------------------------------------------------------------------
# Unit tests — _get_version_tag
# ---------------------------------------------------------------------------

def test_get_version_tag_present():
    """_get_version_tag returns the tag value when the key exists."""
    class FakeVersion:
        tags = {"lineage.algorithm": "random_forest"}

    result = _get_version_tag(FakeVersion(), "lineage.algorithm")
    assert result == "random_forest"


def test_get_version_tag_missing():
    """_get_version_tag returns the fallback when the key is absent."""
    class FakeVersion:
        tags = {}

    result = _get_version_tag(FakeVersion(), "nonexistent.key", "fallback")
    assert result == "fallback"


# ---------------------------------------------------------------------------
# Integration tests — _execute_rollback
# ---------------------------------------------------------------------------

def test_execute_rollback_success(tmp_path, monkeypatch):
    """Rolling back from v2 to v1 promotes v1 to Production and archives v2."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    v1, v2 = _setup_two_versions(tmp_path, config)

    _execute_rollback(config, int(v1.version), "test reason", yes=True)

    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    all_versions = client.search_model_versions(f"name='{model_name}'")

    stages = {int(v.version): v.current_stage for v in all_versions}
    assert stages[int(v1.version)] == "Production"
    assert stages[int(v2.version)] == "Archived"


def test_execute_rollback_version_not_found(tmp_path, monkeypatch):
    """Attempting to roll back to a non-existent version exits with code 1."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    _setup_two_versions(tmp_path, config)

    with pytest.raises(SystemExit) as exc_info:
        _execute_rollback(config, 999, "test", yes=True)

    assert exc_info.value.code == 1


def test_execute_rollback_already_production(tmp_path, monkeypatch):
    """Attempting to roll back to the current Production version exits with code 1."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    v1, v2 = _setup_two_versions(tmp_path, config)

    with pytest.raises(SystemExit) as exc_info:
        _execute_rollback(config, int(v2.version), "test", yes=True)

    assert exc_info.value.code == 1


def test_execute_rollback_writes_decision_json(tmp_path, monkeypatch):
    """_execute_rollback writes a rollback_decision_*.json file with correct content."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    v1, v2 = _setup_two_versions(tmp_path, config)

    _execute_rollback(config, int(v1.version), "test rollback reason", yes=True)

    outputs_dir = tmp_path / "outputs"
    json_files = list(outputs_dir.glob("rollback_decision_*.json"))
    assert len(json_files) >= 1, "Expected at least one rollback_decision_*.json file"

    data = json.loads(json_files[0].read_text())
    assert data["event_type"] == "rollback"
    assert data["to_version"] == int(v1.version)
    assert data["reason"] == "test rollback reason"


def test_execute_rollback_result_in_registry(tmp_path, monkeypatch):
    """After rollback, exactly one Production version exists in the registry and it is v1."""
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    v1, v2 = _setup_two_versions(tmp_path, config)

    _execute_rollback(config, int(v1.version), "test reason", yes=True)

    client = get_mlflow_client(config)
    model_name = resolve_model_name(config)
    production_versions = client.get_latest_versions(model_name, stages=["Production"])

    assert len(production_versions) == 1
    assert int(production_versions[0].version) == int(v1.version)


# ---------------------------------------------------------------------------
# Integration tests — _complete_rollback_audit
# ---------------------------------------------------------------------------

def test_complete_rollback_audit_sets_governance_tags(tmp_path, monkeypatch):
    """_complete_rollback_audit sets governance/promotion tags and finalises the decision JSON."""
    from datetime import datetime, timezone as tz
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    mlflow.tracking.fluent._active_experiment_id = None
    cfg_path = _write_config(tmp_path)
    config = load_config(cfg_path)

    def _make_run():
        model_dir = tmp_path / "model_gov"
        model_dir.mkdir(parents=True, exist_ok=True)
        from sklearn.datasets import load_iris
        from sklearn.ensemble import RandomForestClassifier
        X, y = load_iris(return_X_y=True)
        model = RandomForestClassifier(n_estimators=2, random_state=42).fit(X, y)
        joblib.dump(model, model_dir / "model.joblib")
        (model_dir / "metadata.json").write_text('{}')
        with mlflow.start_run() as run:
            mlflow.log_artifacts(str(model_dir), artifact_path="model")
            mlflow.log_params({
                "algorithm": "random_forest",
                "train_rows": "100",
                "trained_at": "2026-03-10T00:00:00+00:00",
            })
        return run.info.run_id

    run_id = _make_run()
    v1 = register_approved_model(config, run_id)
    promote_to_production(config, v1.version, run_id)

    model_name = resolve_model_name(config)
    client = get_mlflow_client(config)
    target_version = client.get_model_version(model_name, v1.version)

    event_ts = datetime(2026, 3, 10, 12, 0, 0, tzinfo=tz.utc)
    timestamp_str = event_ts.strftime("%Y%m%dT%H%M%SZ")
    actor = "cli:testuser"

    decision_path = _write_rollback_intent(
        config=config,
        model_name=model_name,
        target_version=target_version,
        previous_production=None,
        reason="audit reason",
        actor=actor,
        event_ts=event_ts,
        timestamp_str=timestamp_str,
    )

    _complete_rollback_audit(
        config=config,
        model_name=model_name,
        target_version=target_version,
        previous_production=None,
        reason="audit reason",
        actor=actor,
        event_ts=event_ts,
        timestamp_str=timestamp_str,
        decision_path=decision_path,
    )

    refreshed = client.get_model_version(model_name, v1.version)
    assert refreshed.tags.get("governance.last_rollback_at") is not None
    assert refreshed.tags.get("governance.rollback_reason") == "audit reason"
    assert refreshed.tags.get("governance.rollback_executed_by") == actor
    assert refreshed.tags.get("promotion.promoted_by") == "rollback-cli"
    assert refreshed.tags.get("promotion.decision_run_id", "").startswith("rollback:")

    # Intent JSON should be updated to completed
    data = json.loads(decision_path.read_text())
    assert data["status"] == "completed"
    assert data["executed_by"] == actor
