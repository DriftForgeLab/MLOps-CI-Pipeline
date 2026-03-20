import json

import pytest

from src.pipeline.report import build_run_report, compute_config_hash, write_run_report
from src.pipeline.steps import StageResult


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_stage(stage, status="completed", error=None):
    return StageResult(
        stage=stage,
        status=status,
        started_at="2026-03-20T10:00:00+00:00",
        ended_at="2026-03-20T10:01:00+00:00",
        duration_seconds=60.0,
        error=error,
    )


# ── compute_config_hash ────────────────────────────────────────────────────

class TestComputeConfigHash:
    def test_deterministic_same_content(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("key: value\n", encoding="utf-8")
        hash1 = compute_config_hash(config)
        hash2 = compute_config_hash(config)
        assert hash1 == hash2

    def test_different_content_different_hash(self, tmp_path):
        c1 = tmp_path / "config1.yaml"
        c2 = tmp_path / "config2.yaml"
        c1.write_text("key: value1\n", encoding="utf-8")
        c2.write_text("key: value2\n", encoding="utf-8")
        assert compute_config_hash(c1) != compute_config_hash(c2)

    def test_hash_format(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("test\n", encoding="utf-8")
        h = compute_config_hash(config)
        assert h.startswith("sha256:")
        assert len(h) == len("sha256:") + 64


# ── build_run_report ───────────────────────────────────────────────────────

class TestBuildRunReport:
    def test_all_keys_present(self):
        report = build_run_report(
            project_name="test-project",
            project_version="1.0",
            config_hash="sha256:abc",
            task_type="classification",
            random_seed=42,
            dataset_version_id="v1",
            stage_results=[_make_stage("training")],
        )
        expected_keys = {
            "project_id", "config_hash", "timestamp", "task_type",
            "random_seed", "dataset_version_id", "artifact_path",
            "executed_stages", "overall_status", "pipeline_execution_id",
            "mlflow_run_id",
        }
        assert set(report.keys()) == expected_keys

    def test_project_id_format(self):
        report = build_run_report(
            project_name="myproject",
            project_version="2.1",
            config_hash="sha256:abc",
            task_type="regression",
            random_seed=42,
            dataset_version_id="v1",
            stage_results=[],
        )
        assert report["project_id"] == "myproject-v2.1"

    def test_all_completed_status(self):
        stages = [_make_stage("training"), _make_stage("evaluation")]
        report = build_run_report(
            project_name="p", project_version="1",
            config_hash="h", task_type="t", random_seed=1,
            dataset_version_id="v", stage_results=stages,
        )
        assert report["overall_status"] == "completed"

    def test_any_failed_status(self):
        stages = [
            _make_stage("training"),
            _make_stage("evaluation", status="failed", error="boom"),
        ]
        report = build_run_report(
            project_name="p", project_version="1",
            config_hash="h", task_type="t", random_seed=1,
            dataset_version_id="v", stage_results=stages,
        )
        assert report["overall_status"] == "failed"

    def test_empty_stages_is_completed(self):
        report = build_run_report(
            project_name="p", project_version="1",
            config_hash="h", task_type="t", random_seed=1,
            dataset_version_id="v", stage_results=[],
        )
        assert report["overall_status"] == "completed"

    def test_executed_stages_are_serialized(self):
        stages = [_make_stage("training")]
        report = build_run_report(
            project_name="p", project_version="1",
            config_hash="h", task_type="t", random_seed=1,
            dataset_version_id="v", stage_results=stages,
        )
        assert len(report["executed_stages"]) == 1
        assert report["executed_stages"][0]["stage"] == "training"
        assert isinstance(report["executed_stages"][0], dict)

    def test_optional_fields_default_to_none(self):
        report = build_run_report(
            project_name="p", project_version="1",
            config_hash="h", task_type="t", random_seed=1,
            dataset_version_id="v", stage_results=[],
        )
        assert report["artifact_path"] is None
        assert report["pipeline_execution_id"] is None
        assert report["mlflow_run_id"] is None

    def test_optional_fields_passed_through(self):
        report = build_run_report(
            project_name="p", project_version="1",
            config_hash="h", task_type="t", random_seed=1,
            dataset_version_id="v", stage_results=[],
            artifact_path="/tmp/artifacts",
            pipeline_execution_id="exec-123",
            mlflow_run_id="run-456",
        )
        assert report["artifact_path"] == "/tmp/artifacts"
        assert report["pipeline_execution_id"] == "exec-123"
        assert report["mlflow_run_id"] == "run-456"


# ── write_run_report ───────────────────────────────────────────────────────

class TestWriteRunReport:
    def test_creates_file(self, tmp_path):
        report = {"test": True}
        path = write_run_report(report, str(tmp_path / "output"))
        assert path.exists()
        assert path.name == "run_report.json"

    def test_file_contains_valid_json(self, tmp_path):
        report = {"key": "value", "num": 42}
        path = write_run_report(report, str(tmp_path / "output"))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == report

    def test_creates_nested_dirs(self, tmp_path):
        report = {"ok": True}
        path = write_run_report(report, str(tmp_path / "a" / "b" / "c"))
        assert path.exists()
