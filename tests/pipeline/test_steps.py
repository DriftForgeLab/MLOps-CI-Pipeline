import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.steps import (
    execute_stage,
    StageResult,
    _STAGE_REGISTRY,
    _evaluation_stage,
    _model_analysis_stage,
    _promotion_stage,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dummy_config():
    """Return a MagicMock standing in for PipelineConfig (never inspected by execute_stage)."""
    return MagicMock()


VERSION_ID = "abc123"


# ── Successful execution ─────────────────────────────────────────────────────

class TestExecuteStageSuccess:
    def test_completed_status(self):
        fake_fn = MagicMock()
        with patch.dict(_STAGE_REGISTRY, {"test_stage": fake_fn}):
            result = execute_stage("test_stage", _dummy_config(), VERSION_ID)
        assert result.status == "completed"
        assert result.error is None
        assert result.stage == "test_stage"
        fake_fn.assert_called_once()

    def test_timestamps_are_iso(self):
        with patch.dict(_STAGE_REGISTRY, {"s": MagicMock()}):
            result = execute_stage("s", _dummy_config(), VERSION_ID)
        # ISO 8601 with timezone: ends with +00:00
        assert "T" in result.started_at
        assert "T" in result.ended_at

    def test_duration_is_non_negative(self):
        def slow_stage(config, vid):
            time.sleep(0.05)

        with patch.dict(_STAGE_REGISTRY, {"slow": slow_stage}):
            result = execute_stage("slow", _dummy_config(), VERSION_ID)
        assert result.duration_seconds >= 0

    def test_stage_fn_receives_config_and_version(self):
        received = {}

        def capturing_stage(config, vid):
            received["config"] = config
            received["vid"] = vid

        cfg = _dummy_config()
        with patch.dict(_STAGE_REGISTRY, {"capture": capturing_stage}):
            execute_stage("capture", cfg, VERSION_ID)
        assert received["config"] is cfg
        assert received["vid"] == VERSION_ID


# ── Failed execution ─────────────────────────────────────────────────────────

class TestExecuteStageFailed:
    def test_failed_status_on_exception(self):
        def bad_stage(config, vid):
            raise RuntimeError("boom")

        with patch.dict(_STAGE_REGISTRY, {"bad": bad_stage}):
            result = execute_stage("bad", _dummy_config(), VERSION_ID)
        assert result.status == "failed"
        assert "boom" in result.error

    def test_raises_key_error_for_unknown_stage(self):
        with pytest.raises(KeyError):
            execute_stage("nonexistent", _dummy_config(), VERSION_ID)


# ── model_analysis stage registered ─────────────────────────────────────────

class TestModelAnalysisStageRegistered:
    def test_model_analysis_in_registry(self):
        assert "model_analysis" in _STAGE_REGISTRY

    def test_registry_order_model_analysis_before_promotion(self):
        keys = list(_STAGE_REGISTRY.keys())
        assert keys.index("model_analysis") < keys.index("promotion")


# ── _model_analysis_stage ────────────────────────────────────────────────────

class TestModelAnalysisStage:
    def test_skips_when_disabled(self, tmp_path: Path):
        """model_analysis stage returns immediately when enabled=False."""
        config = _dummy_config()
        config.configs.drift = str(tmp_path / "drift.yaml")

        disabled_config = MagicMock()
        disabled_config.enabled = False

        with patch("src.pipeline.steps.load_drift_config", return_value=disabled_config):
            with patch("src.pipeline.steps._run_image_model_analysis_stage") as mock_run:
                _model_analysis_stage(config, VERSION_ID)
                mock_run.assert_not_called()

    def test_dispatches_to_image_analysis_when_enabled(self, tmp_path: Path):
        """model_analysis stage calls _run_image_model_analysis_stage when enabled."""
        config = _dummy_config()
        config.configs.drift = str(tmp_path / "drift.yaml")

        drift_config = MagicMock()
        drift_config.enabled = True

        with patch("src.pipeline.steps.load_drift_config", return_value=drift_config):
            with patch("src.pipeline.steps._run_image_model_analysis_stage") as mock_run:
                _model_analysis_stage(config, VERSION_ID)
                mock_run.assert_called_once_with(config, VERSION_ID, drift_config)


# ── _promotion_stage ─────────────────────────────────────────────────────────

class TestPromotionStage:
    """Tests that _promotion_stage behaves correctly without drift blocking."""

    def _setup_eval_report(self, output_dir: Path) -> dict:
        report = {
            "metrics": {"accuracy": 0.95, "f1_score": 0.90},
            "comparison": {"overall_verdict": "improved", "has_production_model": False},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f)
        return report

    def _mock_config(self, tmp_path: Path) -> MagicMock:
        config = _dummy_config()
        config.output_dir = str(tmp_path / "outputs")
        config.configs.promotion = str(tmp_path / "promotion.yaml")
        config.task_type = "classification"
        return config

    def test_promotion_passes_with_no_violations(self, tmp_path: Path):
        """Promotion proceeds when all rules pass."""
        output_dir = tmp_path / "outputs"
        self._setup_eval_report(output_dir)
        config = self._mock_config(tmp_path)

        approval_result = MagicMock()
        approval_result.approved = True

        with (
            patch("src.pipeline.steps.load_promotion_config", return_value=MagicMock()),
            patch("src.pipeline.steps.run_promotion_rules", return_value=[]),
            patch("src.pipeline.steps.request_approval", return_value=approval_result) as mock_approval,
            patch("src.pipeline.steps.mlflow") as mock_mlflow,
            patch("src.pipeline.steps.log_promotion_decision_to_mlflow"),
        ):
            mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="run1"))
            with patch("src.registry.model_registry.register_approved_model") as mock_reg, \
                 patch("src.registry.model_registry.promote_to_production"), \
                 patch("src.registry.model_registry.attach_lineage_tags"), \
                 patch("src.registry.model_registry.get_mlflow_client") as mock_client:
                mock_reg.return_value = MagicMock(version="1")
                mock_client.return_value.get_run.return_value = MagicMock()
                _promotion_stage(config, VERSION_ID)

        mock_approval.assert_called_once()
        _, kwargs = mock_approval.call_args
        assert kwargs.get("drift") is None

    def test_promotion_blocked_by_metric_violation(self, tmp_path: Path):
        """A failing promotion rule raises PromotionBlockedError."""
        from src.pipeline.steps import PromotionBlockedError
        output_dir = tmp_path / "outputs"
        self._setup_eval_report(output_dir)
        config = self._mock_config(tmp_path)

        violation = {
            "rule_id": "min_accuracy",
            "metric": "accuracy",
            "observed": 0.5,
            "operator": ">=",
            "threshold": 0.8,
            "description": "Accuracy too low",
        }

        with (
            patch("src.pipeline.steps.load_promotion_config", return_value=MagicMock()),
            patch("src.pipeline.steps.run_promotion_rules", return_value=[violation]),
        ):
            with pytest.raises(PromotionBlockedError, match="min_accuracy"):
                _promotion_stage(config, VERSION_ID)

    def test_no_drift_result_json_written(self, tmp_path: Path):
        """Promotion stage does not create drift_result.json."""
        output_dir = tmp_path / "outputs"
        self._setup_eval_report(output_dir)
        config = self._mock_config(tmp_path)

        approval_result = MagicMock()
        approval_result.approved = True

        with (
            patch("src.pipeline.steps.load_promotion_config", return_value=MagicMock()),
            patch("src.pipeline.steps.run_promotion_rules", return_value=[]),
            patch("src.pipeline.steps.request_approval", return_value=approval_result),
            patch("src.pipeline.steps.mlflow") as mock_mlflow,
            patch("src.pipeline.steps.log_promotion_decision_to_mlflow"),
        ):
            mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="run1"))
            with patch("src.registry.model_registry.register_approved_model") as mock_reg, \
                 patch("src.registry.model_registry.promote_to_production"), \
                 patch("src.registry.model_registry.attach_lineage_tags"), \
                 patch("src.registry.model_registry.get_mlflow_client") as mock_client:
                mock_reg.return_value = MagicMock(version="1")
                mock_client.return_value.get_run.return_value = MagicMock()
                _promotion_stage(config, VERSION_ID)

        assert not (output_dir / "drift_result.json").exists()


class TestEvaluationStage:
    def test_writes_official_test_report_when_present(self, tmp_path: Path):
        config = _dummy_config()
        config.output_dir = str(tmp_path / "outputs")

        report = {
            "metrics": {"accuracy": 0.8, "f1_score": 0.8},
            "comparison": {"overall_verdict": "no_baseline", "has_production_model": False},
            "official_test_report": {
                "split": "test",
                "metrics": {"accuracy": 0.81, "f1_score": 0.8},
            },
        }

        with (
            patch("src.pipeline.steps.evaluate", return_value=report),
            patch("src.pipeline.steps.log_evaluation_to_mlflow"),
            patch("src.pipeline.steps.log_comparison_to_mlflow"),
        ):
            _evaluation_stage(config, VERSION_ID)

        output_dir = Path(config.output_dir)
        assert (output_dir / "evaluation_report.json").exists()
        assert (output_dir / "official_test_evaluation_report.json").exists()
