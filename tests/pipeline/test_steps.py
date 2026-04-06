import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.steps import (
    execute_stage,
    StageResult,
    _STAGE_REGISTRY,
    _drift_stage,
    _promotion_stage,
    _severity_ge,
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
        assert result.duration_seconds >= 0.04


# ── Stage failure ─────────────────────────────────────────────────────────────

class TestExecuteStageFailed:
    def test_exception_sets_failed_status(self):
        def exploding_stage(config, vid):
            raise RuntimeError("boom")

        with patch.dict(_STAGE_REGISTRY, {"bad": exploding_stage}):
            result = execute_stage("bad", _dummy_config(), VERSION_ID)
        assert result.status == "failed"
        assert result.error == "boom"
        assert result.stage == "bad"

    def test_returns_stage_result_on_failure(self):
        """execute_stage catches exceptions and returns a StageResult (does not re-raise)."""
        def failing(config, vid):
            raise ValueError("oops")

        with patch.dict(_STAGE_REGISTRY, {"fail": failing}):
            result = execute_stage("fail", _dummy_config(), VERSION_ID)
        assert isinstance(result, StageResult)
        assert result.duration_seconds >= 0


# ── Unknown stage ────────────────────────────────────────────────────────────

class TestUnknownStage:
    def test_unknown_stage_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown pipeline stage"):
            execute_stage("nonexistent", _dummy_config(), VERSION_ID)


# ── Drift stage registered ──────────────────────────────────────────────────

class TestDriftStageRegistered:
    def test_drift_in_registry(self):
        assert "drift" in _STAGE_REGISTRY

    def test_registry_order_drift_before_promotion(self):
        keys = list(_STAGE_REGISTRY.keys())
        assert keys.index("drift") < keys.index("promotion")


# ── _severity_ge helper ─────────────────────────────────────────────────────

class TestSeverityGe:
    @pytest.mark.parametrize("severity,threshold,expected", [
        ("low", "low", True),
        ("medium", "low", True),
        ("high", "low", True),
        ("low", "medium", False),
        ("medium", "medium", True),
        ("high", "medium", True),
        ("low", "high", False),
        ("medium", "high", False),
        ("high", "high", True),
    ])
    def test_severity_comparison(self, severity, threshold, expected):
        assert _severity_ge(severity, threshold) is expected


# ── _drift_stage ─────────────────────────────────────────────────────────────

class TestDriftStage:
    def test_skips_when_disabled(self, tmp_path: Path):
        """Drift stage should return immediately when enabled=False."""
        config = _dummy_config()
        config.configs.drift = str(tmp_path / "drift.yaml")

        disabled_config = MagicMock()
        disabled_config.enabled = False

        with patch("src.pipeline.steps.load_drift_config", return_value=disabled_config):
            # Should not raise, should not call run_drift_analysis
            with patch("src.pipeline.steps.run_drift_analysis") as mock_run:
                _drift_stage(config, VERSION_ID)
                mock_run.assert_not_called()

    def test_writes_drift_result_json(self, tmp_path: Path):
        """Drift stage should write drift_result.json to output_dir."""
        config = _dummy_config()
        config.configs.drift = str(tmp_path / "drift.yaml")
        config.data.drift_scenarios = str(tmp_path / "drift_scenarios")
        config.output_dir = str(tmp_path / "outputs")

        (tmp_path / "outputs").mkdir(parents=True, exist_ok=True)

        drift_config = MagicMock()
        drift_config.enabled = True

        drift_result = {"overall": {"severity": "low"}, "features": {}}
        evidently_report = MagicMock()

        with (
            patch("src.pipeline.steps.load_drift_config", return_value=drift_config),
            patch("src.pipeline.steps.run_drift_analysis", return_value=(drift_result, evidently_report)),
            patch("src.pipeline.steps.save_drift_report_json") as mock_json,
            patch("src.pipeline.steps.save_drift_report_html") as mock_html,
            patch("src.pipeline.steps.log_drift_metrics_to_mlflow") as mock_mlflow,
        ):
            _drift_stage(config, VERSION_ID)

        mock_json.assert_called_once()
        mock_html.assert_called_once()
        mock_mlflow.assert_called_once_with(drift_result)

        # Verify atomic_write_json was called for drift_result.json
        drift_path = tmp_path / "outputs" / "drift_result.json"
        assert drift_path.exists()
        with open(drift_path) as f:
            saved = json.load(f)
        assert saved == drift_result


# ── _promotion_stage with drift ──────────────────────────────────────────────

class TestPromotionStageWithDrift:
    """Tests that _promotion_stage correctly reads drift_result.json."""

    def _setup_eval_report(self, output_dir: Path) -> dict:
        """Write a minimal evaluation report and return it."""
        report = {
            "metrics": {"accuracy": 0.95, "f1_score": 0.90},
            "comparison": {"overall_verdict": "improved", "has_production_model": False},
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f)
        return report

    def _setup_drift_result(self, output_dir: Path, severity: str) -> dict:
        """Write a drift_result.json with given overall severity."""
        drift = {
            "overall": {
                "severity": severity,
                "dataset_drift_detected": severity != "low",
                "drifted_feature_count": 2 if severity != "low" else 0,
                "total_feature_count": 4,
                "drift_share": 0.5 if severity != "low" else 0.0,
            },
            "features": {},
        }
        with open(output_dir / "drift_result.json", "w") as f:
            json.dump(drift, f)
        return drift

    def _mock_config(self, tmp_path: Path) -> MagicMock:
        config = _dummy_config()
        config.output_dir = str(tmp_path / "outputs")
        config.configs.promotion = str(tmp_path / "promotion.yaml")
        config.task_type = "classification"
        return config

    def test_high_severity_blocks_promotion(self, tmp_path: Path):
        """High severity drift at default threshold should raise ValueError."""
        output_dir = tmp_path / "outputs"
        self._setup_eval_report(output_dir)
        self._setup_drift_result(output_dir, severity="high")
        config = self._mock_config(tmp_path)

        promo_config = MagicMock()
        promo_config.drift_block_severity = "high"

        with (
            patch("src.pipeline.steps.load_promotion_config", return_value=promo_config),
            patch("src.pipeline.steps.run_promotion_rules", return_value=[]),
        ):
            with pytest.raises(ValueError, match="DRIFT_SEVERITY"):
                _promotion_stage(config, VERSION_ID)

    def test_low_severity_passes_drift_to_approval(self, tmp_path: Path):
        """Low severity drift should pass drift dict to request_approval."""
        output_dir = tmp_path / "outputs"
        self._setup_eval_report(output_dir)
        drift = self._setup_drift_result(output_dir, severity="low")
        config = self._mock_config(tmp_path)

        promo_config = MagicMock()
        promo_config.drift_block_severity = "high"

        approval_result = MagicMock()
        approval_result.approved = True

        with (
            patch("src.pipeline.steps.load_promotion_config", return_value=promo_config),
            patch("src.pipeline.steps.run_promotion_rules", return_value=[]),
            patch("src.pipeline.steps.request_approval", return_value=approval_result) as mock_approval,
            patch("src.pipeline.steps.mlflow") as mock_mlflow,
            patch("src.pipeline.steps.log_promotion_decision_to_mlflow"),
            patch("src.pipeline.steps.register_approved_model", create=True),
            patch("src.pipeline.steps.promote_to_production", create=True),
            patch("src.pipeline.steps.attach_lineage_tags", create=True),
            patch("src.pipeline.steps.get_mlflow_client", create=True),
        ):
            mock_mlflow.active_run.return_value = MagicMock(info=MagicMock(run_id="run1"))

            # Patch the deferred imports inside _promotion_stage
            with patch("src.registry.model_registry.register_approved_model") as mock_reg, \
                 patch("src.registry.model_registry.promote_to_production"), \
                 patch("src.registry.model_registry.attach_lineage_tags"), \
                 patch("src.registry.model_registry.get_mlflow_client") as mock_client:
                mock_reg.return_value = MagicMock(version="1")
                mock_client.return_value.get_run.return_value = MagicMock()
                _promotion_stage(config, VERSION_ID)

            # Verify drift was passed to request_approval
            mock_approval.assert_called_once()
            _, kwargs = mock_approval.call_args
            assert kwargs.get("drift") == drift

    def test_no_drift_file_behaves_as_before(self, tmp_path: Path):
        """When drift_result.json is absent, promotion runs normally."""
        output_dir = tmp_path / "outputs"
        self._setup_eval_report(output_dir)
        # Do NOT create drift_result.json
        config = self._mock_config(tmp_path)

        promo_config = MagicMock()
        promo_config.drift_block_severity = "high"

        approval_result = MagicMock()
        approval_result.approved = True

        with (
            patch("src.pipeline.steps.load_promotion_config", return_value=promo_config),
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

            # drift=None passed to request_approval
            _, kwargs = mock_approval.call_args
            assert kwargs.get("drift") is None

    def test_medium_severity_below_high_threshold_passes(self, tmp_path: Path):
        """Medium severity with drift_block_severity='high' should not block."""
        output_dir = tmp_path / "outputs"
        self._setup_eval_report(output_dir)
        self._setup_drift_result(output_dir, severity="medium")
        config = self._mock_config(tmp_path)

        promo_config = MagicMock()
        promo_config.drift_block_severity = "high"

        approval_result = MagicMock()
        approval_result.approved = True

        with (
            patch("src.pipeline.steps.load_promotion_config", return_value=promo_config),
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

            # Should pass — medium < high threshold
            mock_approval.assert_called_once()
