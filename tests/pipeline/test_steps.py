import time
from unittest.mock import MagicMock, patch

import pytest

from src.pipeline.steps import execute_stage, StageResult, _STAGE_REGISTRY


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
