from datetime import datetime, timedelta, timezone

import pytest

from src.promotion.approval import (
    ApprovalResult,
    request_approval,
    _print_summary,
    _print_drift_block,
)


# ── helpers ─────────────────────────────────────────────────────────────────

def _minimal_report(metrics=None, comparison=None):
    return {
        "metrics": metrics or {"accuracy": 0.95, "f1_score": 0.88},
        "comparison": comparison or {
            "has_production_model": False,
            "overall_verdict": "no_baseline",
        },
    }


def _drift_dict(
    dataset_drift_detected=True,
    severity="medium",
    drifted_feature_count=2,
    total_feature_count=4,
    drift_share=0.5,
    features=None,
):
    """Build a minimal drift result dict matching the standard schema."""
    if features is None:
        features = {
            "sepal_length": {
                "column_type": "num",
                "drift_detected": True,
                "drift_score": 0.03,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "medium",
            },
            "sepal_width": {
                "column_type": "num",
                "drift_detected": False,
                "drift_score": 0.42,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "low",
            },
            "petal_length": {
                "column_type": "num",
                "drift_detected": True,
                "drift_score": 0.01,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "medium",
            },
            "petal_width": {
                "column_type": "num",
                "drift_detected": False,
                "drift_score": 0.78,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "low",
            },
        }
    return {
        "overall": {
            "dataset_drift_detected": dataset_drift_detected,
            "drift_share": drift_share,
            "drifted_feature_count": drifted_feature_count,
            "total_feature_count": total_feature_count,
            "severity": severity,
        },
        "features": features,
    }


# ── request_approval: approve path ─────────────────────────────────────────

class TestApprovalApprove:
    def test_approve_with_1(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: "1")
        result = request_approval(_minimal_report())
        assert result.approved is True
        assert result.reason is None

    def test_approve_with_whitespace(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: "  1  ")
        result = request_approval(_minimal_report())
        assert result.approved is True


# ── request_approval: reject path ──────────────────────────────────────────

class TestApprovalReject:
    def test_reject_with_reason(self, monkeypatch):
        inputs = iter(["2", "low accuracy"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        result = request_approval(_minimal_report())
        assert result.approved is False
        assert result.reason == "low accuracy"

    def test_reject_with_empty_reason_is_cancellation(self, monkeypatch):
        inputs = iter(["2", ""])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        result = request_approval(_minimal_report())
        assert result.approved is False
        assert result.reason is None

    def test_reject_with_whitespace_only_reason_is_cancellation(self, monkeypatch):
        inputs = iter(["2", "   "])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        result = request_approval(_minimal_report())
        assert result.approved is False
        assert result.reason is None


# ── request_approval: cancel path ──────────────────────────────────────────

class TestApprovalCancel:
    def test_cancel_with_q(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: "q")
        result = request_approval(_minimal_report())
        assert result.approved is False
        assert result.reason is None

    def test_cancel_with_uppercase_Q(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: "Q")
        result = request_approval(_minimal_report())
        assert result.approved is False

    def test_cancel_with_empty_input(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: "")
        result = request_approval(_minimal_report())
        assert result.approved is False
        assert result.reason is None

    def test_cancel_with_arbitrary_input(self, monkeypatch):
        monkeypatch.setattr("builtins.input", lambda prompt: "xyz")
        result = request_approval(_minimal_report())
        assert result.approved is False


# ── _print_summary: output content ─────────────────────────────────────────

class TestPrintSummary:
    def test_prints_metrics(self, capsys):
        _print_summary(_minimal_report(metrics={"accuracy": 0.95}))
        output = capsys.readouterr().out
        assert "accuracy" in output
        assert "0.9500" in output

    def test_prints_comparison_verdict(self, capsys):
        _print_summary(_minimal_report())
        output = capsys.readouterr().out
        assert "NO_BASELINE" in output

    def test_prints_per_metric_deltas_when_production_exists(self, capsys):
        report = _minimal_report(
            comparison={
                "has_production_model": True,
                "overall_verdict": "better",
                "per_metric": {
                    "accuracy": {"delta": 0.05, "verdict": "better"},
                },
            }
        )
        _print_summary(report)
        output = capsys.readouterr().out
        assert "+0.0500" in output
        assert "better" in output

    def test_handles_empty_metrics(self, capsys):
        _print_summary(_minimal_report(metrics={}))
        output = capsys.readouterr().out
        assert "PROMOTION SUMMARY" in output

    def test_handles_no_per_metric(self, capsys):
        report = _minimal_report(
            comparison={
                "has_production_model": True,
                "overall_verdict": "equal",
                "per_metric": None,
            }
        )
        _print_summary(report)
        output = capsys.readouterr().out
        assert "EQUAL" in output


# ── _print_summary: drift block ──────────────────────────────────────────────

class TestPrintSummaryDrift:
    """Verify drift block is no longer shown in the promotion summary.

    Drift status was removed from the promotion summary because it reflects
    pre-training state and is misleading after fine-tuning. The internal
    _print_drift_block helper still exists for standalone use.
    """

    def test_drift_not_shown_in_summary(self, capsys):
        """Promotion summary must not contain a DRIFT STATUS block."""
        _print_summary(_minimal_report(), drift=_drift_dict())
        output = capsys.readouterr().out
        assert "DRIFT STATUS" not in output

    def test_drift_none_not_shown_in_summary(self, capsys):
        """drift=None must also produce no DRIFT STATUS block in the summary."""
        _print_summary(_minimal_report(), drift=None)
        output = capsys.readouterr().out
        assert "DRIFT STATUS" not in output


# ── request_approval: drift kwarg no longer shown in summary ─────────────────

class TestRequestApprovalDrift:
    def test_drift_kwarg_not_shown_in_summary(self, monkeypatch, capsys):
        """Passing drift= to request_approval must not render a DRIFT STATUS block."""
        monkeypatch.setattr("builtins.input", lambda prompt: "1")
        drift = _drift_dict(severity="low")
        result = request_approval(_minimal_report(), drift=drift)
        output = capsys.readouterr().out
        assert "DRIFT STATUS" not in output
        assert result.approved is True

    def test_no_drift_kwarg_no_drift_block(self, monkeypatch, capsys):
        """Default drift=None produces no drift block."""
        monkeypatch.setattr("builtins.input", lambda prompt: "1")
        result = request_approval(_minimal_report())
        output = capsys.readouterr().out
        assert "DRIFT STATUS" not in output
        assert result.approved is True


# ── _print_drift_block: age / staleness rendering ────────────────────────────

class TestDriftBlockAge:
    def test_drift_block_fresh_report_shows_hours(self, capsys):
        drift = _drift_dict(severity="low")
        drift["generated_at"] = datetime.now(timezone.utc).isoformat()
        _print_drift_block(drift)
        output = capsys.readouterr().out
        assert "Report age:" in output
        assert "hours" in output
        assert "STALE" not in output

    def test_drift_block_stale_report_flags_stale(self, capsys):
        drift = _drift_dict(severity="medium")
        drift["generated_at"] = (
            datetime.now(timezone.utc) - timedelta(days=14)
        ).isoformat()
        _print_drift_block(drift)
        output = capsys.readouterr().out
        assert "14 days" in output
        assert "STALE" in output

    def test_drift_block_omits_age_when_generated_at_missing(self, capsys):
        drift = _drift_dict(severity="low")
        drift.pop("generated_at", None)
        _print_drift_block(drift)
        output = capsys.readouterr().out
        assert "Report age:" not in output

    def test_drift_block_none_shows_no_data_message(self, capsys):
        _print_drift_block(None)
        output = capsys.readouterr().out
        assert "DRIFT STATUS" in output
        assert "No drift data available" in output
