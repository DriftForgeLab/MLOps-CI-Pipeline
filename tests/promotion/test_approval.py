from datetime import datetime, timedelta, timezone

import pytest

from src.promotion.approval import (
    ApprovalResult,
    ApprovalUnavailableError,
    request_approval,
    _print_summary,
    _print_drift_block,
    _print_drift_eval_block,
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

    def test_keyboard_interrupt_at_decision_is_cancellation(self, monkeypatch):
        def _raise(_prompt):
            raise KeyboardInterrupt
        monkeypatch.setattr("builtins.input", _raise)
        result = request_approval(_minimal_report())
        assert result.approved is False
        assert result.reason is None


# ── request_approval: unavailable path (no TTY / EOF) ──────────────────────

class TestApprovalUnavailable:
    """EOFError on the decision prompt must raise ApprovalUnavailableError.

    This separates "couldn't ask the human" (environment constraint) from
    "human chose to cancel" (Q / Ctrl+C). The two had been conflated and
    both rendered as a generic ValueError caught as a 'failed' stage.
    """

    def test_eof_at_decision_raises_unavailable(self, monkeypatch):
        def _raise(_prompt):
            raise EOFError
        monkeypatch.setattr("builtins.input", _raise)
        with pytest.raises(ApprovalUnavailableError):
            request_approval(_minimal_report())

    def test_unavailable_message_mentions_stdin(self, monkeypatch):
        def _raise(_prompt):
            raise EOFError
        monkeypatch.setattr("builtins.input", _raise)
        with pytest.raises(ApprovalUnavailableError, match="stdin"):
            request_approval(_minimal_report())


# ── _print_summary: output content ─────────────────────────────────────────

class TestPrintSummary:
    def test_prints_metrics(self, capsys):
        _print_summary(_minimal_report(metrics={"accuracy": 0.95}))
        output = capsys.readouterr().out
        assert "accuracy" in output
        assert "0.9500" in output

    def test_prints_no_baseline_message(self, capsys):
        _print_summary(_minimal_report())
        output = capsys.readouterr().out
        assert "No production model yet" in output
        assert "first baseline" in output

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
    """Verify the monitoring drift block renders in the promotion summary.

    The monitoring drift block reflects pre-training detection state. It is
    shown as a *fallback* so the reviewer always sees a drift signal, but is
    suppressed when a drift-adaptation eval block is present — after
    fine-tuning the raw "high drift" label is misleading, and the adaptation
    eval block communicates the post-fine-tuning state instead.
    """

    def test_drift_shown_in_summary(self, capsys):
        """Promotion summary shows the DRIFT STATUS block when drift is given."""
        _print_summary(_minimal_report(), drift=_drift_dict())
        output = capsys.readouterr().out
        assert "DRIFT STATUS" in output

    def test_drift_none_shows_no_data_banner(self, capsys):
        """drift=None renders the explicit 'no drift data' banner."""
        _print_summary(_minimal_report(), drift=None)
        output = capsys.readouterr().out
        assert "DRIFT STATUS" in output
        assert "No drift data available" in output

    def test_image_drift_shown_in_summary(self, capsys):
        """An image-drift result also renders the DRIFT STATUS block."""
        drift = _drift_dict()
        drift["drift_type"] = "image_embedding"
        _print_summary(_minimal_report(), drift=drift)
        output = capsys.readouterr().out
        assert "DRIFT STATUS" in output

    def test_drift_block_suppressed_when_drift_eval_present(self, capsys):
        """When a drift-adaptation eval block renders, the monitoring block is suppressed."""
        drift_eval = {
            "n_holdout_images": 4,
            "baseline": {"accuracy": 0.5, "f1_score": 0.33, "precision": 0.25, "recall": 0.5},
            "after_finetuning": {"accuracy": 1.0, "f1_score": 1.0, "precision": 1.0, "recall": 1.0},
            "delta": {"accuracy": 0.5, "f1_score": 0.67, "precision": 0.75, "recall": 0.5},
            "improved": True,
        }
        _print_summary(_minimal_report(), drift=_drift_dict(), drift_eval=drift_eval)
        output = capsys.readouterr().out
        assert "DRIFTED DATA" in output  # drift-adaptation eval block rendered
        assert "DRIFT STATUS" not in output

    def test_drift_eval_shown_when_provided(self, capsys):
        """drift_eval dict must produce a DRIFTED DATA block in the summary."""
        drift_eval = {
            "n_holdout_images": 4,
            "baseline": {"accuracy": 0.5, "f1_score": 0.33, "precision": 0.25, "recall": 0.5},
            "after_finetuning": {"accuracy": 1.0, "f1_score": 1.0, "precision": 1.0, "recall": 1.0},
            "delta": {"accuracy": 0.5, "f1_score": 0.67, "precision": 0.75, "recall": 0.5},
            "improved": True,
        }
        _print_summary(_minimal_report(), drift_eval=drift_eval)
        output = capsys.readouterr().out
        assert "DRIFTED DATA" in output
        assert "IMPROVED" in output
        assert "DRIFT STATUS" not in output

    def test_no_drift_eval_no_drift_adaptation_block(self, capsys):
        """Without drift_eval the DRIFT ADAPTATION block must not appear."""
        _print_summary(_minimal_report())
        output = capsys.readouterr().out
        assert "DRIFT ADAPTATION" not in output


# ── request_approval: drift kwarg not shown; drift_eval kwarg shown ──────────

class TestRequestApprovalDrift:
    def test_drift_kwarg_shown_in_summary(self, monkeypatch, capsys):
        """Passing drift= to request_approval renders the DRIFT STATUS block."""
        monkeypatch.setattr("builtins.input", lambda prompt: "1")
        drift = _drift_dict(severity="low")
        result = request_approval(_minimal_report(), drift=drift)
        output = capsys.readouterr().out
        assert "DRIFT STATUS" in output
        assert result.approved is True

    def test_no_drift_kwarg_shows_no_data_banner(self, monkeypatch, capsys):
        """Default drift=None renders the explicit 'no drift data' banner."""
        monkeypatch.setattr("builtins.input", lambda prompt: "1")
        result = request_approval(_minimal_report())
        output = capsys.readouterr().out
        assert "DRIFT STATUS" in output
        assert "No drift data available" in output
        assert result.approved is True

    def test_drift_eval_kwarg_shown_in_summary(self, monkeypatch, capsys):
        """Passing drift_eval= renders the DRIFTED DATA block before the prompt."""
        monkeypatch.setattr("builtins.input", lambda prompt: "1")
        drift_eval = {
            "n_holdout_images": 4,
            "baseline": {"accuracy": 0.5, "f1_score": 0.33, "precision": 0.25, "recall": 0.5},
            "after_finetuning": {"accuracy": 1.0, "f1_score": 1.0, "precision": 1.0, "recall": 1.0},
            "delta": {"accuracy": 0.5, "f1_score": 0.67, "precision": 0.75, "recall": 0.5},
            "improved": True,
        }
        result = request_approval(_minimal_report(), drift_eval=drift_eval)
        output = capsys.readouterr().out
        assert "DRIFTED DATA" in output
        assert "IMPROVED" in output
        assert result.approved is True


# ── _print_drift_eval_block ──────────────────────────────────────────────────

class TestPrintDriftEvalBlock:
    def _eval_dict(self, improved=True):
        return {
            "n_holdout_images": 4,
            "baseline": {"accuracy": 0.5, "f1_score": 0.33, "precision": 0.25, "recall": 0.5},
            "after_finetuning": {"accuracy": 1.0, "f1_score": 1.0, "precision": 1.0, "recall": 1.0},
            "delta": {"accuracy": 0.5, "f1_score": 0.67, "precision": 0.75, "recall": 0.5},
            "improved": improved,
        }

    def test_shows_drifted_data_header(self, capsys):
        _print_drift_eval_block(self._eval_dict())
        output = capsys.readouterr().out
        assert "DRIFTED DATA" in output

    def test_shows_improved_when_improved(self, capsys):
        _print_drift_eval_block(self._eval_dict(improved=True))
        output = capsys.readouterr().out
        assert "IMPROVED" in output

    def test_shows_no_improvement_when_not_improved(self, capsys):
        _print_drift_eval_block(self._eval_dict(improved=False))
        output = capsys.readouterr().out
        assert "NO IMPROVEMENT" in output

    def test_shows_holdout_count(self, capsys):
        _print_drift_eval_block(self._eval_dict())
        output = capsys.readouterr().out
        assert "4" in output

    def test_shows_before_and_after_accuracy(self, capsys):
        _print_drift_eval_block(self._eval_dict())
        output = capsys.readouterr().out
        assert "0.5000" in output
        assert "1.0000" in output


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
