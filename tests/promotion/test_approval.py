import pytest

from src.promotion.approval import ApprovalResult, request_approval, _print_summary


# ── helpers ─────────────────────────────────────────────────────────────────

def _minimal_report(metrics=None, comparison=None):
    return {
        "metrics": metrics or {"accuracy": 0.95, "f1_score": 0.88},
        "comparison": comparison or {
            "has_production_model": False,
            "overall_verdict": "no_baseline",
        },
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
