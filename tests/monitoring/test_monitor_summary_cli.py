"""Tests for src.monitoring.monitor_summary_cli.

Covers:
- _load_results: skips *_decision.json, skips files without 'overall', sorts by generated_at
- _trend_label: WORSENING / IMPROVING / STABLE / insufficient data
- main(): tabular format (current_dataset.row_count), image format (batch.size)
- main(): --last N limits results
- main(): missing directory exits cleanly (code 0)
- main(): empty directory exits cleanly (code 0)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.monitoring.monitor_summary_cli import _load_results, _trend_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_result(directory: Path, filename: str, data: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(json.dumps(data), encoding="utf-8")


def _image_result(
    timestamp: str = "2024-01-01T00:00:00",
    severity: str = "low",
    score: float = 0.05,
    batch_size: int = 50,
    detected: bool = False,
    method: str = "statistical",
) -> dict:
    return {
        "schema_version": "1.0.0",
        "drift_type": "image_statistical",
        "method": method,
        "generated_at": timestamp,
        "batch": {"size": batch_size},
        "overall": {
            "drift_score": score,
            "severity": severity,
            "dataset_drift_detected": detected,
        },
    }


def _tabular_result(
    timestamp: str = "2024-01-01T00:00:00",
    severity: str = "low",
    score: float = 0.10,
    row_count: int = 200,
    detected: bool = False,
) -> dict:
    return {
        "schema_version": "1.0.0",
        "drift_type": "tabular",
        "method": "evidently",
        "generated_at": timestamp,
        "current_dataset": {"row_count": row_count},
        "overall": {
            "drift_share": score,
            "severity": severity,
            "dataset_drift_detected": detected,
        },
    }


# ---------------------------------------------------------------------------
# _load_results
# ---------------------------------------------------------------------------

class TestLoadResults:
    def test_returns_empty_for_empty_dir(self, tmp_path):
        assert _load_results(tmp_path) == []

    def test_skips_decision_json(self, tmp_path):
        _write_result(tmp_path, "20240101T000000Z_decision.json", {"overall": {}})
        results = _load_results(tmp_path)
        assert results == []

    def test_skips_file_without_overall(self, tmp_path):
        _write_result(tmp_path, "20240101T000000Z.json", {"not_overall": {}})
        results = _load_results(tmp_path)
        assert results == []

    def test_loads_valid_result(self, tmp_path):
        _write_result(tmp_path, "20240101T000000Z.json", _image_result())
        results = _load_results(tmp_path)
        assert len(results) == 1

    def test_sorts_by_generated_at(self, tmp_path):
        _write_result(tmp_path, "b.json", _image_result(timestamp="2024-02-01T00:00:00"))
        _write_result(tmp_path, "a.json", _image_result(timestamp="2024-01-01T00:00:00"))
        results = _load_results(tmp_path)
        assert results[0]["generated_at"].startswith("2024-01")
        assert results[1]["generated_at"].startswith("2024-02")

    def test_source_file_attached(self, tmp_path):
        _write_result(tmp_path, "20240101T000000Z.json", _image_result())
        results = _load_results(tmp_path)
        assert results[0]["_source_file"] == "20240101T000000Z.json"

    def test_skips_invalid_json(self, tmp_path):
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        results = _load_results(tmp_path)
        assert results == []

    def test_loads_multiple_results(self, tmp_path):
        for i in range(3):
            _write_result(
                tmp_path,
                f"result_{i}.json",
                _image_result(timestamp=f"2024-0{i+1}-01T00:00:00"),
            )
        results = _load_results(tmp_path)
        assert len(results) == 3

    def test_decision_and_result_same_timestamp(self, tmp_path):
        """Decision JSON is skipped; result JSON is loaded."""
        ts = "20240101T120000Z"
        _write_result(tmp_path, f"{ts}.json", _image_result())
        _write_result(tmp_path, f"{ts}_decision.json", {"overall": {}, "decision": "approve"})
        results = _load_results(tmp_path)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# _trend_label
# ---------------------------------------------------------------------------

class TestTrendLabel:
    def test_insufficient_data_single(self):
        assert _trend_label(["low"]) == "insufficient data"

    def test_insufficient_data_empty(self):
        assert _trend_label([]) == "insufficient data"

    def test_worsening(self):
        assert _trend_label(["low", "low", "medium", "high"]) == "WORSENING"

    def test_improving(self):
        assert _trend_label(["high", "medium", "low"]) == "IMPROVING"

    def test_stable(self):
        assert _trend_label(["medium", "medium", "medium"]) == "STABLE"

    def test_uses_last_five(self):
        # First value is high, but last 5 are stable at medium
        sevs = ["high"] + ["medium"] * 5
        assert _trend_label(sevs) == "STABLE"

    def test_two_values_stable(self):
        assert _trend_label(["low", "low"]) == "STABLE"


# ---------------------------------------------------------------------------
# main() — integration via subprocess / capsys
# ---------------------------------------------------------------------------

class TestMain:
    def test_missing_dir_exits_zero(self, tmp_path, capsys):
        from src.monitoring.monitor_summary_cli import main
        import sys

        with pytest.raises(SystemExit) as exc:
            sys.argv = ["monitor-summary", "--output-dir", str(tmp_path / "nonexistent")]
            main()
        assert exc.value.code == 0

    def test_empty_dir_exits_zero(self, tmp_path, capsys):
        from src.monitoring.monitor_summary_cli import main
        import sys

        output_dir = tmp_path / "monitoring"
        output_dir.mkdir()
        with pytest.raises(SystemExit) as exc:
            sys.argv = ["monitor-summary", "--output-dir", str(output_dir)]
            main()
        assert exc.value.code == 0

    def test_image_results_printed(self, tmp_path, capsys):
        from src.monitoring.monitor_summary_cli import main
        import sys

        output_dir = tmp_path / "monitoring"
        _write_result(output_dir, "r1.json", _image_result(
            timestamp="2024-01-01T10:00:00", severity="high", detected=True, batch_size=100
        ))

        sys.argv = ["monitor-summary", "--output-dir", str(output_dir)]
        main()

        out = capsys.readouterr().out
        assert "HIGH" in out
        assert "YES" in out
        assert "100" in out

    def test_tabular_results_printed(self, tmp_path, capsys):
        from src.monitoring.monitor_summary_cli import main
        import sys

        output_dir = tmp_path / "monitoring"
        _write_result(output_dir, "r1.json", _tabular_result(
            timestamp="2024-01-01T10:00:00", severity="medium", row_count=200
        ))

        sys.argv = ["monitor-summary", "--output-dir", str(output_dir)]
        main()

        out = capsys.readouterr().out
        assert "MEDIUM" in out
        assert "200" in out

    def test_last_n_limits_output(self, tmp_path, capsys):
        from src.monitoring.monitor_summary_cli import main
        import sys

        output_dir = tmp_path / "monitoring"
        for i in range(5):
            _write_result(
                output_dir,
                f"r{i}.json",
                _image_result(timestamp=f"2024-0{i+1}-01T00:00:00", severity="low"),
            )

        sys.argv = ["monitor-summary", "--output-dir", str(output_dir), "--last", "2"]
        main()

        out = capsys.readouterr().out
        # "Runs shown: 2" should appear in summary
        assert "Runs shown:    2" in out

    def test_trend_shown_in_output(self, tmp_path, capsys):
        from src.monitoring.monitor_summary_cli import main
        import sys

        output_dir = tmp_path / "monitoring"
        severities = ["low", "low", "medium", "high", "high"]
        for i, sev in enumerate(severities):
            _write_result(
                output_dir,
                f"r{i}.json",
                _image_result(timestamp=f"2024-0{i+1}-01T00:00:00", severity=sev),
            )

        sys.argv = ["monitor-summary", "--output-dir", str(output_dir)]
        main()

        out = capsys.readouterr().out
        assert "WORSENING" in out

    def test_model_name_resolves_to_output_dir(self, tmp_path, capsys, monkeypatch):
        """--model-name resolves to outputs/drift_monitoring/<name>/."""
        from src.monitoring import monitor_summary_cli
        import sys

        output_dir = tmp_path / "outputs" / "drift_monitoring" / "my_model"
        _write_result(output_dir, "r1.json", _tabular_result())

        monkeypatch.chdir(tmp_path)
        sys.argv = ["monitor-summary", "--model-name", "my_model"]
        monitor_summary_cli.main()

        out = capsys.readouterr().out
        assert "my_model" in out or "LOW" in out

    def test_drift_detected_counts(self, tmp_path, capsys):
        from src.monitoring.monitor_summary_cli import main
        import sys

        output_dir = tmp_path / "monitoring"
        _write_result(output_dir, "r1.json", _image_result(detected=True))
        _write_result(output_dir, "r2.json", _image_result(detected=False))
        _write_result(output_dir, "r3.json", _image_result(detected=True))

        sys.argv = ["monitor-summary", "--output-dir", str(output_dir)]
        main()

        out = capsys.readouterr().out
        assert "2 / 3" in out
