"""Tests for src.monitoring.reports — drift reporting (CLI + JSON + HTML)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.monitoring.reports import (
    print_drift_summary,
    save_drift_report_html,
    save_drift_report_json,
)


# ---------------------------------------------------------------------------
# Helpers: build a complete drift_result dict conforming to the standard schema
# ---------------------------------------------------------------------------


def _drift_result(
    *,
    overall_severity: str = "high",
    features: dict | None = None,
    drift_share: float = 0.75,
    drifted_count: int = 3,
    total_count: int = 4,
    dataset_drift_detected: bool = True,
    task_type: str = "classification",
    reference_source: str = "train",
    current_source: str = "val",
) -> dict:
    """Build a complete drift_result dict for tests."""
    if features is None:
        features = {
            "sepal_length": {
                "column_type": "num",
                "drift_detected": True,
                "drift_score": 0.0022,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "high",
            },
            "sepal_width": {
                "column_type": "num",
                "drift_detected": False,
                "drift_score": 0.3200,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "low",
            },
            "petal_length": {
                "column_type": "num",
                "drift_detected": True,
                "drift_score": 0.0080,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "medium",
            },
            "petal_width": {
                "column_type": "num",
                "drift_detected": True,
                "drift_score": 0.0150,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
                "severity": "low",
            },
        }
    return {
        "schema_version": "1.0.0",
        "drift_type": "tabular",
        "generated_at": "2026-04-05T12:00:00+00:00",
        "pipeline_execution_id": "exec-abc123",
        "dataset_version_id": "dv-a1b2c3",
        "task_type": task_type,
        "reference_dataset": {
            "source": reference_source,
            "path": "data/processed/iris/v1/preprocessed/train.csv",
            "row_count": 105,
            "feature_count": 4,
        },
        "current_dataset": {
            "source": current_source,
            "path": "data/processed/iris/v1/preprocessed/val.csv",
            "row_count": 45,
            "feature_count": 4,
        },
        "overall": {
            "dataset_drift_detected": dataset_drift_detected,
            "drift_share": drift_share,
            "drifted_feature_count": drifted_count,
            "total_feature_count": total_count,
            "severity": overall_severity,
        },
        "features": features,
        "artifacts": {},
        "config_snapshot": {
            "stattest_numerical": "ks",
            "stattest_categorical": "chisquare",
            "stattest_threshold_numerical": 0.05,
            "stattest_threshold_categorical": 0.05,
            "drift_share_threshold": 0.5,
            "severity_low_max": 0.25,
            "severity_medium_max": 0.50,
        },
    }


# ===========================================================================
# print_drift_summary
# ===========================================================================


class TestPrintDriftSummary:
    def test_prints_header_and_separator(self, capsys):
        print_drift_summary(_drift_result())
        captured = capsys.readouterr().out
        assert "DRIFT ANALYSIS SUMMARY" in captured
        assert "=" * 60 in captured

    def test_prints_reference_and_current_metadata(self, capsys):
        print_drift_summary(_drift_result())
        out = capsys.readouterr().out
        # Reference metadata
        assert "train" in out
        assert "105" in out
        # Current metadata
        assert "val" in out
        assert "45" in out

    def test_prints_task_type(self, capsys):
        print_drift_summary(_drift_result(task_type="classification"))
        out = capsys.readouterr().out
        assert "classification" in out

    def test_prints_overall_drift_detected_and_severity(self, capsys):
        print_drift_summary(_drift_result(overall_severity="high"))
        out = capsys.readouterr().out
        assert "DETECTED" in out
        assert "HIGH" in out

    def test_prints_overall_drift_not_detected(self, capsys):
        result = _drift_result(
            dataset_drift_detected=False,
            drift_share=0.0,
            drifted_count=0,
            total_count=4,
            overall_severity="low",
            features={
                "x": {
                    "column_type": "num",
                    "drift_detected": False,
                    "drift_score": 0.9,
                    "stattest_name": "ks",
                    "stattest_threshold": 0.05,
                    "severity": "low",
                }
            },
        )
        print_drift_summary(result)
        out = capsys.readouterr().out
        assert "NOT DETECTED" in out

    def test_prints_drift_share_percentage(self, capsys):
        print_drift_summary(_drift_result(drift_share=0.75, drifted_count=3, total_count=4))
        out = capsys.readouterr().out
        assert "3 / 4" in out
        assert "75" in out  # 75% or 75.0%

    def test_prints_each_feature_row(self, capsys):
        print_drift_summary(_drift_result())
        out = capsys.readouterr().out
        for feat in ("sepal_length", "sepal_width", "petal_length", "petal_width"):
            assert feat in out

    def test_prints_drifted_and_ok_labels(self, capsys):
        print_drift_summary(_drift_result())
        out = capsys.readouterr().out
        assert "DRIFTED" in out
        assert "OK" in out

    def test_prints_stattest_name_for_features(self, capsys):
        print_drift_summary(_drift_result())
        out = capsys.readouterr().out
        assert "ks" in out

    def test_prints_per_feature_severity(self, capsys):
        print_drift_summary(_drift_result())
        out = capsys.readouterr().out
        # All three severity levels appear in the fixture
        assert "high" in out
        assert "medium" in out
        assert "low" in out

    def test_no_recommendation_lines(self, capsys):
        """Plan requires no recommendation/action lines in drift summary."""
        print_drift_summary(_drift_result())
        out = capsys.readouterr().out
        assert "Recommendation" not in out
        assert "RETRAIN" not in out
        assert "COLLECT_DATA" not in out
        assert "MONITOR" not in out

    def test_no_ansi_color_codes(self, capsys):
        """Project style is color-neutral (matches _print_summary in approval.py)."""
        print_drift_summary(_drift_result())
        out = capsys.readouterr().out
        assert "\x1b[" not in out  # no ANSI escape sequences


# ===========================================================================
# save_drift_report_json
# ===========================================================================


class TestSaveDriftReportJson:
    def test_writes_file_to_output_dir(self, tmp_path):
        result = _drift_result()
        path = save_drift_report_json(result, tmp_path)
        assert path == tmp_path / "drift_report.json"
        assert path.exists()

    def test_returns_path_to_written_file(self, tmp_path):
        result = _drift_result()
        path = save_drift_report_json(result, tmp_path)
        assert isinstance(path, Path)
        assert path.name == "drift_report.json"

    def test_written_json_matches_schema(self, tmp_path):
        result = _drift_result()
        path = save_drift_report_json(result, tmp_path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["schema_version"] == "1.0.0"
        assert loaded["drift_type"] == "tabular"
        assert loaded["overall"]["severity"] == "high"
        assert "sepal_length" in loaded["features"]
        assert "recommendation" not in loaded

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "nested" / "drift"
        assert not nested.exists()
        path = save_drift_report_json(_drift_result(), nested)
        assert path.exists()
        assert nested.is_dir()

    def test_overwrites_existing_file(self, tmp_path):
        # First write
        save_drift_report_json(_drift_result(overall_severity="low"), tmp_path)
        # Second write with different content
        path = save_drift_report_json(
            _drift_result(overall_severity="high"), tmp_path
        )
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["overall"]["severity"] == "high"

    def test_sanitizes_nan_values(self, tmp_path):
        """atomic_write_json routes through sanitize_for_json → NaN becomes null."""
        result = _drift_result()
        result["features"]["sepal_length"]["drift_score"] = float("nan")
        path = save_drift_report_json(result, tmp_path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded["features"]["sepal_length"]["drift_score"] is None


# ===========================================================================
# save_drift_report_html
# ===========================================================================


class TestSaveDriftReportHtml:
    def test_calls_save_html_with_correct_path(self, tmp_path):
        mock_report = MagicMock()
        path = save_drift_report_html(mock_report, tmp_path)
        expected = tmp_path / "drift_report.html"
        assert path == expected
        mock_report.save_html.assert_called_once_with(str(expected))

    def test_returns_path(self, tmp_path):
        mock_report = MagicMock()
        path = save_drift_report_html(mock_report, tmp_path)
        assert isinstance(path, Path)
        assert path.name == "drift_report.html"

    def test_creates_output_dir_if_missing(self, tmp_path):
        nested = tmp_path / "nested" / "drift"
        assert not nested.exists()
        mock_report = MagicMock()
        save_drift_report_html(mock_report, nested)
        assert nested.is_dir()
