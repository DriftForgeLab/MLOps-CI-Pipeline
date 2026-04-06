"""Tests for src.monitoring.drift — batch drift monitoring."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config.schema import DriftConfig, DriftMonitoringConfig
from src.monitoring.drift import monitor_batch, load_reference_for_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_drift_config(**overrides) -> DriftConfig:
    return DriftConfig(**overrides)


def _feature_map() -> dict:
    return {
        "output_features": ["feat_a", "feat_b", "feat_c"],
        "target": "label",
    }


def _reference_df() -> pd.DataFrame:
    return pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
        "feat_c": [0.1, 0.2, 0.3, 0.4, 0.5] * 10,
        "label": [0, 1, 0, 1, 0] * 10,
    })


def _identical_batch_df() -> pd.DataFrame:
    """Batch identical to reference — should show low drift."""
    return pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0] * 10,
        "feat_c": [0.1, 0.2, 0.3, 0.4, 0.5] * 10,
        "label": [0, 1, 0, 1, 0] * 10,
    })


def _shifted_batch_df() -> pd.DataFrame:
    """Batch with shifted distributions — should trigger drift."""
    return pd.DataFrame({
        "feat_a": [100.0, 200.0, 300.0, 400.0, 500.0] * 10,
        "feat_b": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0] * 10,
        "feat_c": [99.0, 99.1, 99.2, 99.3, 99.4] * 10,
        "label": [0, 1, 0, 1, 0] * 10,
    })


def _fake_evidently_result(drifted: bool = False) -> dict:
    """Minimal Evidently-like result for mocking."""
    return {"metrics": []}


def _fake_per_feature(drifted: bool = False) -> dict:
    if drifted:
        return {
            "feat_a": {
                "column_type": "num", "drift_detected": True,
                "drift_score": 0.0001, "stattest_name": "ks", "stattest_threshold": 0.05,
            },
            "feat_b": {
                "column_type": "num", "drift_detected": True,
                "drift_score": 0.0005, "stattest_name": "ks", "stattest_threshold": 0.05,
            },
            "feat_c": {
                "column_type": "num", "drift_detected": True,
                "drift_score": 0.0003, "stattest_name": "ks", "stattest_threshold": 0.05,
            },
        }
    return {
        "feat_a": {
            "column_type": "num", "drift_detected": False,
            "drift_score": 0.95, "stattest_name": "ks", "stattest_threshold": 0.05,
        },
        "feat_b": {
            "column_type": "num", "drift_detected": False,
            "drift_score": 0.88, "stattest_name": "ks", "stattest_threshold": 0.05,
        },
        "feat_c": {
            "column_type": "num", "drift_detected": False,
            "drift_score": 0.72, "stattest_name": "ks", "stattest_threshold": 0.05,
        },
    }


def _fake_overall(drifted: bool = False) -> dict:
    if drifted:
        return {
            "dataset_drift_detected": True,
            "drift_share": 1.0,
            "drifted_feature_count": 3,
            "total_feature_count": 3,
        }
    return {
        "dataset_drift_detected": False,
        "drift_share": 0.0,
        "drifted_feature_count": 0,
        "total_feature_count": 3,
    }


def _fake_drift_result(severity: str = "low") -> dict:
    return {
        "schema_version": "1.0.0",
        "drift_type": "tabular",
        "overall": {
            "severity": severity,
            "dataset_drift_detected": severity != "low",
            "drift_share": 1.0 if severity != "low" else 0.0,
            "drifted_feature_count": 3 if severity != "low" else 0,
            "total_feature_count": 3,
        },
        "features": {},
        "reference_dataset": {"source": "train", "row_count": 50, "feature_count": 3},
        "current_dataset": {"source": "batch", "row_count": 50, "feature_count": 3},
        "task_type": "unknown",
    }


# ---------------------------------------------------------------------------
# Tests — monitor_batch
# ---------------------------------------------------------------------------

class TestMonitorBatch:

    def test_below_min_batch_size_returns_none(self):
        """Batch smaller than min_batch_size should return None with a warning."""
        config = _default_drift_config(
            monitoring=DriftMonitoringConfig(min_batch_size=100)
        )
        small_batch = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0], "feat_c": [3.0]})

        result = monitor_batch(
            batch_df=small_batch,
            reference_df=_reference_df(),
            feature_map=_feature_map(),
            drift_config=config,
            model_name="test_model",
        )

        assert result is None

    def test_below_min_batch_size_logs_warning(self, caplog):
        """Should log a warning when batch is too small."""
        config = _default_drift_config(
            monitoring=DriftMonitoringConfig(min_batch_size=100)
        )
        small_batch = pd.DataFrame({"feat_a": [1.0], "feat_b": [2.0], "feat_c": [3.0]})

        with caplog.at_level(logging.WARNING):
            monitor_batch(
                batch_df=small_batch,
                reference_df=_reference_df(),
                feature_map=_feature_map(),
                drift_config=config,
                model_name="test_model",
            )

        assert "min_batch_size" in caplog.text
        assert "test_model" in caplog.text

    def test_identical_batch_low_severity(self):
        """Identical data should produce low severity."""
        config = _default_drift_config()

        with (
            patch("src.monitoring.drift.run_evidently_drift") as mock_compute,
            patch("src.monitoring.drift.extract_per_feature_results", return_value=_fake_per_feature(drifted=False)),
            patch("src.monitoring.drift.extract_overall_results", return_value=_fake_overall(drifted=False)),
            patch("src.monitoring.drift.build_drift_result", return_value=_fake_drift_result("low")),
        ):
            mock_compute.return_value = (_fake_evidently_result(), MagicMock())

            result = monitor_batch(
                batch_df=_identical_batch_df(),
                reference_df=_reference_df(),
                feature_map=_feature_map(),
                drift_config=config,
            )

        assert result is not None
        assert result["overall"]["severity"] == "low"

    def test_shifted_batch_detected_drift(self):
        """Shifted data should produce high severity."""
        config = _default_drift_config()

        with (
            patch("src.monitoring.drift.run_evidently_drift") as mock_compute,
            patch("src.monitoring.drift.extract_per_feature_results", return_value=_fake_per_feature(drifted=True)),
            patch("src.monitoring.drift.extract_overall_results", return_value=_fake_overall(drifted=True)),
            patch("src.monitoring.drift.build_drift_result", return_value=_fake_drift_result("high")),
        ):
            mock_compute.return_value = (_fake_evidently_result(), MagicMock())

            result = monitor_batch(
                batch_df=_shifted_batch_df(),
                reference_df=_reference_df(),
                feature_map=_feature_map(),
                drift_config=config,
            )

        assert result is not None
        assert result["overall"]["severity"] == "high"
        assert result["overall"]["drifted_feature_count"] == 3

    def test_alert_threshold_triggers_warning(self, caplog):
        """Severity >= alert_severity should log a WARNING."""
        config = _default_drift_config(
            monitoring=DriftMonitoringConfig(alert_severity="medium")
        )

        with (
            patch("src.monitoring.drift.run_evidently_drift") as mock_compute,
            patch("src.monitoring.drift.extract_per_feature_results", return_value=_fake_per_feature(drifted=True)),
            patch("src.monitoring.drift.extract_overall_results", return_value=_fake_overall(drifted=True)),
            patch("src.monitoring.drift.build_drift_result", return_value=_fake_drift_result("high")),
        ):
            mock_compute.return_value = (_fake_evidently_result(), MagicMock())

            with caplog.at_level(logging.WARNING):
                monitor_batch(
                    batch_df=_shifted_batch_df(),
                    reference_df=_reference_df(),
                    feature_map=_feature_map(),
                    drift_config=config,
                    model_name="my_model",
                )

        assert "Drift alert" in caplog.text
        assert "my_model" in caplog.text

    def test_below_alert_threshold_no_warning(self, caplog):
        """Severity below alert_severity should not log a drift alert."""
        config = _default_drift_config(
            monitoring=DriftMonitoringConfig(alert_severity="high")
        )

        with (
            patch("src.monitoring.drift.run_evidently_drift") as mock_compute,
            patch("src.monitoring.drift.extract_per_feature_results", return_value=_fake_per_feature(drifted=False)),
            patch("src.monitoring.drift.extract_overall_results", return_value=_fake_overall(drifted=False)),
            patch("src.monitoring.drift.build_drift_result", return_value=_fake_drift_result("low")),
        ):
            mock_compute.return_value = (_fake_evidently_result(), MagicMock())

            with caplog.at_level(logging.WARNING):
                monitor_batch(
                    batch_df=_identical_batch_df(),
                    reference_df=_reference_df(),
                    feature_map=_feature_map(),
                    drift_config=config,
                )

        assert "Drift alert" not in caplog.text

    def test_alignment_error_raises(self):
        """Should raise ValueError when features don't align."""
        config = _default_drift_config()
        bad_batch = pd.DataFrame({"wrong_col": [1.0] * 50})

        with pytest.raises(ValueError, match="Feature alignment failed"):
            monitor_batch(
                batch_df=bad_batch,
                reference_df=_reference_df(),
                feature_map=_feature_map(),
                drift_config=config,
            )

    def test_current_dataset_source_is_batch(self):
        """The current_info passed to build_drift_result should have source='batch'."""
        config = _default_drift_config()

        with (
            patch("src.monitoring.drift.run_evidently_drift") as mock_compute,
            patch("src.monitoring.drift.extract_per_feature_results", return_value=_fake_per_feature()),
            patch("src.monitoring.drift.extract_overall_results", return_value=_fake_overall()),
            patch("src.monitoring.drift.build_drift_result", return_value=_fake_drift_result()) as mock_build,
        ):
            mock_compute.return_value = (_fake_evidently_result(), MagicMock())

            monitor_batch(
                batch_df=_identical_batch_df(),
                reference_df=_reference_df(),
                feature_map=_feature_map(),
                drift_config=config,
            )

        current_info = mock_build.call_args.kwargs["current_info"]
        assert current_info["source"] == "batch"


# ---------------------------------------------------------------------------
# Tests — load_reference_for_model
# ---------------------------------------------------------------------------

class TestLoadReferenceForModel:

    def test_delegates_to_load_reference_data(self):
        """Should call load_reference_data with the correct arguments."""
        model_info = {
            "dataset_name": "iris",
            "dataset_version_id": "abc123",
        }

        with patch("src.monitoring.drift.load_reference_data") as mock_load:
            mock_load.return_value = (pd.DataFrame(), {})
            load_reference_for_model(model_info, "data/processed")

        mock_load.assert_called_once_with(
            dataset_name="iris",
            version_id="abc123",
            reference_source="train",
            processed_dir="data/processed",
        )

    def test_custom_reference_source(self):
        """Should pass custom reference_source through."""
        model_info = {
            "dataset_name": "iris",
            "dataset_version_id": "abc123",
        }

        with patch("src.monitoring.drift.load_reference_data") as mock_load:
            mock_load.return_value = (pd.DataFrame(), {})
            load_reference_for_model(model_info, "data/processed", reference_source="val")

        mock_load.assert_called_once_with(
            dataset_name="iris",
            version_id="abc123",
            reference_source="val",
            processed_dir="data/processed",
        )
