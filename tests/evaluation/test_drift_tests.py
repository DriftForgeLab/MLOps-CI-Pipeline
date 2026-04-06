"""Tests for src.evaluation.drift_tests — orchestration layer.

These tests mock the underlying drift layers (alignment, compute, interpret)
to verify that run_drift_analysis wires them together correctly.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config.schema import (
    DriftConfig,
    DriftFeatureSeverityConfig,
    DriftSeverityConfig,
    DriftStatTestConfig,
    DriftStatTestThresholdConfig,
)
from src.evaluation.drift_tests import run_drift_analysis


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _default_drift_config(**overrides) -> DriftConfig:
    return DriftConfig(**overrides)


def _mock_pipeline_config(tmp_path: Path) -> MagicMock:
    """Build a mock PipelineConfig with paths rooted in tmp_path."""
    config = MagicMock()
    config.dataset = "iris"
    config.task_type = "classification"
    config.data.processed = str(tmp_path / "processed")
    config.data.drift_scenarios = str(tmp_path / "drift_scenarios")
    config.output_dir = str(tmp_path / "outputs")
    return config


def _setup_preprocessed_data(tmp_path: Path, version_id: str) -> pd.DataFrame:
    """Create minimal preprocessed CSV files and feature_map.json in tmp_path."""
    import json

    preprocessed_dir = (
        tmp_path / "processed" / "iris" / version_id / "preprocessed"
    )
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7, 5.0, 5.4],
        "sepal_width": [3.5, 3.0, 3.2, 3.6, 3.9],
        "petal_length": [1.4, 1.4, 1.3, 1.4, 1.7],
        "petal_width": [0.2, 0.2, 0.2, 0.2, 0.4],
        "species": [0, 0, 0, 0, 0],
    })

    df.to_csv(preprocessed_dir / "train.csv", index=False)
    df.to_csv(preprocessed_dir / "val.csv", index=False)

    feature_map = {
        "output_features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target": "species",
    }
    with open(preprocessed_dir / "feature_map.json", "w") as f:
        json.dump(feature_map, f)

    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunDriftAnalysis:
    """Tests for the run_drift_analysis orchestration function."""

    def test_calls_all_layers_in_order(self, tmp_path: Path):
        """Verify the orchestration calls alignment, compute, and interpret."""
        version_id = "v1"
        config = _mock_pipeline_config(tmp_path)
        drift_config = _default_drift_config()
        df = _setup_preprocessed_data(tmp_path, version_id)

        fake_evidently_report = MagicMock()
        fake_evidently_result = {"metrics": []}
        fake_per_feature = {
            "sepal_length": {
                "column_type": "num",
                "drift_detected": False,
                "drift_score": 1.0,
                "stattest_name": "ks",
                "stattest_threshold": 0.05,
            }
        }
        fake_overall = {
            "dataset_drift_detected": False,
            "drift_share": 0.0,
            "drifted_feature_count": 0,
            "total_feature_count": 4,
        }

        with (
            patch("src.evaluation.drift_tests.run_evidently_drift") as mock_compute,
            patch("src.evaluation.drift_tests.extract_per_feature_results") as mock_extract_feat,
            patch("src.evaluation.drift_tests.extract_overall_results") as mock_extract_overall,
            patch("src.evaluation.drift_tests.build_drift_result") as mock_build,
        ):
            mock_compute.return_value = (fake_evidently_result, fake_evidently_report)
            mock_extract_feat.return_value = fake_per_feature
            mock_extract_overall.return_value = fake_overall
            mock_build.return_value = {
                "overall": {"severity": "low", "drifted_feature_count": 0, "total_feature_count": 4}
            }

            result, report = run_drift_analysis(config, version_id, drift_config)

        mock_compute.assert_called_once()
        mock_extract_feat.assert_called_once_with(fake_evidently_result)
        mock_extract_overall.assert_called_once_with(fake_evidently_result)
        mock_build.assert_called_once()
        assert report is fake_evidently_report

    def test_returns_standard_drift_result_keys(self, tmp_path: Path):
        """Result dict from build_drift_result is returned as-is."""
        version_id = "v2"
        config = _mock_pipeline_config(tmp_path)
        drift_config = _default_drift_config()
        _setup_preprocessed_data(tmp_path, version_id)

        expected_result = {
            "schema_version": "1.0.0",
            "drift_type": "tabular",
            "overall": {
                "severity": "low",
                "drifted_feature_count": 0,
                "total_feature_count": 4,
            },
            "features": {},
        }

        with (
            patch("src.evaluation.drift_tests.run_evidently_drift") as mock_compute,
            patch("src.evaluation.drift_tests.extract_per_feature_results", return_value={}),
            patch("src.evaluation.drift_tests.extract_overall_results", return_value={
                "dataset_drift_detected": False, "drift_share": 0.0,
                "drifted_feature_count": 0, "total_feature_count": 4,
            }),
            patch("src.evaluation.drift_tests.build_drift_result", return_value=expected_result),
        ):
            mock_compute.return_value = ({}, MagicMock())
            result, _ = run_drift_analysis(config, version_id, drift_config)

        assert result == expected_result

    def test_raises_on_alignment_error(self, tmp_path: Path):
        """Should raise ValueError when feature alignment fails."""
        version_id = "v3"
        config = _mock_pipeline_config(tmp_path)
        drift_config = _default_drift_config()
        _setup_preprocessed_data(tmp_path, version_id)

        with patch(
            "src.evaluation.drift_tests.validate_feature_alignment",
            return_value=["Features missing in current data: ['col_x']"],
        ):
            with pytest.raises(ValueError, match="Feature alignment failed"):
                run_drift_analysis(config, version_id, drift_config)

    def test_raises_on_missing_val_csv(self, tmp_path: Path):
        """Should raise FileNotFoundError when val.csv doesn't exist."""
        import json

        version_id = "v4"
        config = _mock_pipeline_config(tmp_path)
        drift_config = _default_drift_config()

        # Create only train.csv and feature_map, but no val.csv
        preprocessed_dir = (
            tmp_path / "processed" / "iris" / version_id / "preprocessed"
        )
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df.to_csv(preprocessed_dir / "train.csv", index=False)

        feature_map = {"output_features": ["a", "b"], "target": "t"}
        with open(preprocessed_dir / "feature_map.json", "w") as f:
            json.dump(feature_map, f)

        with pytest.raises(FileNotFoundError, match="Validation split not found"):
            run_drift_analysis(config, version_id, drift_config)

    def test_raises_on_missing_reference_data(self, tmp_path: Path):
        """Should raise FileNotFoundError when reference data is missing."""
        version_id = "v5"
        config = _mock_pipeline_config(tmp_path)
        drift_config = _default_drift_config()

        # Don't create any preprocessed files
        with pytest.raises(FileNotFoundError):
            run_drift_analysis(config, version_id, drift_config)

    def test_feature_columns_exclude_target(self, tmp_path: Path):
        """Feature columns passed to run_evidently_drift should not include target."""
        version_id = "v6"
        config = _mock_pipeline_config(tmp_path)
        drift_config = _default_drift_config()
        _setup_preprocessed_data(tmp_path, version_id)

        with (
            patch("src.evaluation.drift_tests.run_evidently_drift") as mock_compute,
            patch("src.evaluation.drift_tests.extract_per_feature_results", return_value={}),
            patch("src.evaluation.drift_tests.extract_overall_results", return_value={
                "dataset_drift_detected": False, "drift_share": 0.0,
                "drifted_feature_count": 0, "total_feature_count": 4,
            }),
            patch("src.evaluation.drift_tests.build_drift_result", return_value={
                "overall": {"severity": "low", "drifted_feature_count": 0, "total_feature_count": 4}
            }),
        ):
            mock_compute.return_value = ({}, MagicMock())
            run_drift_analysis(config, version_id, drift_config)

        # Inspect the feature_columns arg passed to run_evidently_drift
        call_args = mock_compute.call_args
        feature_columns = call_args.kwargs.get("feature_columns") or call_args[0][2]
        assert "species" not in feature_columns
        assert "sepal_length" in feature_columns
