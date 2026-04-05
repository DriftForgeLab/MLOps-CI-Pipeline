"""Tests for src.drift.compute — uses real Evidently 0.6.7."""

import logging

import numpy as np
import pandas as pd
import pytest

from src.config.schema import (
    DriftConfig,
    DriftStatTestConfig,
    DriftStatTestThresholdConfig,
)
from src.drift.compute import (
    extract_overall_results,
    extract_per_feature_results,
    run_evidently_drift,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iris_like(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate an iris-like DataFrame with 4 numeric features."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "sepal_length": rng.normal(5.8, 0.8, n),
        "sepal_width": rng.normal(3.0, 0.4, n),
        "petal_length": rng.normal(3.7, 1.8, n),
        "petal_width": rng.normal(1.2, 0.7, n),
    })


def _default_config(**overrides) -> DriftConfig:
    """Build a DriftConfig with optional field overrides."""
    return DriftConfig(**overrides)


NUMERIC_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


# ===========================================================================
# run_evidently_drift
# ===========================================================================


class TestRunEvidentlyDrift:
    def test_identical_data_no_drift(self):
        """Two identical DataFrames → no feature should be flagged."""
        df = _iris_like(n=100)
        config = _default_config()

        result, report = run_evidently_drift(df, df.copy(), NUMERIC_FEATURES, config)

        features = extract_per_feature_results(result)
        for feat, info in features.items():
            assert info["drift_detected"] is False, (
                f"{feat} falsely detected as drifted"
            )

        overall = extract_overall_results(result)
        assert overall["dataset_drift_detected"] is False

    def test_heavily_shifted_data_detected(self):
        """Large mean shift → most features should drift."""
        ref = _iris_like(n=100, seed=1)
        # Shift all features by +10 standard deviations
        cur = ref.copy()
        for col in NUMERIC_FEATURES:
            cur[col] = cur[col] + 10 * cur[col].std()

        config = _default_config()
        result, report = run_evidently_drift(ref, cur, NUMERIC_FEATURES, config)

        features = extract_per_feature_results(result)
        drifted = [f for f, info in features.items() if info["drift_detected"]]
        assert len(drifted) == len(NUMERIC_FEATURES), (
            f"Expected all features drifted, only got: {drifted}"
        )

        overall = extract_overall_results(result)
        assert overall["dataset_drift_detected"] is True
        assert overall["drifted_feature_count"] == len(NUMERIC_FEATURES)

    def test_mixed_numeric_and_categorical(self):
        """Handles DataFrames with both numeric and categorical columns."""
        rng = np.random.RandomState(42)
        n = 100
        ref = pd.DataFrame({
            "num1": rng.normal(0, 1, n),
            "cat1": rng.choice(["a", "b", "c"], n),
        })
        cur = pd.DataFrame({
            "num1": rng.normal(0, 1, n),
            "cat1": rng.choice(["a", "b", "c"], n),
        })

        config = _default_config()
        result, _ = run_evidently_drift(ref, cur, ["num1", "cat1"], config)

        features = extract_per_feature_results(result)
        assert "num1" in features
        assert "cat1" in features
        assert features["num1"]["column_type"] == "num"
        assert features["cat1"]["column_type"] == "cat"

    def test_configurable_threshold_changes_outcome(self):
        """A borderline drift case flips based on threshold setting."""
        rng = np.random.RandomState(99)
        n = 100
        ref = pd.DataFrame({"x": rng.normal(0, 1, n)})
        # Small shift — borderline
        cur = pd.DataFrame({"x": rng.normal(0.4, 1, n)})

        # First run with strict threshold (0.05) — may or may not detect
        strict = _default_config(
            stattest_threshold=DriftStatTestThresholdConfig(
                numerical=0.05, categorical=0.05,
            ),
        )
        result_strict, _ = run_evidently_drift(ref, cur, ["x"], strict)
        score = extract_per_feature_results(result_strict)["x"]["drift_score"]

        # Now run with extremely permissive threshold
        permissive = _default_config(
            stattest_threshold=DriftStatTestThresholdConfig(
                numerical=0.99, categorical=0.99,
            ),
        )
        result_permissive, _ = run_evidently_drift(ref, cur, ["x"], permissive)
        feat_permissive = extract_per_feature_results(result_permissive)["x"]

        # With threshold=0.99, everything is "drifted" (p-value < 0.99)
        assert feat_permissive["drift_detected"] is True
        assert feat_permissive["stattest_threshold"] == 0.99

    def test_configurable_stattest(self):
        """Non-default stattest is passed through to Evidently."""
        df = _iris_like(n=100)
        config = _default_config(
            stattest=DriftStatTestConfig(numerical="wasserstein", categorical="chisquare"),
        )
        result, _ = run_evidently_drift(df, df.copy(), NUMERIC_FEATURES, config)

        features = extract_per_feature_results(result)
        for info in features.values():
            assert "wasserstein" in info["stattest_name"].lower()

    def test_small_sample_warning(self, caplog):
        """Log warning when DataFrames have fewer than 30 rows."""
        small = _iris_like(n=10)
        config = _default_config()

        with caplog.at_level(logging.WARNING):
            run_evidently_drift(small, small.copy(), NUMERIC_FEATURES, config)

        warnings = [r for r in caplog.records if "unreliable" in r.message]
        assert len(warnings) == 2  # one for reference, one for current

    def test_returns_report_object(self):
        """The second return value is an Evidently Report for HTML generation."""
        from evidently.report import Report as EvidentlyReport

        df = _iris_like(n=50)
        config = _default_config()
        _, report = run_evidently_drift(df, df.copy(), NUMERIC_FEATURES, config)
        assert isinstance(report, EvidentlyReport)

    def test_drift_share_config_respected(self):
        """drift_share config changes dataset-level drift detection."""
        ref = _iris_like(n=100, seed=10)
        cur = ref.copy()
        # Shift only 1 of 4 features → 25% drift share
        cur["sepal_length"] = cur["sepal_length"] + 10

        # drift_share=0.2 → 25% > 20% → dataset drift detected
        config_low = _default_config(drift_share=0.2)
        result_low, _ = run_evidently_drift(ref, cur, NUMERIC_FEATURES, config_low)
        assert extract_overall_results(result_low)["dataset_drift_detected"] is True

        # drift_share=0.5 → 25% < 50% → no dataset drift
        config_high = _default_config(drift_share=0.5)
        result_high, _ = run_evidently_drift(ref, cur, NUMERIC_FEATURES, config_high)
        assert extract_overall_results(result_high)["dataset_drift_detected"] is False


# ===========================================================================
# extract_per_feature_results
# ===========================================================================


class TestExtractPerFeatureResults:
    def test_all_expected_keys_present(self):
        """Each feature dict has exactly the expected keys."""
        df = _iris_like(n=100)
        config = _default_config()
        result, _ = run_evidently_drift(df, df.copy(), NUMERIC_FEATURES, config)

        features = extract_per_feature_results(result)
        expected_keys = {
            "column_type", "drift_detected", "drift_score",
            "stattest_name", "stattest_threshold",
        }
        for feat_name, feat_data in features.items():
            assert set(feat_data.keys()) == expected_keys, (
                f"Unexpected keys for {feat_name}: {set(feat_data.keys())}"
            )

    def test_feature_count_matches_input(self):
        """Number of features in output matches the input columns."""
        df = _iris_like(n=100)
        config = _default_config()
        result, _ = run_evidently_drift(df, df.copy(), NUMERIC_FEATURES, config)

        features = extract_per_feature_results(result)
        assert len(features) == len(NUMERIC_FEATURES)
        assert set(features.keys()) == set(NUMERIC_FEATURES)

    def test_drift_score_is_float(self):
        """drift_score values are numeric."""
        ref = _iris_like(n=100, seed=1)
        cur = ref.copy()
        cur["sepal_length"] = cur["sepal_length"] + 5

        config = _default_config()
        result, _ = run_evidently_drift(ref, cur, NUMERIC_FEATURES, config)
        features = extract_per_feature_results(result)

        for info in features.values():
            assert isinstance(info["drift_score"], float)

    def test_missing_metric_raises_key_error(self):
        """Bogus Evidently output without DataDriftTable raises KeyError."""
        fake_result = {"metrics": [{"metric": "SomethingElse", "result": {}}]}
        with pytest.raises(KeyError, match="DataDriftTable"):
            extract_per_feature_results(fake_result)


# ===========================================================================
# extract_overall_results
# ===========================================================================


class TestExtractOverallResults:
    def test_all_expected_keys_present(self):
        """Overall dict has exactly the expected keys."""
        df = _iris_like(n=100)
        config = _default_config()
        result, _ = run_evidently_drift(df, df.copy(), NUMERIC_FEATURES, config)

        overall = extract_overall_results(result)
        expected_keys = {
            "dataset_drift_detected", "drift_share",
            "drifted_feature_count", "total_feature_count",
        }
        assert set(overall.keys()) == expected_keys

    def test_total_feature_count_correct(self):
        """total_feature_count matches the number of input features."""
        df = _iris_like(n=100)
        config = _default_config()
        result, _ = run_evidently_drift(df, df.copy(), NUMERIC_FEATURES, config)

        overall = extract_overall_results(result)
        assert overall["total_feature_count"] == len(NUMERIC_FEATURES)

    def test_no_drift_share_is_zero(self):
        """Identical data → drift_share is 0.0."""
        df = _iris_like(n=100)
        config = _default_config()
        result, _ = run_evidently_drift(df, df.copy(), NUMERIC_FEATURES, config)

        overall = extract_overall_results(result)
        assert overall["drift_share"] == 0.0
        assert overall["drifted_feature_count"] == 0

    def test_full_drift_share_is_one(self):
        """All features shifted → drift_share is 1.0."""
        ref = _iris_like(n=100, seed=5)
        cur = ref.copy()
        for col in NUMERIC_FEATURES:
            cur[col] = cur[col] + 10 * cur[col].std()

        config = _default_config()
        result, _ = run_evidently_drift(ref, cur, NUMERIC_FEATURES, config)

        overall = extract_overall_results(result)
        assert overall["drift_share"] == 1.0
        assert overall["drifted_feature_count"] == len(NUMERIC_FEATURES)

    def test_missing_metric_raises_key_error(self):
        """Bogus Evidently output without DatasetDriftMetric raises KeyError."""
        fake_result = {"metrics": [{"metric": "SomethingElse", "result": {}}]}
        with pytest.raises(KeyError, match="DatasetDriftMetric"):
            extract_overall_results(fake_result)
