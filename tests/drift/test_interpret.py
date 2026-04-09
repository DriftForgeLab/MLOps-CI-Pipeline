"""Tests for src.drift.interpret — pure-function severity classification logic."""

import pytest

from src.config.schema import (
    DriftConfig,
    DriftFeatureSeverityConfig,
    DriftSeverityConfig,
)
from src.drift.interpret import (
    DRIFT_TYPE_TABULAR,
    SCHEMA_VERSION,
    build_drift_result,
    classify_feature_severity,
    classify_overall_severity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_info(rows: int = 100, features: int = 4) -> dict:
    return {
        "source": "train",
        "path": "data/processed/iris/v1/preprocessed/train.csv",
        "row_count": rows,
        "feature_count": features,
    }


def _cur_info(rows: int = 50, features: int = 4) -> dict:
    return {
        "source": "val",
        "path": "data/processed/iris/v1/preprocessed/val.csv",
        "row_count": rows,
        "feature_count": features,
    }


def _feature(
    *,
    drift_detected: bool,
    drift_score: float,
    column_type: str = "num",
    stattest_name: str = "ks",
    stattest_threshold: float = 0.05,
) -> dict:
    return {
        "column_type": column_type,
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "stattest_name": stattest_name,
        "stattest_threshold": stattest_threshold,
    }


def _overall_raw(
    *,
    drift_share: float,
    drifted_feature_count: int,
    total_feature_count: int,
    dataset_drift_detected: bool,
) -> dict:
    return {
        "dataset_drift_detected": dataset_drift_detected,
        "drift_share": drift_share,
        "drifted_feature_count": drifted_feature_count,
        "total_feature_count": total_feature_count,
    }


# ===========================================================================
# classify_feature_severity
# ===========================================================================


class TestClassifyFeatureSeverity:
    def test_not_drifted_is_low(self):
        """drift_detected=False always returns 'low' regardless of score."""
        cfg = DriftFeatureSeverityConfig()
        assert classify_feature_severity(0.0, False, cfg) == "low"
        assert classify_feature_severity(0.5, False, cfg) == "low"
        assert classify_feature_severity(1.0, False, cfg) == "low"

    def test_very_small_pvalue_is_high(self):
        """p-value below high_below → 'high'."""
        cfg = DriftFeatureSeverityConfig(high_below=0.001, medium_below=0.01)
        assert classify_feature_severity(0.0005, True, cfg) == "high"

    def test_medium_pvalue(self):
        """p-value below medium_below but >= high_below → 'medium'."""
        cfg = DriftFeatureSeverityConfig(high_below=0.001, medium_below=0.01)
        assert classify_feature_severity(0.005, True, cfg) == "medium"

    def test_drifted_but_large_pvalue_is_low(self):
        """drift_detected=True but p-value >= medium_below → 'low'."""
        cfg = DriftFeatureSeverityConfig(high_below=0.001, medium_below=0.01)
        assert classify_feature_severity(0.03, True, cfg) == "low"

    def test_boundary_exactly_high_below(self):
        """p-value exactly equal to high_below is NOT high (strict <)."""
        cfg = DriftFeatureSeverityConfig(high_below=0.001, medium_below=0.01)
        # 0.001 is not < 0.001, so falls through to medium check (0.001 < 0.01) → medium
        assert classify_feature_severity(0.001, True, cfg) == "medium"

    def test_boundary_exactly_medium_below(self):
        """p-value exactly equal to medium_below is NOT medium (strict <) → low."""
        cfg = DriftFeatureSeverityConfig(high_below=0.001, medium_below=0.01)
        assert classify_feature_severity(0.01, True, cfg) == "low"

    def test_zero_pvalue_is_high(self):
        """p-value of exactly 0 → 'high'."""
        cfg = DriftFeatureSeverityConfig()
        assert classify_feature_severity(0.0, True, cfg) == "high"


# ===========================================================================
# classify_overall_severity
# ===========================================================================


class TestClassifyOverallSeverity:
    def test_zero_share_is_low(self):
        assert classify_overall_severity(0.0, DriftSeverityConfig()) == "low"

    def test_full_share_is_high(self):
        assert classify_overall_severity(1.0, DriftSeverityConfig()) == "high"

    def test_middle_band_is_medium(self):
        cfg = DriftSeverityConfig(low_max=0.25, medium_max=0.50)
        assert classify_overall_severity(0.40, cfg) == "medium"

    def test_high_band(self):
        cfg = DriftSeverityConfig(low_max=0.25, medium_max=0.50)
        assert classify_overall_severity(0.75, cfg) == "high"

    def test_boundary_exactly_low_max(self):
        """drift_share == low_max → 'low' (inclusive upper edge)."""
        cfg = DriftSeverityConfig(low_max=0.25, medium_max=0.50)
        assert classify_overall_severity(0.25, cfg) == "low"

    def test_boundary_exactly_medium_max(self):
        """drift_share == medium_max → 'medium' (inclusive upper edge)."""
        cfg = DriftSeverityConfig(low_max=0.25, medium_max=0.50)
        assert classify_overall_severity(0.50, cfg) == "medium"

    def test_just_above_low_max(self):
        cfg = DriftSeverityConfig(low_max=0.25, medium_max=0.50)
        assert classify_overall_severity(0.2501, cfg) == "medium"

    def test_just_above_medium_max(self):
        cfg = DriftSeverityConfig(low_max=0.25, medium_max=0.50)
        assert classify_overall_severity(0.5001, cfg) == "high"


# ===========================================================================
# build_drift_result
# ===========================================================================


class TestBuildDriftResult:
    def _call(self, overall_raw, features_raw, drift_config=None, task_type="classification"):
        return build_drift_result(
            overall_raw=overall_raw,
            features_raw=features_raw,
            drift_config=drift_config or DriftConfig(),
            reference_info=_ref_info(),
            current_info=_cur_info(),
            pipeline_execution_id="pipeline-abc-123",
            dataset_version_id="dataset-v1",
            task_type=task_type,
        )

    def test_top_level_schema_fields(self):
        """Result contains every top-level schema field."""
        features_raw = {
            "f1": _feature(drift_detected=False, drift_score=0.8),
        }
        overall_raw = _overall_raw(
            drift_share=0.0,
            drifted_feature_count=0,
            total_feature_count=1,
            dataset_drift_detected=False,
        )
        result = self._call(overall_raw, features_raw)

        expected = {
            "schema_version", "drift_type", "method", "generated_at",
            "pipeline_execution_id", "dataset_version_id", "task_type",
            "reference_dataset", "current_dataset",
            "overall", "features", "artifacts", "config_snapshot",
        }
        assert set(result.keys()) == expected

    def test_schema_version_and_drift_type(self):
        features_raw = {"f1": _feature(drift_detected=False, drift_score=0.8)}
        overall_raw = _overall_raw(
            drift_share=0.0,
            drifted_feature_count=0,
            total_feature_count=1,
            dataset_drift_detected=False,
        )
        result = self._call(overall_raw, features_raw)
        assert result["schema_version"] == SCHEMA_VERSION
        assert result["drift_type"] == DRIFT_TYPE_TABULAR

    def test_artifacts_empty_dict(self):
        result = self._call(
            overall_raw=_overall_raw(
                drift_share=0.0, drifted_feature_count=0,
                total_feature_count=1, dataset_drift_detected=False,
            ),
            features_raw={"f1": _feature(drift_detected=False, drift_score=0.8)},
        )
        assert result["artifacts"] == {}

    def test_generated_at_is_iso_utc(self):
        result = self._call(
            overall_raw=_overall_raw(
                drift_share=0.0, drifted_feature_count=0,
                total_feature_count=1, dataset_drift_detected=False,
            ),
            features_raw={"f1": _feature(drift_detected=False, drift_score=0.8)},
        )
        ts = result["generated_at"]
        assert isinstance(ts, str)
        # ISO 8601 w/ timezone suffix (either +00:00 or Z-style from fromisoformat)
        assert "T" in ts
        assert "+" in ts or ts.endswith("+00:00")

    def test_per_feature_severity_classified(self):
        """Each feature in the result has a 'severity' key added."""
        features_raw = {
            "f_high": _feature(drift_detected=True, drift_score=0.0005),
            "f_medium": _feature(drift_detected=True, drift_score=0.005),
            "f_low_drift": _feature(drift_detected=True, drift_score=0.04),
            "f_none": _feature(drift_detected=False, drift_score=0.8),
        }
        overall_raw = _overall_raw(
            drift_share=0.75,
            drifted_feature_count=3,
            total_feature_count=4,
            dataset_drift_detected=True,
        )
        result = self._call(overall_raw, features_raw)

        assert result["features"]["f_high"]["severity"] == "high"
        assert result["features"]["f_medium"]["severity"] == "medium"
        assert result["features"]["f_low_drift"]["severity"] == "low"
        assert result["features"]["f_none"]["severity"] == "low"

    def test_per_feature_preserves_raw_fields(self):
        features_raw = {
            "f1": _feature(
                drift_detected=True,
                drift_score=0.0005,
                column_type="num",
                stattest_name="ks",
                stattest_threshold=0.05,
            ),
        }
        overall_raw = _overall_raw(
            drift_share=1.0,
            drifted_feature_count=1,
            total_feature_count=1,
            dataset_drift_detected=True,
        )
        result = self._call(overall_raw, features_raw)
        feat = result["features"]["f1"]
        assert feat["column_type"] == "num"
        assert feat["drift_detected"] is True
        assert feat["drift_score"] == 0.0005
        assert feat["stattest_name"] == "ks"
        assert feat["stattest_threshold"] == 0.05
        assert feat["severity"] == "high"

    def test_overall_has_severity_field(self):
        features_raw = {
            "f1": _feature(drift_detected=True, drift_score=0.0001),
            "f2": _feature(drift_detected=True, drift_score=0.0001),
            "f3": _feature(drift_detected=True, drift_score=0.0001),
            "f4": _feature(drift_detected=True, drift_score=0.0001),
        }
        overall_raw = _overall_raw(
            drift_share=1.0,
            drifted_feature_count=4,
            total_feature_count=4,
            dataset_drift_detected=True,
        )
        result = self._call(overall_raw, features_raw)
        assert result["overall"]["severity"] == "high"
        assert result["overall"]["drift_share"] == 1.0
        assert result["overall"]["drifted_feature_count"] == 4
        assert result["overall"]["dataset_drift_detected"] is True

    def test_all_drifted_is_high_severity(self):
        """All features drifted with high scores → severity=high."""
        features_raw = {
            f"f{i}": _feature(drift_detected=True, drift_score=0.0001)
            for i in range(4)
        }
        overall_raw = _overall_raw(
            drift_share=1.0,
            drifted_feature_count=4,
            total_feature_count=4,
            dataset_drift_detected=True,
        )
        result = self._call(overall_raw, features_raw)
        assert result["overall"]["severity"] == "high"

    def test_no_drift_is_low_severity(self):
        features_raw = {
            f"f{i}": _feature(drift_detected=False, drift_score=0.8)
            for i in range(4)
        }
        overall_raw = _overall_raw(
            drift_share=0.0,
            drifted_feature_count=0,
            total_feature_count=4,
            dataset_drift_detected=False,
        )
        result = self._call(overall_raw, features_raw)
        assert result["overall"]["severity"] == "low"

    def test_no_recommendation_or_user_decision_fields(self):
        """Schema must not contain recommendation or user_decision fields."""
        features_raw = {
            "a": _feature(drift_detected=True, drift_score=0.0001),
            "b": _feature(drift_detected=False, drift_score=0.5),
        }
        overall_raw = _overall_raw(
            drift_share=0.5,
            drifted_feature_count=1,
            total_feature_count=2,
            dataset_drift_detected=True,
        )
        result = self._call(overall_raw, features_raw)
        assert "recommendation" not in result
        assert "user_decision" not in result

    def test_config_snapshot_contents(self):
        config = DriftConfig(
            severity=DriftSeverityConfig(low_max=0.3, medium_max=0.6),
        )
        features_raw = {"f1": _feature(drift_detected=False, drift_score=0.8)}
        overall_raw = _overall_raw(
            drift_share=0.0,
            drifted_feature_count=0,
            total_feature_count=1,
            dataset_drift_detected=False,
        )
        result = self._call(overall_raw, features_raw, drift_config=config)

        snap = result["config_snapshot"]
        assert snap["severity_low_max"] == 0.3
        assert snap["severity_medium_max"] == 0.6
        assert snap["stattest_numerical"] == "ks"
        assert snap["stattest_categorical"] == "chisquare"
        assert snap["drift_share_threshold"] == 0.5
        # blocking threshold is NOT in config_snapshot — it lives on PromotionConfig
        assert "block_on_severity" not in snap

    def test_metadata_passthrough(self):
        features_raw = {"f1": _feature(drift_detected=False, drift_score=0.8)}
        overall_raw = _overall_raw(
            drift_share=0.0,
            drifted_feature_count=0,
            total_feature_count=1,
            dataset_drift_detected=False,
        )
        result = self._call(overall_raw, features_raw, task_type="regression")
        assert result["pipeline_execution_id"] == "pipeline-abc-123"
        assert result["dataset_version_id"] == "dataset-v1"
        assert result["task_type"] == "regression"
        assert result["reference_dataset"]["source"] == "train"
        assert result["current_dataset"]["source"] == "val"

    def test_works_for_regression(self):
        """Interpretation logic is task-type agnostic — regression case works identically."""
        features_raw = {
            "f1": _feature(drift_detected=True, drift_score=0.0001),
            "f2": _feature(drift_detected=True, drift_score=0.0001),
        }
        overall_raw = _overall_raw(
            drift_share=1.0,
            drifted_feature_count=2,
            total_feature_count=2,
            dataset_drift_detected=True,
        )
        result = self._call(overall_raw, features_raw, task_type="regression")
        assert result["task_type"] == "regression"
        assert result["overall"]["severity"] == "high"

    def test_single_feature_fully_drifted(self):
        """Single-feature dataset, drift_share=1.0 works correctly."""
        features_raw = {"only": _feature(drift_detected=True, drift_score=0.0001)}
        overall_raw = _overall_raw(
            drift_share=1.0,
            drifted_feature_count=1,
            total_feature_count=1,
            dataset_drift_detected=True,
        )
        result = self._call(overall_raw, features_raw)
        assert result["overall"]["severity"] == "high"

    def test_determinism_of_non_timestamp_fields(self):
        """Apart from generated_at, two calls with identical inputs produce identical output."""
        features_raw = {
            "a": _feature(drift_detected=True, drift_score=0.0001),
            "b": _feature(drift_detected=False, drift_score=0.5),
        }
        overall_raw = _overall_raw(
            drift_share=0.5,
            drifted_feature_count=1,
            total_feature_count=2,
            dataset_drift_detected=True,
        )
        r1 = self._call(overall_raw, features_raw)
        r2 = self._call(overall_raw, features_raw)

        r1.pop("generated_at")
        r2.pop("generated_at")
        assert r1 == r2

    def test_reference_and_current_info_are_copies(self):
        """Mutating caller dicts must not affect the result."""
        ref = _ref_info()
        cur = _cur_info()
        features_raw = {"f1": _feature(drift_detected=False, drift_score=0.8)}
        overall_raw = _overall_raw(
            drift_share=0.0,
            drifted_feature_count=0,
            total_feature_count=1,
            dataset_drift_detected=False,
        )
        result = build_drift_result(
            overall_raw=overall_raw,
            features_raw=features_raw,
            drift_config=DriftConfig(),
            reference_info=ref,
            current_info=cur,
            pipeline_execution_id="p1",
            dataset_version_id="d1",
            task_type="classification",
        )
        ref["row_count"] = 99999
        cur["row_count"] = 99999
        assert result["reference_dataset"]["row_count"] == 100
        assert result["current_dataset"]["row_count"] == 50
