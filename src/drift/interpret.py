"""Drift interpretation: severity classification.

Converts raw Evidently drift scores into severity labels. Assembles
the complete standard drift result JSON consumed by downstream layers.
Severity alone is the decision input — no recommendation vocabulary.

All functions are pure — no I/O, no randomness, no external state.
"""

from datetime import datetime, timezone

from src.config.schema import (
    DriftConfig,
    DriftFeatureSeverityConfig,
    DriftSeverityConfig,
)

SCHEMA_VERSION = "1.0.0"
DRIFT_TYPE_TABULAR = "tabular"

# Ordinal ranking used to compare severity labels against configured minima.
_SEVERITY_ORD: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def classify_feature_severity(
    drift_score: float,
    drift_detected: bool,
    feature_severity_config: DriftFeatureSeverityConfig,
) -> str:
    """Classify a single feature's drift severity from its p-value.

    For KS / chi-square tests, ``drift_score`` is the p-value — smaller
    values mean stronger evidence of drift.

    Args:
        drift_score:             p-value produced by the statistical test.
        drift_detected:          Whether Evidently flagged the feature as drifted.
        feature_severity_config: Thresholds for ``high_below`` and ``medium_below``.

    Returns:
        ``"low"``, ``"medium"``, or ``"high"``.
    """
    if not drift_detected:
        return "low"
    if drift_score < feature_severity_config.high_below:
        return "high"
    if drift_score < feature_severity_config.medium_below:
        return "medium"
    return "low"


def classify_overall_severity(
    drift_share: float,
    severity_config: DriftSeverityConfig,
) -> str:
    """Classify dataset-level drift severity from the drift share.

    Boundaries are inclusive on the upper edge (``<=``) so that
    ``drift_share == low_max`` is still ``"low"`` and
    ``drift_share == medium_max`` is still ``"medium"``.

    Args:
        drift_share:     Fraction of features that drifted (0.0–1.0).
        severity_config: Band thresholds ``low_max`` and ``medium_max``.

    Returns:
        ``"low"``, ``"medium"``, or ``"high"``.
    """
    if drift_share <= severity_config.low_max:
        return "low"
    if drift_share <= severity_config.medium_max:
        return "medium"
    return "high"


def build_drift_result(
    overall_raw: dict,
    features_raw: dict,
    drift_config: DriftConfig,
    reference_info: dict,
    current_info: dict,
    pipeline_execution_id: str,
    dataset_version_id: str,
    task_type: str,
) -> dict:
    """Assemble the complete standard drift result JSON.

    Performs per-feature and overall severity classification, derives the
    recommendation, and stamps metadata. The returned dict conforms to the
    standard schema.

    Args:
        overall_raw:           Output of ``extract_overall_results``.
        features_raw:          Output of ``extract_per_feature_results``.
        drift_config:          Drift configuration dataclass.
        reference_info:        Reference dataset metadata
                               (``source``, ``path``, ``row_count``, ``feature_count``).
        current_info:          Current dataset metadata (same shape).
        pipeline_execution_id: UUID linking to the pipeline run.
        dataset_version_id:    Content-addressed dataset version hash.
        task_type:             ``"classification"`` or ``"regression"``.

    Returns:
        Dict conforming to the standard drift result schema. ``artifacts``
        is an empty dict (populated later by the reporting layer).
    """
    # --- Per-feature severity classification ---
    features_out: dict = {}
    severity_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}
    drifted_features: list[str] = []

    for feat_name, feat_data in features_raw.items():
        severity = classify_feature_severity(
            drift_score=feat_data["drift_score"],
            drift_detected=feat_data["drift_detected"],
            feature_severity_config=drift_config.feature_severity,
        )
        severity_counts[severity] += 1
        if feat_data["drift_detected"]:
            drifted_features.append(feat_name)

        features_out[feat_name] = {
            "column_type": feat_data["column_type"],
            "drift_detected": feat_data["drift_detected"],
            "drift_score": feat_data["drift_score"],
            "stattest_name": feat_data["stattest_name"],
            "stattest_threshold": feat_data["stattest_threshold"],
            "severity": severity,
        }

    # --- Overall severity ---
    overall_severity = classify_overall_severity(
        drift_share=overall_raw["drift_share"],
        severity_config=drift_config.severity,
    )

    overall_out = {
        "dataset_drift_detected": overall_raw["dataset_drift_detected"],
        "drift_share": overall_raw["drift_share"],
        "drifted_feature_count": overall_raw["drifted_feature_count"],
        "total_feature_count": overall_raw["total_feature_count"],
        "severity": overall_severity,
    }

    config_snapshot = {
        "stattest_numerical": drift_config.stattest.numerical,
        "stattest_categorical": drift_config.stattest.categorical,
        "stattest_threshold_numerical": drift_config.stattest_threshold.numerical,
        "stattest_threshold_categorical": drift_config.stattest_threshold.categorical,
        "drift_share_threshold": drift_config.drift_share,
        "severity_low_max": drift_config.severity.low_max,
        "severity_medium_max": drift_config.severity.medium_max,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "drift_type": DRIFT_TYPE_TABULAR,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_execution_id": pipeline_execution_id,
        "dataset_version_id": dataset_version_id,
        "task_type": task_type,
        "reference_dataset": dict(reference_info),
        "current_dataset": dict(current_info),
        "overall": overall_out,
        "features": features_out,
        "artifacts": {},
        "config_snapshot": config_snapshot,
    }
