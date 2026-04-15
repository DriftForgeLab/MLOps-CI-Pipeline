"""Per-feature drift computation using Evidently.

Wraps Evidently's DataDriftPreset to produce raw statistical results.
"""

import logging

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.config.schema import DriftConfig

logger = logging.getLogger(__name__)

_SMALL_SAMPLE_THRESHOLD = 10


def run_evidently_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    drift_config: DriftConfig,
) -> tuple[dict, Report]:
    """Run Evidently DataDriftPreset on reference vs current data.

    Args:
        reference_df:    Reference (training) DataFrame.
        current_df:      Incoming DataFrame to compare.
        feature_columns: Feature columns to analyse (excludes target).
        drift_config:    Drift configuration dataclass.

    Returns:
        Tuple of (evidently_result_dict, evidently_report_object).
        The dict is the parsed ``report.as_dict()`` for downstream extraction.
        The Report object is kept for HTML generation.
    """
    for label, df in [("reference", reference_df), ("current", current_df)]:
        if len(df) < _SMALL_SAMPLE_THRESHOLD:
            logger.warning(
                "%s DataFrame has %d rows (< %d) — "
                "statistical tests may be unreliable.",
                label,
                len(df),
                _SMALL_SAMPLE_THRESHOLD,
            )

    numerical = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(reference_df[c])
    ]
    categorical = [
        c for c in feature_columns
        if not pd.api.types.is_numeric_dtype(reference_df[c])
    ]

    mapping = ColumnMapping()
    mapping.numerical_features = numerical
    mapping.categorical_features = categorical

    preset = DataDriftPreset(
        num_stattest=drift_config.stattest.numerical,
        cat_stattest=drift_config.stattest.categorical,
        num_stattest_threshold=drift_config.stattest_threshold.numerical,
        cat_stattest_threshold=drift_config.stattest_threshold.categorical,
        drift_share=drift_config.drift_share,
    )

    report = Report(metrics=[preset])
    report.run(
        reference_data=reference_df[feature_columns],
        current_data=current_df[feature_columns],
        column_mapping=mapping,
    )

    result_dict = report.as_dict()
    logger.info(
        "Evidently drift report computed: %d features analysed.",
        len(feature_columns),
    )
    return result_dict, report


def extract_per_feature_results(evidently_result: dict) -> dict:
    """Extract per-feature drift results from Evidently's as_dict() output.

    Navigates the Evidently output structure to pull per-column results
    from the DataDriftTable metric.

    Args:
        evidently_result: The dict returned by ``Report.as_dict()``.

    Returns:
        Dict keyed by feature name, each value containing:
        ``column_type``, ``drift_detected``, ``drift_score``,
        ``stattest_name``, ``stattest_threshold``.
    """
    drift_table = _find_metric(evidently_result, "DataDriftTable")
    drift_by_columns = drift_table["drift_by_columns"]

    features = {}
    for col_name, col_data in drift_by_columns.items():
        features[col_name] = {
            "column_type": col_data["column_type"],
            "drift_detected": col_data["drift_detected"],
            "drift_score": col_data["drift_score"],
            "stattest_name": col_data["stattest_name"],
            "stattest_threshold": col_data["stattest_threshold"],
        }

    return features


def extract_overall_results(evidently_result: dict) -> dict:
    """Extract dataset-level drift results from Evidently's as_dict() output.

    Uses the DatasetDriftMetric for authoritative overall results.

    Args:
        evidently_result: The dict returned by ``Report.as_dict()``.

    Returns:
        Dict with ``dataset_drift_detected``, ``drift_share``,
        ``drifted_feature_count``, ``total_feature_count``.
    """
    dataset_metric = _find_metric(evidently_result, "DatasetDriftMetric")

    return {
        "dataset_drift_detected": dataset_metric["dataset_drift"],
        "drift_share": dataset_metric["share_of_drifted_columns"],
        "drifted_feature_count": dataset_metric["number_of_drifted_columns"],
        "total_feature_count": dataset_metric["number_of_columns"],
    }


def _find_metric(evidently_result: dict, metric_name: str) -> dict:
    """Find a metric result by name in the Evidently output."""
    for entry in evidently_result["metrics"]:
        if entry["metric"] == metric_name:
            return entry["result"]
    raise KeyError(
        f"Metric '{metric_name}' not found in Evidently output. "
        f"Available: {[m['metric'] for m in evidently_result['metrics']]}"
    )
