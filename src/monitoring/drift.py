"""Offline batch drift monitoring for tabular data.

Provides ``monitor_batch()`` to compare a new data batch against a
reference (training) dataset and produce a standard drift result dict.
Also provides ``load_reference_for_model()`` to resolve the reference
data from a model's metadata.

This module is consumed by the ``monitor-drift`` CLI command and can
also be called programmatically.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config.schema import DriftConfig
from src.drift.alignment import (
    load_reference_data,
    validate_feature_alignment,
)
from src.drift.compute import (
    run_evidently_drift,
    extract_per_feature_results,
    extract_overall_results,
)
from src.drift.interpret import build_drift_result, _SEVERITY_ORD
from src.monitoring.reports import print_drift_summary

logger = logging.getLogger(__name__)


def load_reference_for_model(
    model_info: dict,
    processed_dir: str | Path,
    reference_source: str = "train",
) -> tuple[pd.DataFrame, dict]:
    """Load reference data for a registered model.

    Resolves the preprocessed reference split using the model's
    ``dataset_name`` and ``dataset_version_id`` metadata.

    Args:
        model_info:       Dict with ``dataset_name`` and ``dataset_version_id``.
        processed_dir:    Root processed data directory.
        reference_source: Which split to use as baseline (default ``"train"``).

    Returns:
        Tuple of (reference_df, feature_map_dict).

    Raises:
        FileNotFoundError: If the reference CSV or feature_map.json is missing.
    """
    return load_reference_data(
        dataset_name=model_info["dataset_name"],
        version_id=model_info["dataset_version_id"],
        reference_source=reference_source,
        processed_dir=processed_dir,
    )


def monitor_batch(
    batch_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    feature_map: dict,
    drift_config: DriftConfig,
    model_name: str | None = None,
    *,
    interactive: bool = False,
) -> dict | None:
    """Run drift analysis on a new data batch vs reference data.

    Args:
        batch_df:      The incoming batch DataFrame.
        reference_df:  The reference (training) DataFrame.
        feature_map:   Feature map dict with ``output_features``.
        drift_config:  Drift configuration dataclass.
        model_name:    Optional model name for logging context.
        interactive:   If True, prints drift summary to stdout.

    Returns:
        Standard drift result dict, or ``None`` if the batch is
        below ``drift_config.monitoring.min_batch_size``.
    """
    min_size = drift_config.monitoring.min_batch_size
    if len(batch_df) < min_size:
        logger.warning(
            "Batch has %d rows (< min_batch_size=%d) — skipping drift analysis.%s",
            len(batch_df),
            min_size,
            f" Model: {model_name}" if model_name else "",
        )
        return None

    # Validate feature alignment
    errors = validate_feature_alignment(reference_df, batch_df, feature_map)
    if errors:
        raise ValueError(
            "Feature alignment failed:\n  - " + "\n  - ".join(errors)
        )

    # Determine feature columns (exclude target)
    output_features: list[str] = feature_map.get("output_features", [])
    feature_columns = [
        col for col in output_features
        if col in reference_df.columns and col in batch_df.columns
    ]

    # Run Evidently drift detection
    evidently_result, _report = run_evidently_drift(
        reference_df=reference_df,
        current_df=batch_df,
        feature_columns=feature_columns,
        drift_config=drift_config,
    )

    # Extract raw results
    per_feature_raw = extract_per_feature_results(evidently_result)
    overall_raw = extract_overall_results(evidently_result)

    # Build standard drift result
    drift_result = build_drift_result(
        overall_raw=overall_raw,
        features_raw=per_feature_raw,
        drift_config=drift_config,
        reference_info={
            "source": drift_config.reference_source,
            "row_count": len(reference_df),
            "feature_count": len(feature_columns),
        },
        current_info={
            "source": "batch",
            "row_count": len(batch_df),
            "feature_count": len(feature_columns),
        },
        pipeline_execution_id="monitoring",
        dataset_version_id="batch",
        task_type="unknown",
    )

    # Check alert threshold
    overall_severity = drift_result["overall"]["severity"]
    alert_severity = drift_config.monitoring.alert_severity
    if _SEVERITY_ORD.get(overall_severity, 0) >= _SEVERITY_ORD.get(alert_severity, 0):
        drifted = drift_result["overall"]["drifted_feature_count"]
        total = drift_result["overall"]["total_feature_count"]
        logger.warning(
            "Drift alert: severity=%s (>= threshold %s), "
            "drifted features: %d/%d.%s",
            overall_severity,
            alert_severity,
            drifted,
            total,
            f" Model: {model_name}" if model_name else "",
        )

    if interactive:
        print_drift_summary(drift_result)

    return drift_result
