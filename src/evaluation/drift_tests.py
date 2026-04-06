"""Drift analysis orchestration — data in, result out.

Pure orchestration function that wires alignment, compute, and interpret
layers together. No file I/O, no MLflow calls, no interactive code.
The caller (_drift_stage in steps.py) handles persistence and logging.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from evidently.report import Report

from src.config.schema import DriftConfig, PipelineConfig
from src.drift.alignment import (
    load_reference_data,
    validate_feature_alignment,
    PREPROCESSED_SUBDIR,
)
from src.drift.compute import (
    run_evidently_drift,
    extract_per_feature_results,
    extract_overall_results,
)
from src.drift.interpret import build_drift_result

logger = logging.getLogger(__name__)


def run_drift_analysis(
    config: PipelineConfig,
    version_id: str,
    drift_config: DriftConfig,
) -> tuple[dict, Report]:
    """Run end-to-end drift analysis: align, compute, interpret.

    Loads reference and current data, validates feature alignment,
    runs Evidently drift detection, and assembles the standard drift
    result dict with severity classification.

    Args:
        config:       Pipeline configuration (dataset name, paths, task type).
        version_id:   Content-addressed dataset version hash.
        drift_config: Drift-specific configuration (thresholds, tests, severity).

    Returns:
        Tuple of (drift_result_dict, evidently_report_object).

    Raises:
        FileNotFoundError: If reference data or feature map is missing.
        ValueError:        If feature alignment fails.
    """
    # 1. Load reference data and feature map
    reference_df, feature_map = load_reference_data(
        dataset_name=config.dataset,
        version_id=version_id,
        reference_source=drift_config.reference_source,
        processed_dir=config.data.processed,
    )

    # 2. Load current data (validation split from the same version)
    preprocessed_dir = (
        Path(config.data.processed)
        / config.dataset
        / version_id
        / PREPROCESSED_SUBDIR
    )
    current_path = preprocessed_dir / "val.csv"
    if not current_path.exists():
        raise FileNotFoundError(
            f"Validation split not found at '{current_path}'. "
            "Run the preprocessing stage before drift analysis."
        )
    current_df = pd.read_csv(current_path)

    # 3. Validate feature alignment
    output_features: list[str] = feature_map.get("output_features", [])
    errors = validate_feature_alignment(reference_df, current_df, feature_map)
    if errors:
        raise ValueError(
            "Feature alignment failed:\n  - " + "\n  - ".join(errors)
        )

    # 4. Determine feature columns (exclude target)
    feature_columns = [
        col for col in output_features
        if col in reference_df.columns and col in current_df.columns
    ]

    # 5. Run Evidently drift detection
    evidently_result, evidently_report = run_evidently_drift(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=feature_columns,
        drift_config=drift_config,
    )

    # 6. Extract raw results
    per_feature_raw = extract_per_feature_results(evidently_result)
    overall_raw = extract_overall_results(evidently_result)

    # 7. Build standard drift result with severity classification
    drift_result = build_drift_result(
        overall_raw=overall_raw,
        features_raw=per_feature_raw,
        drift_config=drift_config,
        reference_info={
            "source": drift_config.reference_source,
            "path": str(preprocessed_dir / f"{drift_config.reference_source}.csv"),
            "row_count": len(reference_df),
            "feature_count": len(feature_columns),
        },
        current_info={
            "source": "val",
            "path": str(current_path),
            "row_count": len(current_df),
            "feature_count": len(feature_columns),
        },
        pipeline_execution_id=version_id,
        dataset_version_id=version_id,
        task_type=config.task_type,
    )

    logger.info(
        "Drift analysis complete: severity=%s, drifted=%d/%d",
        drift_result["overall"]["severity"],
        drift_result["overall"]["drifted_feature_count"],
        drift_result["overall"]["total_feature_count"],
    )

    return drift_result, evidently_report
