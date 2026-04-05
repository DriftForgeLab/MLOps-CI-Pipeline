"""Dataset alignment for drift detection.

Loads the reference (training) dataset and validates that the current dataset's
schema is compatible before drift computation.
"""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PREPROCESSED_SUBDIR = "preprocessed"
FEATURE_MAP_FILENAME = "feature_map.json"


def load_reference_data(
    dataset_name: str,
    version_id: str,
    reference_source: str,
    processed_dir: str | Path,
) -> tuple[pd.DataFrame, dict]:
    """Load the preprocessed reference split and its feature map.

    Args:
        dataset_name:    Name of the dataset (e.g. "iris").
        version_id:      Content-addressed version hash.
        reference_source: Which split to use as baseline (e.g. "train").
        processed_dir:   Root processed data directory.

    Returns:
        Tuple of (reference_df, feature_map_dict).

    Raises:
        FileNotFoundError: If the split CSV or feature_map.json is missing.
    """
    preprocessed_dir = (
        Path(processed_dir) / dataset_name / version_id / PREPROCESSED_SUBDIR
    )

    feature_map_path = preprocessed_dir / FEATURE_MAP_FILENAME
    if not feature_map_path.exists():
        raise FileNotFoundError(
            f"feature_map.json not found at '{feature_map_path}'. "
            "Run the preprocessing stage before drift analysis."
        )

    with open(feature_map_path) as f:
        feature_map: dict = json.load(f)

    csv_path = preprocessed_dir / f"{reference_source}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Reference split '{reference_source}.csv' not found at '{csv_path}'. "
            "Run the preprocessing stage before drift analysis."
        )

    reference_df = pd.read_csv(csv_path)
    logger.info(
        "Loaded reference data: %s (%d rows, %d columns)",
        csv_path,
        len(reference_df),
        len(reference_df.columns),
    )
    return reference_df, feature_map


def validate_feature_alignment(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_map: dict,
) -> list[str]:
    """Check that current data is schema-compatible with the reference.

    Validates that all output features from the feature map exist in both
    DataFrames and that numeric/categorical dtypes are consistent.

    Args:
        reference_df: The reference (training) DataFrame.
        current_df:   The incoming DataFrame to compare against.
        feature_map:  Dict with at least ``output_features`` and ``target``.

    Returns:
        List of error strings. Empty list means alignment is valid.
    """
    errors: list[str] = []
    output_features: list[str] = feature_map.get("output_features", [])

    if not output_features:
        errors.append("feature_map contains no output_features.")
        return errors

    # --- Check that required features exist in both DataFrames ---
    ref_cols = set(reference_df.columns)
    cur_cols = set(current_df.columns)

    missing_in_ref = sorted(set(output_features) - ref_cols)
    if missing_in_ref:
        errors.append(
            f"Features missing in reference data: {missing_in_ref}"
        )

    missing_in_cur = sorted(set(output_features) - cur_cols)
    if missing_in_cur:
        errors.append(
            f"Features missing in current data: {missing_in_cur}"
        )

    # Warn about extra columns in current (not errors)
    expected = set(output_features) | {feature_map.get("target", "")}
    extra_in_cur = sorted(cur_cols - expected - {""})
    if extra_in_cur:
        logger.warning(
            "Extra columns in current data will be ignored for drift: %s",
            extra_in_cur,
        )

    # If columns are missing we can't do dtype checks, return early
    if errors:
        return errors

    # --- Check dtype consistency for shared features ---
    for col in output_features:
        ref_numeric = pd.api.types.is_numeric_dtype(reference_df[col])
        cur_numeric = pd.api.types.is_numeric_dtype(current_df[col])

        if ref_numeric and not cur_numeric:
            errors.append(
                f"Type mismatch for '{col}': numeric in reference, "
                f"non-numeric ({current_df[col].dtype}) in current."
            )
        elif not ref_numeric and cur_numeric:
            errors.append(
                f"Type mismatch for '{col}': categorical in reference, "
                f"numeric ({current_df[col].dtype}) in current."
            )

    # Warn if current DataFrame has zero rows
    if len(current_df) == 0:
        logger.warning(
            "Current DataFrame has 0 rows — drift computation will not "
            "produce meaningful results."
        )

    return errors
