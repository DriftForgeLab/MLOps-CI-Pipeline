# =============================================================================
# src/data/validate.py — Dataset validation
# =============================================================================
# Validates that a dataset in data/processed/ has the correct structure
# and content before it is used in the pipeline.
#
# Two validation layers:
#   validate_dataset()    — structural check on the versioned dataset dir
#                           (files present, yaml keys, column presence).
#   validate_split_data() — data contract check on a loaded DataFrame
#                           (dtypes, null counts, label set, row count).
#                           Called per split inside run_preprocessing().
# =============================================================================

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

### !!! IMPORTANT: This file needs to be updated later, it needs stronger validation

_ALLOWED_SCHEMA_TYPES: set[str] = {"float", "int", "string", "bool"}
_ALLOWED_CONSTRAINT_KEYS: set[str] = {"min_rows", "max_null_fraction", "label_classes"}
_ALLOWED_DATASET_YAML_KEYS: set[str] = {
    "name",
    "task_type",
    "description",
    "source",
    "created_at",
    "features",
    "target",
    "schema",
    "constraints",
    # Added by ingestion/versioning pipeline:
    "version_id",
    "split",
    "versioned_at"
}

_REQUIRED_DATASET_YAML_KEYS: set[str] = {"name", "task_type", "features", "target", "schema"}


def _is_dtype_compatible(series: pd.Series, declared_type: str) -> bool:
    """
    Robust dtype compatibility check.
    Avoids relying on dtype.kind for strings because pandas can use
    StringDtype (string[python]/string[pyarrow]) which is not kind 'O'.

    Args:
        series: The pandas Series to check.
        declared_type: The declared type from the schema (e.g., "int", "float", "string").

    Returns:
        bool: True if the series is compatible with the declared type, False otherwise.
    """
    
    if declared_type == "float":
        return pd.api.types.is_float_dtype(series)
    elif declared_type == "int":
        return pd.api.types.is_integer_dtype(series)
    elif declared_type == "bool":
        return pd.api.types.is_bool_dtype(series)
    elif declared_type == "string":
        return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)
    return True  ### unknown declared type, may need to make this stricter later
def validate_dataset(dataset_name: str, version_id: str, processed_dir: Path = Path("data/processed")) -> None:
    """Validates that the specified dataset version has the required files and that

    Args:
        dataset_name (str): The name of the dataset to validate (e.g., "customer_churn").
        version_id (str): The version ID of the dataset to validate (e.g., "v1.0").
        processed_dir (Path, optional): The path to the processed data directory. Defaults to Path("data/processed").
    """
    version_dir = processed_dir / dataset_name / version_id
    csv_path = version_dir / "data.csv"
    yaml_path = version_dir / "dataset.yaml"

    errors = []

    if not yaml_path.exists():
        errors.append(f"Missing dataset.yaml in {version_dir}")
    if not csv_path.exists():
        errors.append(f"Missing data.csv in {version_dir}")

    if errors:
        _fail(errors)

    with open(yaml_path, "r") as f:
        metadata = yaml.safe_load(f)

    if not isinstance(metadata, dict):
        _fail([f"Invalid dataset.yaml at '{yaml_path}': expected a YAML mapping/object"])

    missing_keys = _REQUIRED_DATASET_YAML_KEYS - metadata.keys()
    if missing_keys:
        errors.append(f"dataset.yaml is missing required fields: {', '.join(sorted(missing_keys))}")
        
    extra_top = set(metadata.keys()) - _ALLOWED_DATASET_YAML_KEYS
    if extra_top:
        logger.warning(
            "Unknown keys in dataset.yaml (%s): %s",
            dataset_name,
            ", ".join(sorted(extra_top)),
        )

    if metadata.get("task_type") not in {"classification", "regression"}:
        errors.append(f"Invalid task_type '{metadata.get('task_type')}' — must be classification or regression")

    if errors:
        _fail(errors)

    df_head = pd.read_csv(csv_path, nrows=0)

    expected_columns = set(metadata["features"]) | {metadata["target"]}
    actual_columns = set(df_head.columns)
    missing_columns = expected_columns - actual_columns
    if missing_columns:
        errors.append(f"data.csv is missing columns defined in schema: {', '.join(sorted(missing_columns))}")

    extra_columns = actual_columns - expected_columns
    if extra_columns:
        logger.warning(
            "data.csv contains extra column(s) not referenced by features/target: %s",
            ", ".join(sorted(extra_columns)),
        )

    schema_cols = set((metadata.get("schema") or {}).keys())
    extra_schema = schema_cols - expected_columns
    if extra_schema:
        logger.warning(
            "dataset.yaml schema contains column(s) not in features/target: %s",
            ", ".join(sorted(extra_schema)),
        )

    constraints = metadata.get("constraints", {}) or {}
    if isinstance(constraints, dict):
        extra_constraints = set(constraints.keys()) - _ALLOWED_CONSTRAINT_KEYS
        if extra_constraints:
            logger.warning(
                "Unknown keys in dataset.yaml constraints: %s",
                ", ".join(sorted(extra_constraints)),
            )

    df_one = pd.read_csv(csv_path, nrows=1)
    if df_one.empty:
        errors.append("data.csv is empty")

    if errors:
        _fail(errors)

    logger.info("  Dataset '%s' version '%s' passed validation.", dataset_name, version_id)


def _fail(errors: list[str]) -> None:
    error_report = "\n  - ".join(errors)
    raise ValueError(f"Dataset validation failed:\n  - {error_report}")


def validate_split_data(
    df: pd.DataFrame,
    metadata: dict,
    split_name: str,
) -> None:
    """ Validates a loaded DataFrame against the dataset.yaml contract.

    Args:
        df:         Loaded split DataFrame (train, val, or test).
        metadata:   Parsed dataset.yaml dict (must contain schema, constraints,
                    task_type, target).
        split_name: Used in error messages so CI logs point to the failing split.

    Raises:
        ValueError: If any critical constraint is violated. Warnings (null
                    fraction below threshold) are logged but do not raise.
    """
    errors: list[str] = []
    schema: dict = metadata.get("schema", {})
    constraints: dict = metadata.get("constraints", {})
    task_type: str = metadata.get("task_type", "regression")
    target: str = metadata.get("target", "")
    max_null_fraction: float = float(constraints.get("max_null_fraction", 0.0))
    min_rows: int = int(constraints.get("min_rows", 1))

    # --- Row count ---
    if len(df) < min_rows:
        errors.append(
            f"[{split_name}] Only {len(df)} row(s) present, minimum is {min_rows}"
        )

    # --- Dtype checks ---
    for col, declared_type in schema.items():
        if col not in df.columns:
            continue  
        if declared_type not in _ALLOWED_SCHEMA_TYPES:
            logger.warning(
                "[%s] Unknown declared schema type '%s' for column '%s' — skipping dtype check",
                split_name, declared_type, col
                )
            continue
        if not _is_dtype_compatible(df[col], declared_type):
            errors.append(
                f"[{split_name}] Column '{col}' declared as '{declared_type}' "
                f"but has dtype '{df[col].dtype}'"
            )

    # --- Null checks ---
    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue
        null_frac = null_count / len(df)
        if null_frac > max_null_fraction:
            # Consistent enforcement: contract violation is a hard fail.
            errors.append(
                f"[{split_name}] Column '{col}' null fraction {null_frac * 100:.1f}% "
                f"({null_count} null(s)) exceeds max {max_null_fraction * 100:.1f}%"
            )

    # --- Label set check (classification only) ---
    if task_type == "classification" and target in df.columns:
        allowed_labels: set = set(constraints.get("label_classes", []))
        if allowed_labels:
            actual_labels: set = set(df[target].dropna().unique())
            unknown = actual_labels - allowed_labels
            if unknown:
                errors.append(
                    f"[{split_name}] Target '{target}' contains unknown label(s): "
                    f"{sorted(unknown)}"
                )

    if errors:
        _fail(["Data contract validation failed:"] + errors)