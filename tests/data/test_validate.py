# =============================================================================
# tests/data/test_validate.py
# =============================================================================

import pytest
import pandas as pd
import numpy as np

from src.data.validate import validate_split_data


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_metadata(task_type="classification"):
    return {
        "task_type": task_type,
        "target": "species",
        "features": ["sepal_len", "sepal_wid"],
        "schema": {
            "sepal_len": "float",
            "sepal_wid": "float",
            "species": "string",
        },
        "constraints": {
            "min_rows": 2,
            "max_null_fraction": 0.0,
            "label_classes": ["setosa", "versicolor"],
        },
    }


def _valid_df():
    return pd.DataFrame({
        "sepal_len": [5.1, 4.9],
        "sepal_wid": [3.5, 3.0],
        "species": ["setosa", "versicolor"],
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_valid_data_passes_without_error():
    validate_split_data(_valid_df(), _base_metadata(), split_name="train")


# ---------------------------------------------------------------------------
# Type violations
# ---------------------------------------------------------------------------

def test_string_in_float_column_raises():
    df = _valid_df()
    df["sepal_len"] = df["sepal_len"].astype(str)   # object dtype — declared as float

    with pytest.raises(ValueError, match="sepal_len"):
        validate_split_data(df, _base_metadata(), split_name="train")


def test_type_error_message_includes_split_name():
    df = _valid_df()
    df["sepal_len"] = df["sepal_len"].astype(str)

    with pytest.raises(ValueError, match="train"):
        validate_split_data(df, _base_metadata(), split_name="train")


# ---------------------------------------------------------------------------
# Null violations
# ---------------------------------------------------------------------------

def test_null_in_feature_column_raises():
    df = _valid_df()
    df.loc[0, "sepal_len"] = np.nan

    with pytest.raises(ValueError, match="sepal_len"):
        validate_split_data(df, _base_metadata(), split_name="train")


def test_null_in_target_column_raises():
    df = _valid_df()
    df.loc[0, "species"] = np.nan

    with pytest.raises(ValueError, match="species"):
        validate_split_data(df, _base_metadata(), split_name="train")


def test_null_within_threshold_does_not_raise():
    """A column with nulls below the configured max_null_fraction must not fail."""
    meta = _base_metadata()
    meta["constraints"]["max_null_fraction"] = 0.6   # allow up to 60% nulls

    df = _valid_df()
    df.loc[0, "sepal_len"] = np.nan   # 50% null — under threshold

    # Should not raise
    validate_split_data(df, meta, split_name="train")


# ---------------------------------------------------------------------------
# Label violations (classification only)
# ---------------------------------------------------------------------------

def test_unknown_label_raises():
    df = _valid_df()
    df.loc[0, "species"] = "unknown_class"

    with pytest.raises(ValueError, match="unknown_class"):
        validate_split_data(df, _base_metadata(), split_name="test")


def test_label_validation_skipped_for_regression():
    """Regression tasks have no label_classes — label validation must not run."""
    meta = _base_metadata(task_type="regression")
    meta["constraints"].pop("label_classes", None)

    df = _valid_df()
    df["species"] = ["any_value", "another_value"]   # arbitrary strings — fine for regression

    validate_split_data(df, meta, split_name="train")


# ---------------------------------------------------------------------------
# Row count violations
# ---------------------------------------------------------------------------

def test_too_few_rows_raises():
    df = _valid_df().head(1)   # only 1 row, min_rows=2

    with pytest.raises(ValueError, match="row"):
        validate_split_data(df, _base_metadata(), split_name="val")


def test_exactly_min_rows_passes():
    df = _valid_df()   # 2 rows, min_rows=2

    validate_split_data(df, _base_metadata(), split_name="val")


# ---------------------------------------------------------------------------
# Multiple errors are collected and reported together
# ---------------------------------------------------------------------------

def test_multiple_violations_reported_in_single_error():
    """All violations must appear in one ValueError — not just the first one."""
    df = _valid_df()
    df["sepal_len"] = df["sepal_len"].astype(str)   # type error
    df.loc[0, "species"] = "unknown_class"           # label error

    with pytest.raises(ValueError) as exc_info:
        validate_split_data(df, _base_metadata(), split_name="train")

    message = str(exc_info.value)
    assert "sepal_len" in message
    assert "unknown_class" in message
