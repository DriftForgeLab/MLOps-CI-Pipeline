import json
import logging

import pandas as pd
import pytest

from src.drift.alignment import load_reference_data, validate_feature_alignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_preprocessed_dir(tmp_path, dataset="iris", version="abc123"):
    """Create the standard preprocessed directory structure."""
    d = tmp_path / dataset / version / "preprocessed"
    d.mkdir(parents=True)
    return d


def _write_feature_map(preprocessed_dir, feature_map):
    path = preprocessed_dir / "feature_map.json"
    path.write_text(json.dumps(feature_map))


def _write_csv(preprocessed_dir, name, df):
    df.to_csv(preprocessed_dir / name, index=False)


IRIS_FEATURE_MAP = {
    "output_features": [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ],
    "target": "species",
    "numeric_input": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
    "categorical_input": [],
}

IRIS_REF = pd.DataFrame(
    {
        "sepal_length": [5.1, 4.9, 4.7],
        "sepal_width": [3.5, 3.0, 3.2],
        "petal_length": [1.4, 1.4, 1.3],
        "petal_width": [0.2, 0.2, 0.2],
        "species": [0, 1, 2],
    }
)


# ===========================================================================
# load_reference_data
# ===========================================================================


class TestLoadReferenceData:
    def test_happy_path(self, tmp_path):
        d = _make_preprocessed_dir(tmp_path)
        _write_feature_map(d, IRIS_FEATURE_MAP)
        _write_csv(d, "train.csv", IRIS_REF)

        df, fm = load_reference_data("iris", "abc123", "train", tmp_path)
        assert list(df.columns) == list(IRIS_REF.columns)
        assert len(df) == 3
        assert fm["output_features"] == IRIS_FEATURE_MAP["output_features"]

    def test_missing_feature_map(self, tmp_path):
        d = _make_preprocessed_dir(tmp_path)
        _write_csv(d, "train.csv", IRIS_REF)

        with pytest.raises(FileNotFoundError, match="feature_map.json"):
            load_reference_data("iris", "abc123", "train", tmp_path)

    def test_missing_csv(self, tmp_path):
        d = _make_preprocessed_dir(tmp_path)
        _write_feature_map(d, IRIS_FEATURE_MAP)

        with pytest.raises(FileNotFoundError, match="train.csv"):
            load_reference_data("iris", "abc123", "train", tmp_path)

    def test_alternate_split(self, tmp_path):
        d = _make_preprocessed_dir(tmp_path)
        _write_feature_map(d, IRIS_FEATURE_MAP)
        _write_csv(d, "val.csv", IRIS_REF)

        df, _ = load_reference_data("iris", "abc123", "val", tmp_path)
        assert len(df) == 3

    def test_logs_info_on_load(self, tmp_path, caplog):
        d = _make_preprocessed_dir(tmp_path)
        _write_feature_map(d, IRIS_FEATURE_MAP)
        _write_csv(d, "train.csv", IRIS_REF)

        with caplog.at_level(logging.INFO, logger="src.drift.alignment"):
            load_reference_data("iris", "abc123", "train", tmp_path)

        assert "Loaded reference data" in caplog.text
        assert "3 rows" in caplog.text


# ===========================================================================
# validate_feature_alignment
# ===========================================================================


class TestValidateFeatureAlignment:
    def test_identical_schemas_pass(self):
        errors = validate_feature_alignment(IRIS_REF, IRIS_REF.copy(), IRIS_FEATURE_MAP)
        assert errors == []

    def test_missing_column_in_current(self):
        current = IRIS_REF.drop(columns=["petal_width"])
        errors = validate_feature_alignment(IRIS_REF, current, IRIS_FEATURE_MAP)
        assert len(errors) == 1
        assert "petal_width" in errors[0]
        assert "current" in errors[0].lower()

    def test_missing_column_in_reference(self):
        ref = IRIS_REF.drop(columns=["petal_length"])
        errors = validate_feature_alignment(ref, IRIS_REF.copy(), IRIS_FEATURE_MAP)
        assert len(errors) == 1
        assert "petal_length" in errors[0]
        assert "reference" in errors[0].lower()

    def test_multiple_missing_columns(self):
        current = IRIS_REF.drop(columns=["petal_width", "sepal_width"])
        errors = validate_feature_alignment(IRIS_REF, current, IRIS_FEATURE_MAP)
        assert any("petal_width" in e for e in errors)
        assert any("sepal_width" in e for e in errors)

    def test_extra_column_in_current_warns(self, caplog):
        current = IRIS_REF.copy()
        current["extra_col"] = 1.0

        with caplog.at_level(logging.WARNING, logger="src.drift.alignment"):
            errors = validate_feature_alignment(IRIS_REF, current, IRIS_FEATURE_MAP)

        assert errors == []
        assert "extra_col" in caplog.text

    def test_type_mismatch_numeric_to_string(self):
        current = IRIS_REF.copy()
        current["sepal_length"] = current["sepal_length"].astype(str)
        errors = validate_feature_alignment(IRIS_REF, current, IRIS_FEATURE_MAP)
        assert len(errors) == 1
        assert "sepal_length" in errors[0]
        assert "numeric" in errors[0].lower()

    def test_type_mismatch_categorical_to_numeric(self):
        fm = {
            "output_features": ["cat_a", "num_b"],
            "target": "y",
        }
        ref = pd.DataFrame({"cat_a": ["x", "y", "z"], "num_b": [1, 2, 3], "y": [0, 1, 0]})
        cur = pd.DataFrame({"cat_a": [1, 2, 3], "num_b": [4, 5, 6], "y": [1, 0, 1]})

        errors = validate_feature_alignment(ref, cur, fm)
        assert len(errors) == 1
        assert "cat_a" in errors[0]

    def test_empty_current_warns(self, caplog):
        current = IRIS_REF.iloc[:0]

        with caplog.at_level(logging.WARNING, logger="src.drift.alignment"):
            errors = validate_feature_alignment(IRIS_REF, current, IRIS_FEATURE_MAP)

        assert errors == []
        assert "0 rows" in caplog.text

    def test_empty_output_features(self):
        fm = {"output_features": [], "target": "species"}
        errors = validate_feature_alignment(IRIS_REF, IRIS_REF.copy(), fm)
        assert len(errors) == 1
        assert "no output_features" in errors[0].lower()

    def test_different_row_counts_ok(self):
        """Different row counts are expected and should not produce errors."""
        current = IRIS_REF.iloc[:1]
        errors = validate_feature_alignment(IRIS_REF, current, IRIS_FEATURE_MAP)
        assert errors == []

    def test_classification_and_regression_both_work(self):
        """Alignment is task-type-agnostic — works for any tabular dataset."""
        # Regression-style: continuous target
        fm = {
            "output_features": ["x1", "x2"],
            "target": "price",
        }
        ref = pd.DataFrame({"x1": [1.0, 2.0], "x2": [3.0, 4.0], "price": [10.0, 20.0]})
        cur = pd.DataFrame({"x1": [5.0, 6.0], "x2": [7.0, 8.0], "price": [30.0, 40.0]})

        errors = validate_feature_alignment(ref, cur, fm)
        assert errors == []
