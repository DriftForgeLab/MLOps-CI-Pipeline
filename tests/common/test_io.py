import json
import math

import numpy as np
import pandas as pd
import pytest

from src.common.io import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_pickle,
    atomic_write_text,
    sanitize_for_json,
)


# ── sanitize_for_json ──────────────────────────────────────────────────────

class TestSanitizeForJson:
    def test_nan_becomes_none(self):
        assert sanitize_for_json(float("nan")) is None

    def test_inf_becomes_none(self):
        assert sanitize_for_json(float("inf")) is None

    def test_negative_inf_becomes_none(self):
        assert sanitize_for_json(float("-inf")) is None

    def test_normal_float_passes_through(self):
        assert sanitize_for_json(3.14) == 3.14

    def test_int_passes_through(self):
        assert sanitize_for_json(42) == 42

    def test_string_passes_through(self):
        assert sanitize_for_json("hello") == "hello"

    def test_bool_passes_through(self):
        assert sanitize_for_json(True) is True

    def test_none_passes_through(self):
        assert sanitize_for_json(None) is None

    def test_numpy_integer_becomes_int(self):
        assert sanitize_for_json(np.int64(5)) == 5
        assert isinstance(sanitize_for_json(np.int64(5)), int)

    def test_numpy_float_becomes_float(self):
        result = sanitize_for_json(np.float64(3.14))
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_numpy_nan_becomes_none(self):
        assert sanitize_for_json(np.float64("nan")) is None

    def test_numpy_inf_becomes_none(self):
        assert sanitize_for_json(np.float64("inf")) is None

    def test_nested_dict_sanitized(self):
        data = {"a": float("nan"), "b": {"c": np.int64(10)}}
        result = sanitize_for_json(data)
        assert result == {"a": None, "b": {"c": 10}}

    def test_list_sanitized(self):
        data = [1, float("nan"), "text", np.float64(2.5)]
        result = sanitize_for_json(data)
        assert result == [1, None, "text", 2.5]

    def test_tuple_becomes_list(self):
        result = sanitize_for_json((1, 2, 3))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_dict_keys_become_strings(self):
        result = sanitize_for_json({1: "a", 2: "b"})
        assert result == {"1": "a", "2": "b"}

    def test_unknown_type_becomes_string(self):
        class Custom:
            def __str__(self):
                return "custom_obj"

        assert sanitize_for_json(Custom()) == "custom_obj"


# ── atomic_write_json ──────────────────────────────────────────────────────

class TestAtomicWriteJson:
    def test_writes_valid_json(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"key": "value", "count": 42})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"key": "value", "count": 42}

    def test_nan_in_data_produces_valid_json(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"score": float("nan"), "name": "test"})
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["score"] is None
        assert data["name"] == "test"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.json"
        atomic_write_json(path, {"ok": True})
        assert path.exists()

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = tmp_path / "out.json"
        atomic_write_json(path, {"ok": True})
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


# ── atomic_write_text ──────────────────────────────────────────────────────

class TestAtomicWriteText:
    def test_writes_text_content(self, tmp_path):
        path = tmp_path / "out.txt"
        atomic_write_text(path, "hello world")
        assert path.read_text(encoding="utf-8") == "hello world"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "out.txt"
        atomic_write_text(path, "data")
        assert path.exists()

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = tmp_path / "out.txt"
        atomic_write_text(path, "data")
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


# ── atomic_write_csv ───────────────────────────────────────────────────────

class TestAtomicWriteCsv:
    def test_preserves_dataframe_content(self, tmp_path):
        path = tmp_path / "out.csv"
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        atomic_write_csv(df, path)
        loaded = pd.read_csv(path)
        pd.testing.assert_frame_equal(loaded, df)

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = tmp_path / "out.csv"
        df = pd.DataFrame({"x": [1]})
        atomic_write_csv(df, path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []


# ── atomic_write_pickle ────────────────────────────────────────────────────

class TestAtomicWritePickle:
    def test_round_trip_with_joblib(self, tmp_path):
        import joblib
        path = tmp_path / "model.pkl"
        obj = {"key": [1, 2, 3], "nested": {"a": "b"}}
        atomic_write_pickle(obj, path)
        loaded = joblib.load(path)
        assert loaded == obj

    def test_no_tmp_file_left_behind(self, tmp_path):
        path = tmp_path / "model.pkl"
        atomic_write_pickle({"x": 1}, path)
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert tmp_files == []
