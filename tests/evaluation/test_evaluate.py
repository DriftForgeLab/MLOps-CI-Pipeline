"""Tests for evaluate helpers — held-out test-split loading."""

from unittest.mock import MagicMock

import pandas as pd

from src.evaluation.evaluate import _load_optional_test_split


def _config(task_type="classification"):
    config = MagicMock()
    config.task_type = task_type
    return config


def test_returns_none_when_test_split_missing(tmp_path):
    assert _load_optional_test_split(_config(), tmp_path, ["a"], "y") is None


def test_returns_none_when_test_split_empty(tmp_path):
    pd.DataFrame({"a": [], "y": []}).to_csv(tmp_path / "test.csv", index=False)
    assert _load_optional_test_split(_config(), tmp_path, ["a"], "y") is None


def test_returns_arrays_when_test_split_populated(tmp_path):
    pd.DataFrame({"a": [1, 2, 3], "y": [0, 1, 0]}).to_csv(tmp_path / "test.csv", index=False)
    result = _load_optional_test_split(_config(), tmp_path, ["a"], "y")
    assert result is not None
    X, y = result
    assert len(y) == 3
