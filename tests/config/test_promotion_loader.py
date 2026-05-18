"""Tests for load_promotion_config — promotion_evaluation_split parsing."""

import pytest

from src.config.promotion_loader import load_promotion_config

_RULES = """
classification:
  rules:
    - id: min_accuracy
      metric: accuracy
      threshold: 0.8
      operator: ">="
      description: acc
regression:
  rules:
    - id: min_r2
      metric: r2
      threshold: 0.8
      operator: ">="
      description: r2
"""


def _write(tmp_path, extra=""):
    p = tmp_path / "promotion.yaml"
    p.write_text(extra + _RULES)
    return p


def test_defaults_to_val_when_key_absent(tmp_path):
    cfg = load_promotion_config(_write(tmp_path))
    assert cfg.promotion_evaluation_split == "val"


def test_parses_explicit_test_split(tmp_path):
    cfg = load_promotion_config(_write(tmp_path, 'promotion_evaluation_split: "test"\n'))
    assert cfg.promotion_evaluation_split == "test"


def test_parses_both(tmp_path):
    cfg = load_promotion_config(_write(tmp_path, 'promotion_evaluation_split: "both"\n'))
    assert cfg.promotion_evaluation_split == "both"


def test_rejects_invalid_split(tmp_path):
    with pytest.raises(ValueError, match="promotion_evaluation_split"):
        load_promotion_config(_write(tmp_path, 'promotion_evaluation_split: "holdout"\n'))
