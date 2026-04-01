"""Evaluation config loader — load and return an EvaluationConfig."""

from pathlib import Path

from src.config.schema import ClassificationEvalConfig, RegressionEvalConfig, EvaluationConfig
from src.config.validation import _load_yaml


def load_evaluation_config(path: Path) -> EvaluationConfig:
    raw = _load_yaml(path)
    cls_raw = raw.get("classification", {}) or {}
    averaging = cls_raw.get("averaging", "weighted")
    if averaging not in {"weighted", "macro", "binary"}:
        raise ValueError(
            f"Invalid averaging strategy '{averaging}' — must be weighted, macro, or binary."
        )
    return EvaluationConfig(
        classification=ClassificationEvalConfig(averaging=averaging),
        regression=RegressionEvalConfig(),
    )
