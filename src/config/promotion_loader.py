"""Promotion config loader — load and return a PromotionConfig."""

from pathlib import Path

from src.config.schema import (
    VALID_OPERATORS,
    _REQUIRED_RULE_KEYS,
    PromotionRule,
    PromotionTaskConfig,
    PromotionConfig,
)
from src.config.validation import _load_yaml


def _build_promotion_task_config(raw: dict) -> PromotionTaskConfig:
    errors: list[str] = []
    rules = []
    for i, rule in enumerate(raw.get("rules", [])):
        if not isinstance(rule, dict):
            errors.append(f"Rule at index {i}: must be a mapping, got {type(rule).__name__}")
            continue
        missing = _REQUIRED_RULE_KEYS - rule.keys()
        if missing:
            errors.append(
                f"Rule at index {i} (id={rule.get('id', '?')}): "
                f"missing required keys: {', '.join(sorted(missing))}"
            )
            continue
        if rule["operator"] not in VALID_OPERATORS:
            errors.append(
                f"Rule '{rule['id']}': invalid operator '{rule['operator']}' "
                f"— must be one of {sorted(VALID_OPERATORS)}"
            )
            continue
        rules.append(PromotionRule(
            id=rule["id"],
            metric=rule["metric"],
            threshold=float(rule["threshold"]),
            operator=rule["operator"],
            description=rule.get("description", ""),
        ))
    if errors:
        raise ValueError("Promotion config validation failed:\n  - " + "\n  - ".join(errors))
    return PromotionTaskConfig(rules=tuple(rules))


def load_promotion_config(path: Path) -> PromotionConfig:
    raw = _load_yaml(path)
    return PromotionConfig(
        classification=_build_promotion_task_config(raw.get("classification", {})),
        regression=_build_promotion_task_config(raw.get("regression", {})),
    )
