# =============================================================================
# src/promotion/rules.py — Promotion rule engine
# =============================================================================
# Evaluates evaluation metrics against configured promotion thresholds.
# Any rule violation hard-fails the pipeline with a structured error message.
# =============================================================================

import logging
from pathlib import Path

from src.config.loader import PromotionConfig, PromotionRule, load_promotion_config

logger = logging.getLogger(__name__)


def _evaluate_rule(rule: PromotionRule, observed: float) -> bool:
    """Evaluate a single rule. Returns True if passed, False if violated."""
    ops = {
        ">=": observed >= rule.threshold,
        "<=": observed <= rule.threshold,
        ">":  observed > rule.threshold,
        "<":  observed < rule.threshold,
        "==": observed == rule.threshold,
    }
    return ops[rule.operator]


def run_promotion_rules(
    metrics: dict,
    task_type: str,
    promotion_config: PromotionConfig,
) -> list[dict]:
    """
    Evaluate all promotion rules for the given task_type against metrics.

    Args:
        metrics:          Flat metrics dict from evaluation report.
        task_type:        "classification" or "regression"
        promotion_config: Loaded PromotionConfig

    Returns:
        List of violation dicts. Empty list means all rules passed.

    Each violation dict contains:
        - rule_id:     str
        - metric:      str
        - observed:    float
        - threshold:   float
        - operator:    str
        - description: str
    """
    task_config = (
        promotion_config.classification if task_type == "classification" else promotion_config.regression
    )

    violations: list[dict] = []

    for rule in task_config.rules:
        if rule.metric not in metrics:
            logger.warning(
                "Rule '%s' references metric '%s' which is not in evaluation report — skipping.",
                rule.id, rule.metric,
            )
            continue

        observed = float(metrics[rule.metric])
        passed = _evaluate_rule(rule, observed)

        if passed:
            logger.info(
                "  Rule '%s': %s=%.4f %s %.4f — PASSED",
                rule.id, rule.metric, observed, rule.operator, rule.threshold,
            )
        else:
            logger.warning(
                "  Rule '%s': %s=%.4f %s %.4f — FAILED",
                rule.id, rule.metric, observed, rule.operator, rule.threshold,
            )
            violations.append({
                "rule_id":     rule.id,
                "metric":      rule.metric,
                "observed":    observed,
                "threshold":   rule.threshold,
                "operator":    rule.operator,
                "description": rule.description,
            })

    return violations
