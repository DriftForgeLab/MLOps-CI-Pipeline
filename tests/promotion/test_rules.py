import pytest

from src.config.loader import PromotionConfig, PromotionRule, PromotionTaskConfig
from src.promotion.rules import _evaluate_rule, run_promotion_rules


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_rule(id, metric, threshold, operator=">=", description="test rule"):
    return PromotionRule(
        id=id, metric=metric, threshold=threshold,
        operator=operator, description=description,
    )


def _make_config(classification_rules, regression_rules=None):
    return PromotionConfig(
        classification=PromotionTaskConfig(rules=tuple(classification_rules)),
        regression=PromotionTaskConfig(rules=tuple(regression_rules or [])),
    )


# ── _evaluate_rule: all operators ───────────────────────────────────────────

class TestEvaluateRule:
    @pytest.mark.parametrize(
        "operator, observed, threshold, expected",
        [
            (">=", 0.95, 0.90, True),
            (">=", 0.90, 0.90, True),
            (">=", 0.85, 0.90, False),
            ("<=", 0.10, 0.20, True),
            ("<=", 0.20, 0.20, True),
            ("<=", 0.30, 0.20, False),
            (">", 0.95, 0.90, True),
            (">", 0.90, 0.90, False),
            ("<", 0.10, 0.20, True),
            ("<", 0.20, 0.20, False),
            ("==", 0.90, 0.90, True),
            ("==", 0.89, 0.90, False),
            ("==", 0.90 + 1e-7, 0.90, True),   # within tolerance
            ("==", 0.90 + 2e-6, 0.90, False),   # outside tolerance
        ],
    )
    def test_operator_evaluation(self, operator, observed, threshold, expected):
        rule = _make_rule("r1", "accuracy", threshold, operator)
        assert _evaluate_rule(rule, observed) is expected

    def test_invalid_operator_raises_key_error(self):
        rule = _make_rule("r1", "accuracy", 0.9, "!=")
        with pytest.raises(KeyError):
            _evaluate_rule(rule, 0.95)


# ── run_promotion_rules: pass/fail/skip ─────────────────────────────────────

class TestRunPromotionRules:
    def test_all_rules_pass_returns_empty(self):
        config = _make_config([
            _make_rule("r1", "accuracy", 0.80),
            _make_rule("r2", "f1_score", 0.70),
        ])
        violations = run_promotion_rules(
            {"accuracy": 0.95, "f1_score": 0.85},
            "classification", config,
        )
        assert violations == []

    def test_one_rule_fails_returns_violation(self):
        config = _make_config([
            _make_rule("r1", "accuracy", 0.90),
            _make_rule("r2", "f1_score", 0.90),
        ])
        violations = run_promotion_rules(
            {"accuracy": 0.95, "f1_score": 0.80},
            "classification", config,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v["rule_id"] == "r2"
        assert v["metric"] == "f1_score"
        assert v["observed"] == 0.80
        assert v["threshold"] == 0.90
        assert v["operator"] == ">="
        assert v["description"] == "test rule"

    def test_multiple_rules_fail_returns_multiple_violations(self):
        config = _make_config([
            _make_rule("r1", "accuracy", 0.90),
            _make_rule("r2", "f1_score", 0.90),
        ])
        violations = run_promotion_rules(
            {"accuracy": 0.70, "f1_score": 0.60},
            "classification", config,
        )
        assert len(violations) == 2

    def test_missing_metric_is_a_violation(self):
        config = _make_config([
            _make_rule("r1", "accuracy", 0.80),
            _make_rule("r2", "nonexistent_metric", 0.50),
        ])
        violations = run_promotion_rules(
            {"accuracy": 0.90},
            "classification", config,
        )
        assert len(violations) == 1
        v = violations[0]
        assert v["rule_id"] == "r2"
        assert v["metric"] == "nonexistent_metric"
        assert v["observed"] is None
        assert "not found" in v["description"]

    def test_empty_rules_returns_empty(self):
        config = _make_config([])
        violations = run_promotion_rules(
            {"accuracy": 0.90},
            "classification", config,
        )
        assert violations == []


# ── run_promotion_rules: task type routing ──────────────────────────────────

class TestTaskTypeRouting:
    def setup_method(self):
        self.config = PromotionConfig(
            classification=PromotionTaskConfig(
                rules=(
                    _make_rule("cls1", "accuracy", 0.80),
                ),
            ),
            regression=PromotionTaskConfig(
                rules=(
                    _make_rule("reg1", "rmse", 0.50, "<="),
                ),
            ),
        )

    def test_classification_uses_classification_rules(self):
        violations = run_promotion_rules(
            {"accuracy": 0.70}, "classification", self.config,
        )
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "cls1"

    def test_image_classification_uses_classification_rules(self):
        violations = run_promotion_rules(
            {"accuracy": 0.70}, "image_classification", self.config,
        )
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "cls1"

    def test_image_classification_cnn_uses_classification_rules(self):
        violations = run_promotion_rules(
            {"accuracy": 0.70}, "image_classification_cnn", self.config,
        )
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "cls1"

    def test_regression_uses_regression_rules(self):
        violations = run_promotion_rules(
            {"rmse": 0.80}, "regression", self.config,
        )
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "reg1"

    def test_unknown_task_type_falls_back_to_regression(self):
        violations = run_promotion_rules(
            {"rmse": 0.80}, "unknown_type", self.config,
        )
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "reg1"
