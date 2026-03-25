import pytest

from src.promotion.comparator import _verdict, compare_metrics, no_baseline_comparison


# ── _verdict: higher-is-better metrics ──────────────────────────────────────

class TestVerdictHigherIsBetter:
    @pytest.mark.parametrize(
        "metric, candidate, production, expected",
        [
            ("accuracy", 0.95, 0.90, "better"),
            ("f1_score", 0.80, 0.90, "worse"),
            ("precision", 0.70, 0.70, "equal"),
            ("recall", 0.85, 0.80, "better"),
            ("r2", 0.50, 0.60, "worse"),
        ],
    )
    def test_higher_is_better_direction(self, metric, candidate, production, expected):
        assert _verdict(metric, candidate, production) == expected


# ── _verdict: lower-is-better metrics ───────────────────────────────────────

class TestVerdictLowerIsBetter:
    @pytest.mark.parametrize(
        "metric, candidate, production, expected",
        [
            ("mae", 0.10, 0.20, "better"),
            ("mae", 0.30, 0.20, "worse"),
            ("mse", 0.05, 0.05, "equal"),
            ("rmse", 0.15, 0.25, "better"),
            ("rmse", 0.30, 0.20, "worse"),
        ],
    )
    def test_lower_is_better_direction(self, metric, candidate, production, expected):
        assert _verdict(metric, candidate, production) == expected


# ── _verdict: tolerance boundary ────────────────────────────────────────────

class TestVerdictTolerance:
    def test_within_tolerance_is_equal(self):
        assert _verdict("accuracy", 0.9000001, 0.9000000) == "equal"

    def test_exactly_at_tolerance_boundary_is_equal(self):
        # Use values where the diff is exactly representable as <= 1e-6
        assert _verdict("accuracy", 1.0000010, 1.0) == "equal"

    def test_just_outside_tolerance_is_better(self):
        assert _verdict("accuracy", 0.9 + 2e-6, 0.9) == "better"

    def test_float_repr_boundary_is_better(self):
        # 0.9 + 1e-6 - 0.9 > 1e-6 in IEEE 754 due to representation
        assert _verdict("accuracy", 0.9 + 1e-6, 0.9) == "better"

    def test_identical_values_are_equal(self):
        assert _verdict("f1_score", 0.85, 0.85) == "equal"

    def test_zero_diff_is_equal(self):
        assert _verdict("mae", 1.0, 1.0) == "equal"

    def test_large_values_tolerance_still_applies(self):
        base = 1e10
        assert _verdict("accuracy", base + 1e-7, base) == "equal"


# ── compare_metrics: overall verdicts ───────────────────────────────────────

class TestCompareMetrics:
    def test_all_better_returns_better(self):
        result = compare_metrics(
            {"accuracy": 0.95, "f1_score": 0.90},
            {"accuracy": 0.80, "f1_score": 0.75},
            ["accuracy", "f1_score"],
        )
        assert result["overall_verdict"] == "better"
        assert result["is_better"] is True
        assert result["has_production_model"] is True

    def test_all_worse_returns_worse(self):
        result = compare_metrics(
            {"accuracy": 0.70, "f1_score": 0.60},
            {"accuracy": 0.80, "f1_score": 0.75},
            ["accuracy", "f1_score"],
        )
        assert result["overall_verdict"] == "worse"
        assert result["is_better"] is False

    def test_all_equal_returns_equal(self):
        result = compare_metrics(
            {"accuracy": 0.90, "f1_score": 0.85},
            {"accuracy": 0.90, "f1_score": 0.85},
            ["accuracy", "f1_score"],
        )
        assert result["overall_verdict"] == "equal"
        assert result["is_better"] is False

    def test_mixed_verdicts_returns_mixed(self):
        result = compare_metrics(
            {"accuracy": 0.95, "f1_score": 0.60},
            {"accuracy": 0.80, "f1_score": 0.75},
            ["accuracy", "f1_score"],
        )
        assert result["overall_verdict"] == "mixed"
        assert result["is_better"] is False

    def test_empty_metrics_to_compare_returns_equal(self):
        result = compare_metrics({"accuracy": 0.9}, {"accuracy": 0.8}, [])
        assert result["overall_verdict"] == "equal"
        assert result["is_better"] is False

    def test_missing_candidate_metric_is_skipped(self):
        result = compare_metrics(
            {"accuracy": 0.95},
            {"accuracy": 0.80, "f1_score": 0.75},
            ["accuracy", "f1_score"],
        )
        assert "f1_score" not in result["per_metric"]
        assert result["overall_verdict"] == "better"

    def test_missing_production_metric_is_skipped(self):
        result = compare_metrics(
            {"accuracy": 0.95, "f1_score": 0.90},
            {"accuracy": 0.80},
            ["accuracy", "f1_score"],
        )
        assert "f1_score" not in result["per_metric"]
        assert result["overall_verdict"] == "better"


# ── compare_metrics: per_metric structure ───────────────────────────────────

class TestCompareMetricsPerMetric:
    def test_per_metric_structure(self):
        result = compare_metrics(
            {"accuracy": 0.95},
            {"accuracy": 0.90},
            ["accuracy"],
        )
        entry = result["per_metric"]["accuracy"]
        assert entry["candidate"] == 0.95
        assert entry["production"] == 0.90
        assert entry["delta"] == pytest.approx(0.05)
        assert entry["verdict"] == "better"

    def test_delta_is_rounded_to_6_decimals(self):
        result = compare_metrics(
            {"accuracy": 1 / 3},
            {"accuracy": 1 / 7},
            ["accuracy"],
        )
        delta = result["per_metric"]["accuracy"]["delta"]
        assert delta == round(1 / 3 - 1 / 7, 6)

    def test_string_metric_values_are_cast_to_float(self):
        result = compare_metrics(
            {"accuracy": "0.95"},
            {"accuracy": "0.90"},
            ["accuracy"],
        )
        assert result["per_metric"]["accuracy"]["candidate"] == 0.95
        assert result["per_metric"]["accuracy"]["verdict"] == "better"


# ── no_baseline_comparison ──────────────────────────────────────────────────

class TestNoBaselineComparison:
    def test_structure(self):
        result = no_baseline_comparison()
        assert result["has_production_model"] is False
        assert result["per_metric"] is None
        assert result["overall_verdict"] == "no_baseline"
        assert result["is_better"] is None
