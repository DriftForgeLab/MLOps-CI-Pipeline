# =============================================================================
# src/promotion/comparator.py — Candidate vs production performance comparator
# =============================================================================
# Responsibility: Compare candidate model metrics against production model
# metrics and produce a structured per-metric verdict.
#
# Design: compare_metrics() is a pure function — no I/O, no side effects.
# It takes two metric dicts and a list of metric names to compare (driven
# by the promotion rules in promotion.yaml), and returns a structured
# comparison summary ready for inclusion in the evaluation report.
#
# Verdicts per metric: "better" / "worse" / "equal"
# Higher-is-better metrics: accuracy, f1_score, precision, recall, r2
# Lower-is-better metrics:  mae, mse, rmse
# =============================================================================

import logging

logger = logging.getLogger(__name__)

# Metrics where a lower value is better.
# All other metrics are assumed higher-is-better.
_LOWER_IS_BETTER: set[str] = {"mae", "mse", "rmse"}

# Tolerance for floating-point equality (treated as "equal" within this band).
_EQUAL_TOLERANCE: float = 1e-6


def _verdict(metric: str, candidate: float, production: float) -> str:
    """
    Return 'better', 'worse', or 'equal' for a single metric comparison.

    Args:
        metric:     Metric name — used to determine direction (higher/lower is better).
        candidate:  Candidate model's value for this metric.
        production: Production model's value for this metric.

    Returns:
        'equal'  if |candidate - production| <= _EQUAL_TOLERANCE
        'better' if candidate is strictly better than production
        'worse'  if candidate is strictly worse than production
    """
    diff = candidate - production

    if abs(diff) <= _EQUAL_TOLERANCE:
        return "equal"

    lower_is_better = metric in _LOWER_IS_BETTER
    if lower_is_better:
        return "better" if diff < 0 else "worse"
    else:
        return "better" if diff > 0 else "worse"


def compare_metrics(
    candidate_metrics: dict,
    production_metrics: dict,
    metrics_to_compare: list[str],
) -> dict:
    """
    Compare candidate metrics against production metrics.

    Only compares the metrics listed in metrics_to_compare (driven by the
    promotion rules in promotion.yaml). Metrics present in the evaluation
    report but not in the rules are ignored.

    Args:
        candidate_metrics:  Metrics dict from the current evaluation report.
        production_metrics: Metrics dict from the production model pointer.
        metrics_to_compare: List of metric names to include in the comparison.
                            Typically the set of metrics referenced by promotion rules.

    Returns:
        Structured comparison summary dict:
        {
            "has_production_model": bool,
            "per_metric": {
                "<metric_name>": {
                    "candidate":  float,
                    "production": float,
                    "delta":      float,
                    "verdict":    "better" | "worse" | "equal"
                },
                ...
            },
            "overall_verdict": "better" | "worse" | "equal" | "mixed",
            "is_better": bool,
        }
    """
    per_metric: dict[str, dict] = {}

    for metric in metrics_to_compare:
        if metric not in candidate_metrics:
            logger.warning(
                "  Comparison: metric '%s' not found in candidate metrics — skipping.", metric
            )
            continue
        if metric not in production_metrics:
            logger.warning(
                "  Comparison: metric '%s' not found in production metrics — skipping.", metric
            )
            continue

        candidate_val = float(candidate_metrics[metric])
        production_val = float(production_metrics[metric])
        delta = round(candidate_val - production_val, 6)
        v = _verdict(metric, candidate_val, production_val)

        per_metric[metric] = {
            "candidate":  candidate_val,
            "production": production_val,
            "delta":      delta,
            "verdict":    v,
        }

        logger.info(
            "  Comparison: %s — candidate=%.4f, production=%.4f, delta=%+.4f → %s",
            metric, candidate_val, production_val, delta, v.upper(),
        )

    # Derive overall verdict from individual verdicts
    verdicts = [m["verdict"] for m in per_metric.values()]
    if not verdicts:
        overall = "equal"
    elif all(v == "better" for v in verdicts):
        overall = "better"
    elif all(v == "worse" for v in verdicts):
        overall = "worse"
    elif all(v == "equal" for v in verdicts):
        overall = "equal"
    else:
        overall = "mixed"

    is_better = overall in {"better", "mixed"}

    return {
        "has_production_model": True,
        "per_metric": per_metric,
        "overall_verdict": overall,
        "is_better": is_better,
    }


def no_baseline_comparison() -> dict:
    """
    Return a clearly labelled 'no baseline' comparison state.

    Used when no production model exists yet (bootstrap scenario).
    The promotion step still requires explicit approval even in this state.

    Returns:
        {
            "has_production_model": False,
            "per_metric":           None,
            "overall_verdict":      "no_baseline",
            "is_better":            None,
        }
    """
    return {
        "has_production_model": False,
        "per_metric": None,
        "overall_verdict": "no_baseline",
        "is_better": None,
    }