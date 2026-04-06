# =============================================================================
# src/promotion/approval.py — Manual approval gate for model promotion
# =============================================================================
# Responsibility: Present a promotion summary to the user via CLI and require
# an explicit approve/reject decision before promotion can proceed.
#
# Design: Pure I/O function — no pipeline state is modified here.
# The caller (_promotion_stage in steps.py) decides what to do with the result.
# =============================================================================
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ApprovalResult:
    """
    Result of the manual approval gate.

    Attributes:
        approved: True if the user explicitly approved, False otherwise.
        reason:   Mandatory rejection reason if approved=False, else None.
    """
    approved: bool
    reason: str | None = None


def request_approval(report: dict, drift: dict | None = None) -> ApprovalResult:
    """
    Display a promotion summary and request explicit user approval via CLI.

    Always prints the promotion summary before asking. If the user rejects,
    a mandatory reason must be provided — an empty reason is treated as a
    cancellation and the pipeline stops.

    Args:
        report: The evaluation report dict produced by evaluate.py.
        drift:  Optional drift result dict (from drift_result.json). When
                provided, a drift status block is rendered inside the summary.

    Returns:
        ApprovalResult with approved=True, or approved=False with a reason.
    """
    _print_summary(report, drift=drift)

    print("\nManual approval required.")
    print("  [1] Approve — promote candidate to production")
    print("  [2] Reject  — block promotion (reason required)")
    print("  [Q] Cancel  — abort without decision")

    try:
        decision = input("\nYour decision [1/2/Q]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        decision = "q"

    if decision == "1":
        logger.info("  Approval gate: APPROVED by user.")
        return ApprovalResult(approved=True)

    if decision == "2":
        try:
            reason = input("Rejection reason (mandatory): ").strip()
        except (EOFError, KeyboardInterrupt):
            reason = ""
        if not reason:
            logger.warning("  Approval gate: REJECTED — no reason provided. Pipeline stopped.")
            return ApprovalResult(approved=False, reason=None)
        logger.info("  Approval gate: REJECTED by user. Reason: %s", reason)
        return ApprovalResult(approved=False, reason=reason)

    # Anything else (Q, empty, EOF, Ctrl+C, etc.) is treated as cancellation
    logger.warning("  Approval gate: CANCELLED — no decision provided. Pipeline stopped.")
    return ApprovalResult(approved=False, reason=None)


def _print_summary(report: dict, drift: dict | None = None) -> None:
    """Print a human-readable promotion summary to stdout."""
    metrics = report.get("metrics", {})
    comparison = report.get("comparison", {})
    overall_verdict = comparison.get("overall_verdict", "no_baseline")
    has_prod = comparison.get("has_production_model", False)

    print("\n" + "=" * 60)
    print("  PROMOTION SUMMARY")
    print("=" * 60)

    print("\n  Evaluation metrics:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"    {name:<20} {value:.4f}")

    print(f"\n  Comparison verdict:  {overall_verdict.upper()}")
    print(f"  Has production model: {has_prod}")

    if has_prod:
        per_metric = comparison.get("per_metric") or {}
        if per_metric:
            print("\n  Per-metric delta vs production:")
            for metric_name, data in per_metric.items():
                delta = data.get("delta")
                verdict = data.get("verdict", "")
                if delta is not None:
                    sign = "+" if delta > 0 else ""
                    print(f"    {metric_name:<20} {sign}{delta:.4f}  ({verdict})")
                else:
                    print(f"    {metric_name:<20} {'N/A':<10}  ({verdict})")

    if drift is not None:
        _print_drift_block(drift)

    print("\n" + "=" * 60)


def _print_drift_block(drift: dict) -> None:
    """Render the drift status block inside the promotion summary."""
    overall = drift.get("overall", {})
    features = drift.get("features", {})

    dataset_drift = overall.get("dataset_drift_detected", False)
    severity = overall.get("severity", "unknown")
    drifted_count = overall.get("drifted_feature_count", 0)
    total_count = overall.get("total_feature_count", 0)

    drifted_names = [
        name for name, data in features.items()
        if data.get("drift_detected")
    ]

    print("\n" + "-" * 60)
    print("  DRIFT STATUS")
    print("-" * 60)
    print(f"  Dataset drift detected:   {dataset_drift}")
    print(f"  Overall severity:         {severity.upper()}")
    if drifted_names:
        print(f"  Drifted features ({drifted_count}/{total_count}):   {', '.join(drifted_names)}")
    else:
        print(f"  Drifted features ({drifted_count}/{total_count}):   none")