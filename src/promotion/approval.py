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


def request_approval(report: dict) -> ApprovalResult:
    """
    Display a promotion summary and request explicit user approval via CLI.

    Always prints the promotion summary before asking. If the user rejects,
    a mandatory reason must be provided — an empty reason is treated as a
    cancellation and the pipeline stops.

    Args:
        report: The evaluation report dict produced by evaluate.py.

    Returns:
        ApprovalResult with approved=True, or approved=False with a reason.
    """
    _print_summary(report)

    print("\nManual approval required.")
    print("  [1] Approve — promote candidate to production")
    print("  [2] Reject  — block promotion (reason required)")
    print("  [Q] Cancel  — abort without decision")

    decision = input("\nYour decision [1/2/Q]: ").strip().lower()

    if decision == "1":
        logger.info("  Approval gate: APPROVED by user.")
        return ApprovalResult(approved=True)

    if decision == "2":
        reason = input("Rejection reason (mandatory): ").strip()
        if not reason:
            logger.warning("  Approval gate: REJECTED — no reason provided. Pipeline stopped.")
            return ApprovalResult(approved=False, reason=None)
        logger.info("  Approval gate: REJECTED by user. Reason: %s", reason)
        return ApprovalResult(approved=False, reason=reason)

    # Anything else (Q, empty, Ctrl+C, etc.) is treated as cancellation
    logger.warning("  Approval gate: CANCELLED — no decision provided. Pipeline stopped.")
    return ApprovalResult(approved=False, reason=None)


def _print_summary(report: dict) -> None:
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
                sign = "+" if delta and delta > 0 else ""
                print(f"    {metric_name:<20} {sign}{delta:.4f}  ({verdict})")

    print("\n" + "=" * 60)