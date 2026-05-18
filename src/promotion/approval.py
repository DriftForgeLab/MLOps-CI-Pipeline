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
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_DRIFT_STALE_AGE_DAYS = 7


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


class ApprovalUnavailableError(Exception):
    """
    Raised when no human decision could be obtained at the approval gate
    because stdin is closed or not attached to a TTY (e.g. CI runners).

    Distinct from an active user cancellation (Q / Ctrl+C): this signals
    that the *environment* could not present the prompt to a human, not
    that a human chose not to decide. Callers should treat it as a
    structural outcome separate from `failed` (engineering breakage) and
    `blocked` (governance rejection).
    """


def request_approval(
    report: dict,
    drift: dict | None = None,
    drift_eval: dict | None = None,
) -> ApprovalResult:
    """
    Display a promotion summary and request explicit user approval via CLI.

    Always prints the promotion summary before asking. If the user rejects,
    a mandatory reason must be provided — an empty reason is treated as a
    cancellation and the pipeline stops.

    Args:
        report:     The evaluation report dict produced by evaluate.py.
        drift:      Optional drift result dict (from drift_result.json). When
                    provided, a drift status block is rendered inside the summary.
        drift_eval: Optional drift adaptation evaluation dict (from
                    drift_adaptation_eval.json). When provided, a before/after
                    holdout performance comparison is rendered so the user can
                    factor fine-tuning improvement into their promotion decision.

    Returns:
        ApprovalResult with approved=True, or approved=False with a reason.

    Raises:
        ApprovalUnavailableError: stdin returned EOF before any decision was
            entered, indicating the gate ran in a non-interactive environment
            (e.g. CI). Distinct from an active user cancellation.
    """
    _print_summary(report, drift=drift, drift_eval=drift_eval)

    print("\nManual approval required.")
    print("  [1] Approve — promote candidate to production")
    print("  [2] Reject  — block promotion (reason required)")
    print("  [Q] Cancel  — abort without decision")

    try:
        decision = input("\nYour decision [1/2/Q]: ").strip().lower()
    except EOFError as exc:
        raise ApprovalUnavailableError(
            "Approval gate could not run: stdin returned EOF before a decision "
            "was entered (typical of CI runners and other non-interactive shells)."
        ) from exc
    except KeyboardInterrupt:
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


def _print_summary(
    report: dict,
    drift: dict | None = None,
    drift_eval: dict | None = None,
) -> None:
    """Print a human-readable promotion summary to stdout."""
    metrics = report.get("metrics", {})
    comparison = report.get("comparison", {})
    overall_verdict = comparison.get("overall_verdict", "no_baseline")
    has_prod = comparison.get("has_production_model", False)
    per_metric = comparison.get("per_metric") or {}

    print("\n" + "=" * 60)
    print("  PROMOTION SUMMARY")
    print("=" * 60)

    if drift_eval is not None and has_prod:
        # Drift-adaptive retrain: split into two clearly labelled sections so
        # the user can see both the clean-data trade-off and the drift improvement.
        print()
        print("  This model was retrained to adapt to data drift.")
        print("  Two comparisons are shown: performance on the original")
        print("  (clean) test set, and performance on the drifted data.")

        # ── Section 1: clean test set vs. production ─────────────────
        print("\n" + "-" * 60)
        print("  CLEAN TEST SET  —  candidate vs. current production model")
        print("-" * 60)
        if per_metric:
            print(f"\n  {'Metric':<14}  {'Candidate':>10}  {'Production':>10}  {'Delta':>10}")
            print(f"  {'-'*14}  {'-'*10}  {'-'*10}  {'-'*10}")
            for mname, data in per_metric.items():
                cand    = data.get("candidate")
                prod    = data.get("production")
                delta   = data.get("delta")
                verdict = data.get("verdict", "")
                cand_s  = f"{cand:.4f}"  if isinstance(cand,  float) else "N/A"
                prod_s  = f"{prod:.4f}"  if isinstance(prod,  float) else "N/A"
                sign    = "+" if isinstance(delta, float) and delta > 0 else ""
                delta_s = f"{sign}{delta:.4f}" if isinstance(delta, float) else "N/A"
                print(f"  {mname:<14}  {cand_s:>10}  {prod_s:>10}  {delta_s:>10}  ({verdict})")
            print(f"\n  Overall: {overall_verdict.upper()} on the original data distribution.")
            if overall_verdict in ("worse", "equal"):
                print("  A small clean-data drop is expected — retraining on drifted")
                print("  samples shifts the model toward the new data distribution.")

        # ── Section 2: drifted holdout before vs. after ───────────────
        _print_drift_eval_block(drift_eval)

    else:
        # Standard run: flat layout.
        print("\n  Evaluation metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                print(f"    {name:<20} {value:.4f}")

        if not has_prod:
            print("\n  No production model yet — this will become the first baseline.")
        else:
            print(f"\n  Comparison vs. current production model: {overall_verdict.upper()}")
            if per_metric:
                print(f"\n  {'Metric':<20}  {'Candidate':>10}  {'Production':>10}  {'Delta':>10}")
                print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
                for mname, data in per_metric.items():
                    cand    = data.get("candidate")
                    prod    = data.get("production")
                    delta   = data.get("delta")
                    verdict = data.get("verdict", "")
                    cand_s  = f"{cand:.4f}"  if isinstance(cand,  float) else "N/A"
                    prod_s  = f"{prod:.4f}"  if isinstance(prod,  float) else "N/A"
                    sign    = "+" if isinstance(delta, float) and delta > 0 else ""
                    delta_s = f"{sign}{delta:.4f}" if isinstance(delta, float) else "N/A"
                    print(f"  {mname:<20}  {cand_s:>10}  {prod_s:>10}  {delta_s:>10}  ({verdict})")

        if drift_eval is not None:
            _print_drift_eval_block(drift_eval)

    # Fallback: when no drift-adaptation eval block was rendered, surface the
    # monitoring drift status so the reviewer always sees the governance
    # signal. _print_drift_block handles drift=None with an explicit
    # "no drift data" banner. The block is suppressed when a drift_eval block
    # is shown — the raw pre-training "high drift" label would mislead after
    # fine-tuning, and the adaptation eval block already covers that state.
    if drift_eval is None:
        _print_drift_block(drift)

    print("\n" + "=" * 60)


def _print_drift_eval_block(drift_eval: dict) -> None:
    """Render the drift adaptation evaluation block inside the promotion summary."""
    baseline = drift_eval.get("baseline", {})
    after    = drift_eval.get("after_finetuning", {})
    delta    = drift_eval.get("delta", {})
    improved = drift_eval.get("improved", False)
    n        = drift_eval.get("n_holdout_samples", drift_eval.get("n_holdout_images", "?"))

    def _fmt(v) -> str:
        return f"{v:.4f}" if isinstance(v, float) else "N/A"

    def _fmt_delta(key: str) -> str:
        v = delta.get(key)
        if v is None:
            return "N/A"
        return f"{'+'if v >= 0 else ''}{v:.4f}"

    sample_label = "drifted images" if drift_eval.get("task_type", "").startswith("image") else "drifted samples"
    print("\n" + "-" * 60)
    print(f"  DRIFTED DATA  —  production model before vs. retrained model after")
    print(f"  ({n} {sample_label})")
    print("-" * 60)
    clf_keys = [("accuracy","Accuracy"),("f1_score","F1 score"),("precision","Precision"),("recall","Recall")]
    reg_keys = [("r2","R2"),("mae","MAE"),("rmse","RMSE")]
    metric_pairs = reg_keys if "r2" in baseline else clf_keys

    print(f"\n  {'Metric':<12}  {'Before':>10}  {'After':>10}  {'Delta':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for key, label in metric_pairs:
        print(
            f"  {label:<12}  {_fmt(baseline.get(key)):>10}"
            f"  {_fmt(after.get(key)):>10}  {_fmt_delta(key):>10}"
        )
    verdict = "IMPROVED" if improved else "NO IMPROVEMENT"
    print(f"\n  Result: {verdict}")


def _print_drift_block(drift: dict | None) -> None:
    """Render the drift status block inside the promotion summary.

    When ``drift`` is ``None`` a "no drift data" banner is rendered instead —
    explicit > silent, so reviewers can see the governance gap.
    """
    print("\n" + "-" * 60)
    print("  DRIFT STATUS")
    print("-" * 60)

    if drift is None:
        print("  No drift data available for this model.")
        print("  Run monitor-drift against a production batch to populate history.")
        return

    overall = drift.get("overall", {})
    features = drift.get("features", {})

    dataset_drift = overall.get("dataset_drift_detected", False)
    severity = overall.get("severity", "unknown")
    drift_type = drift.get("drift_type", "")
    is_image_drift = drift_type.startswith("image_")

    print(f"  Dataset drift detected:   {dataset_drift}")
    print(f"  Overall severity:         {severity.upper()}")

    if not is_image_drift:
        drifted_count = overall.get("drifted_feature_count", 0)
        total_count = overall.get("total_feature_count", 0)
        drifted_names = [
            name for name, data in features.items()
            if data.get("drift_detected")
        ]
        if drifted_names:
            print(f"  Drifted features ({drifted_count}/{total_count}):   {', '.join(drifted_names)}")
        else:
            print(f"  Drifted features ({drifted_count}/{total_count}):   none")
    else:
        drift_score = overall.get("drift_score")
        if drift_score is not None:
            print(f"  Drift score:              {drift_score:.4f}")

    warnings = drift.get("warnings") or []
    if warnings:
        print(f"  Warnings:                 {', '.join(warnings)}")

    generated_at = drift.get("generated_at")
    if generated_at:
        age_line = _format_drift_age(generated_at)
        if age_line is not None:
            print(age_line)


def _format_drift_age(generated_at: str) -> str | None:
    """Return a line reporting the age of the drift report, or ``None`` on parse failure."""
    try:
        ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age = datetime.now(timezone.utc) - ts
    days = age.days
    if days >= _DRIFT_STALE_AGE_DAYS:
        return f"  Report age:               {days} days (STALE — older than {_DRIFT_STALE_AGE_DAYS} days)"
    if days >= 1:
        return f"  Report age:               {days} days"
    hours = age.total_seconds() / 3600
    return f"  Report age:               {hours:.1f} hours"