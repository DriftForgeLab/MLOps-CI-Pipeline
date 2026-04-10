# =============================================================================
# src/monitoring/drift_decision.py — Drift response decision gate
# =============================================================================
# Responsibility: When drift exceeds the configured alert threshold, present
# a structured menu of response options and record the user's choice.
#
# Analogous to src/promotion/approval.py — pure I/O, no pipeline state is
# modified here. The caller (steps.py or monitor_cli.py) persists the result.
#
# Decision options
# ----------------
# retrain       Trigger model retraining with updated/additional data.
# collect_data  Flag the batch for additional data collection before acting.
# adjust_isp    Adjust ISP parameters and re-preprocess (raw-image pipelines).
#               Only offered when is_image_isp=True.
# accept        Accept drift — continue with the current model as-is.
# escalate      Escalate to a senior reviewer or incident-response track.
#
# Output dict (returned by DriftDecision.to_dict())
# -------------------------------------------------
# schema_version    "1.0.0"
# decided_at        ISO 8601 timestamp
# drift_type        drift_result["drift_type"], e.g. "tabular" | "image_isp"
# drift_severity    drift_result["overall"]["severity"]
# option            one of the keys listed above
# reason            mandatory free-text reason (empty string if cancelled)
# drift_report_linked  name of the associated drift report file (traceability)
# =============================================================================
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

DRIFT_DECISION_OPTIONS: dict[str, str] = {
    "retrain":      "Trigger model retraining with updated/additional data",
    "collect_data": "Flag batch for additional data collection before acting",
    "adjust_isp":   "Adjust ISP parameters and re-preprocess (raw-image pipelines)",
    "accept":       "Accept drift — continue with current model as-is",
    "escalate":     "Escalate to senior reviewer / incident-response track",
}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DriftDecision:
    """Captures the user's drift response choice.

    Attributes:
        option:              One of the DRIFT_DECISION_OPTIONS keys, or
                             "cancelled" if the user aborted without a decision.
        reason:              Mandatory free-text reason provided by the user.
        decided_at:          ISO 8601 timestamp of the decision.
        drift_type:          Drift type string from the drift result dict.
        drift_severity:      Overall severity from the drift result dict.
        drift_report_linked: File name of the associated drift report for
                             traceability (e.g. "drift_result.json" or
                             "sensitivity_report.json").
    """
    option: str
    reason: str
    decided_at: str
    drift_type: str
    drift_severity: str
    drift_report_linked: str = "drift_result.json"

    def to_dict(self) -> dict:
        return {
            "schema_version":      "1.0.0",
            "decided_at":          self.decided_at,
            "drift_type":          self.drift_type,
            "drift_severity":      self.drift_severity,
            "option":              self.option,
            "reason":              self.reason,
            "drift_report_linked": self.drift_report_linked,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def request_drift_decision(
    drift_result: dict,
    is_image_isp: bool = False,
    drift_report_linked: str = "drift_result.json",
) -> DriftDecision | None:
    """Present a drift summary and prompt the user to choose a response action.

    Prints a human-readable drift summary, then presents a numbered menu of
    response options. The user must provide a reason for their choice.
    Returns None if the user cancels (Q / EOF / Ctrl+C / empty reason).

    Args:
        drift_result:        Standard drift result dict as produced by the drift
                             stage (tabular) or image drift stage. Must contain
                             at minimum: drift_result["overall"]["severity"] and
                             drift_result["drift_type"].
        is_image_isp:        True for ISP-based image drift (adds the
                             "adjust_isp" option and renders scenario table).
        drift_report_linked: File name to embed in the decision for traceability.
                             Callers should pass the actual report file name.

    Returns:
        DriftDecision if the user made a valid choice, else None.
    """
    _print_drift_summary(drift_result, is_image_isp=is_image_isp)

    # Build the menu — adjust_isp only for raw-ISP image drift
    menu: dict[str, tuple[str, str]] = {}
    idx = 1
    for key, label in DRIFT_DECISION_OPTIONS.items():
        if key == "adjust_isp" and not is_image_isp:
            continue
        menu[str(idx)] = (key, label)
        idx += 1

    print("\nDrift response required.")
    for num, (key, label) in menu.items():
        print(f"  [{num}] {key:<14} — {label}")
    print("  [Q] Cancel        — Abort without logging a decision")

    try:
        choice = input("\nYour choice: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        choice = "q"

    if choice == "q" or choice not in menu:
        logger.warning(
            "  Drift decision gate: CANCELLED — no decision recorded."
        )
        return None

    option_key, option_label = menu[choice]

    # Mandatory reason
    try:
        reason = input(f"Reason for '{option_key}' (mandatory): ").strip()
    except (EOFError, KeyboardInterrupt):
        reason = ""

    if not reason:
        logger.warning(
            "  Drift decision gate: CANCELLED — no reason provided."
        )
        return None

    overall = drift_result.get("overall", {})
    decision = DriftDecision(
        option=option_key,
        reason=reason,
        decided_at=datetime.now(timezone.utc).isoformat(),
        drift_type=drift_result.get("drift_type", "unknown"),
        drift_severity=overall.get("severity", "unknown"),
        drift_report_linked=drift_report_linked,
    )

    logger.info(
        "  Drift decision: option=%s  severity=%s  reason=%s",
        decision.option, decision.drift_severity, decision.reason,
    )
    return decision


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_drift_summary(drift_result: dict, is_image_isp: bool = False) -> None:
    """Print a human-readable drift alert to stdout.

    Dispatches on drift_type:
      image_statistical / image_embedding  → _print_image_monitor_block()
      anything else (tabular)              → _print_tabular_block()
    """
    overall = drift_result.get("overall", {})
    severity = overall.get("severity", "unknown").upper()
    drift_type = drift_result.get("drift_type", "unknown")

    print("\n" + "=" * 60)
    print("  DRIFT ALERT")
    print("=" * 60)
    print(f"\n  Drift type:  {drift_type}")
    print(f"  Severity:    {severity}")

    if drift_type.startswith("image_"):
        _print_image_monitor_block(drift_result)
    else:
        _print_tabular_block(drift_result)

    print("\n" + "=" * 60)


def _print_tabular_block(drift_result: dict) -> None:
    """Render the tabular drift detail block (Evidently output format)."""
    overall = drift_result.get("overall", {})
    dataset_drift = overall.get("dataset_drift_detected", False)
    drifted_count = overall.get("drifted_feature_count", 0)
    total_count   = overall.get("total_feature_count", 0)
    drift_share   = overall.get("drift_share", 0.0)

    print(f"  Dataset drift detected:   {dataset_drift}")
    print(f"  Drifted features:         {drifted_count}/{total_count}  "
          f"(share={drift_share:.0%})")

    features = drift_result.get("features", {})
    drifted = [(n, d) for n, d in features.items() if d.get("drift_detected")]
    if drifted:
        print("\n  Drifted features:")
        for name, data in drifted:
            score = data.get("drift_score", "N/A")
            sev   = data.get("severity", "")
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)
            print(f"    {name:<20} score={score_str}  severity={sev}")


def _print_image_monitor_block(drift_result: dict) -> None:
    """Render the image drift detail block (monitor_image_batch() output format).

    Shows the per-channel Wasserstein scores from the drift measurement.
    If ISP scenario interpretation is available, also shows the closest
    scenario and its estimated accuracy impact.
    """
    overall       = drift_result.get("overall", {})
    drift_score   = overall.get("drift_score")
    dataset_drift = overall.get("dataset_drift_detected", False)
    method        = drift_result.get("method", "statistical")
    batch_info    = drift_result.get("batch", {})
    ref_info      = drift_result.get("reference", {})
    batch_size    = batch_info.get("size", "?")
    ref_size      = ref_info.get("size", "?")

    print(f"  Method:              {method}")
    print(f"  Batch / reference:   {batch_size} / {ref_size} images")
    if drift_score is not None:
        print(f"  Drift score:         {drift_score:.6f}")
    print(f"  Drift detected:      {dataset_drift}")

    # Per-channel breakdown (statistical method only)
    channels = drift_result.get("channels") or {}
    if channels:
        print("\n  Per-channel Wasserstein scores:")
        print(f"  {'Channel':<10} {'Score':>12}  {'Severity':<10}")
        print(f"  {'-'*10} {'-'*12}  {'-'*10}")
        for ch, data in channels.items():
            score = data.get("drift_score", 0)
            sev   = data.get("severity", "")
            print(f"  {ch:<10} {score:>12.6f}  {sev:<10}")

    # ISP scenario interpretation (only present for raw-image pipelines
    # where --drift-scenarios-dir was passed to monitor-drift-image)
    scenario_interp = drift_result.get("scenario_interpretation")
    if scenario_interp:
        name = scenario_interp.get("matched_scenario", "unknown")
        desc = scenario_interp.get("description", "")
        dist = scenario_interp.get("match_distance", 0)
        conf = scenario_interp.get("confidence", "")
        drop = scenario_interp.get("estimated_accuracy_drop")

        print(f"\n  Closest ISP scenario: {name}")
        if desc:
            print(f"  Description:          {desc}")
        print(f"  Match distance:       {dist:.4f}  (confidence: {conf})")
        if drop is not None:
            print(f"  Est. accuracy drop:   {drop:+.4f}")
