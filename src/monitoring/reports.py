"""Drift reporting: CLI summary, JSON persistence, and HTML export.

Presentation layer for the standard drift result schema produced by
``src.drift.interpret.build_drift_result``. All functions are stateless:
they either write to stdout, to a filesystem path, or delegate to an
Evidently report object — no pipeline state is mutated here.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io import atomic_write_json

_SEPARATOR = "=" * 60


def print_drift_summary(drift_result: dict) -> None:
    """Print a human-readable drift monitoring summary to stdout.

    Args:
        drift_result: Standard drift result dict as returned by
                      ``build_drift_result``.
    """
    reference = drift_result["reference_dataset"]
    current = drift_result["current_dataset"]
    overall = drift_result["overall"]
    features = drift_result["features"]

    print("\n" + _SEPARATOR)
    print("  DRIFT MONITORING RESULT")
    print(_SEPARATOR)

    # --- Dataset metadata ---
    method = drift_result.get("method", "")
    if method:
        print(f"\n  Method:    {method}")
    print(
        f"  Reference: {reference['source']} split "
        f"({reference['row_count']} rows, {reference['feature_count']} features)"
    )
    print(
        f"  Batch:     {current['source']} "
        f"({current['row_count']} rows, {current['feature_count']} features)"
    )

    # --- Overall drift status ---
    detected_label = "DETECTED" if overall["dataset_drift_detected"] else "NOT DETECTED"
    severity_label = overall["severity"].upper()
    drift_score = overall.get("drift_score", overall.get("drift_share", 0))
    drift_share_pct = drift_score * 100.0
    print(f"\n  Drift score:      {drift_score:.4f}  ({drift_share_pct:.1f}% of features)")
    print(f"  Drift detected:   {detected_label}")
    print(f"  Severity:         {severity_label}")
    print(
        f"  Drifted features: {overall['drifted_feature_count']} / "
        f"{overall['total_feature_count']}"
    )

    # --- Per-feature breakdown ---
    print("\n  Per-feature breakdown:")
    for feat_name, feat_data in features.items():
        status = "DRIFTED" if feat_data["drift_detected"] else "OK     "
        print(
            f"    {feat_name:<20} "
            f"{feat_data['stattest_name']:<8} "
            f"p={feat_data['drift_score']:.4f}   "
            f"{status}   "
            f"severity: {feat_data['severity']}"
        )

    print("\n" + _SEPARATOR)


def save_drift_report_json(drift_result: dict, output_dir: Path) -> Path:
    """Persist the drift result as JSON to ``output_dir / drift_report.json``.

    Uses ``atomic_write_json`` so NaN/Inf values are sanitized and the write
    is atomic (write-to-temp + rename). Creates ``output_dir`` if missing.

    Args:
        drift_result: Standard drift result dict.
        output_dir:   Directory to write into.

    Returns:
        Path to the written JSON file.
    """
    path = Path(output_dir) / "drift_report.json"
    atomic_write_json(path, drift_result)
    return path


def save_drift_report_html(evidently_report: Any, output_dir: Path) -> Path:
    """Save the Evidently report HTML dashboard to ``output_dir / drift_report.html``.

    Creates ``output_dir`` if missing and delegates rendering to the
    Evidently report's ``save_html`` method.

    Args:
        evidently_report: An Evidently Report object with a ``save_html`` method.
        output_dir:       Directory to write into.

    Returns:
        Path to the written HTML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "drift_report.html"
    evidently_report.save_html(str(path))
    return path
