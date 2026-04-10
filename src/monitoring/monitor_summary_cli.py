"""CLI for summarizing drift monitoring history over time.

Reads all JSON monitoring results from a drift monitoring output directory
and prints a chronological trend table — drift scores, severities, and batch
sizes over time. Works for both tabular (monitor-drift) and image
(monitor-drift-image) monitoring results.

Usage — tabular model:
    monitor-summary --model-name iris_rf

Usage — image model:
    monitor-summary --output-dir outputs/drift_monitoring/images

Usage — custom directory:
    monitor-summary --output-dir path/to/my/monitoring/results

Output:
    A timestamped table showing drift score, severity, batch size and
    drift detection status for each stored monitoring run. A trailing
    summary line shows the trend direction.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_SEVERITY_ORD: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize drift monitoring history from stored JSON results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=(
            "Model name used by monitor-drift (tabular). "
            "Resolves to outputs/drift_monitoring/<model-name>/. "
            "Use --output-dir for image monitoring or custom paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory containing monitoring JSON result files. "
            "Defaults to outputs/drift_monitoring/images/ when --model-name is not given."
        ),
    )
    parser.add_argument(
        "--last",
        type=int,
        default=None,
        metavar="N",
        help="Show only the last N runs. Default: show all.",
    )
    return parser.parse_args()


def _load_results(output_dir: Path) -> list[dict]:
    """Load all monitoring result JSON files from a directory.

    Skips *_decision.json files (those are user decision records, not results).
    Returns results sorted by their generated_at timestamp ascending.
    """
    results = []
    for path in sorted(output_dir.glob("*.json")):
        if path.name.endswith("_decision.json"):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            # Must have an 'overall' section to be a monitoring result
            if "overall" not in data:
                continue
            data["_source_file"] = path.name
            results.append(data)
        except Exception as e:
            logger.warning("Skipping '%s': %s", path.name, e)

    # Sort by generated_at if present, else by filename
    results.sort(key=lambda r: r.get("generated_at", r["_source_file"]))
    return results


def _fmt_score(score) -> str:
    if score is None:
        return "   N/A  "
    try:
        return f"{float(score):.6f}"
    except (TypeError, ValueError):
        return str(score)


def _fmt_severity(sev: str) -> str:
    return sev.upper().ljust(6)


def _trend_label(severities: list[str]) -> str:
    """Compute a simple trend label from the last 5 severity values."""
    if len(severities) < 2:
        return "insufficient data"
    window = severities[-5:]
    ords = [_SEVERITY_ORD.get(s, 0) for s in window]
    if ords[-1] > ords[0]:
        return "WORSENING"
    if ords[-1] < ords[0]:
        return "IMPROVING"
    return "STABLE"


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    args = _parse_args()

    # --- Resolve output directory ---
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.model_name:
        output_dir = Path("outputs/drift_monitoring") / args.model_name
    else:
        output_dir = Path("outputs/drift_monitoring/images")

    if not output_dir.exists():
        print(f"No monitoring results found at '{output_dir}'.")
        print("Run monitor-drift or monitor-drift-image first.")
        sys.exit(0)

    results = _load_results(output_dir)

    if not results:
        print(f"No drift monitoring result files found in '{output_dir}'.")
        sys.exit(0)

    if args.last:
        results = results[-args.last:]

    # --- Print table ---
    print("\n" + "=" * 80)
    print(f"  DRIFT MONITORING SUMMARY  —  {output_dir}")
    print("=" * 80)
    print(f"\n  {'Timestamp':<22}  {'Type':<18}  {'Score':>10}  {'Severity':<8}  "
          f"{'Detected':<9}  {'Batch':>6}  {'Method':<14}")
    print("  " + "-" * 76)

    severities: list[str] = []

    for r in results:
        overall = r.get("overall", {})
        generated_at = r.get("generated_at", "")[:19].replace("T", " ")
        drift_type = r.get("drift_type", r.get("drift_type", "?"))[:18]
        score = overall.get("drift_score") or overall.get("drift_share")
        severity = overall.get("severity", "?")
        detected = overall.get("dataset_drift_detected")
        method = r.get("method", "?")[:14]

        # Batch size: image reports use batch.size, tabular use current_dataset.row_count
        batch_info = r.get("batch", {})
        if isinstance(batch_info, dict) and "size" in batch_info:
            batch_size = batch_info["size"]
        else:
            batch_size = r.get("current_dataset", {}).get("row_count", "?")

        detected_str = "YES" if detected else ("NO" if detected is False else "?")
        severities.append(severity if severity != "?" else "low")

        print(f"  {generated_at:<22}  {drift_type:<18}  {_fmt_score(score):>10}  "
              f"{_fmt_severity(severity):<8}  {detected_str:<9}  {str(batch_size):>6}  {method:<14}")

    print("  " + "-" * 76)

    trend = _trend_label(severities)
    total = len(results)
    drifted = sum(1 for r in results if r.get("overall", {}).get("dataset_drift_detected"))
    high_count = sum(1 for s in severities if s == "high")
    medium_count = sum(1 for s in severities if s == "medium")

    print(f"\n  Runs shown:    {total}")
    print(f"  Drift detected:{drifted} / {total}  ({100 * drifted // total if total else 0}%)")
    print(f"  Severity breakdown: HIGH={high_count}  MEDIUM={medium_count}  LOW={total - high_count - medium_count}")
    print(f"  Trend (last 5 runs): {trend}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
