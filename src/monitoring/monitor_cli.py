"""CLI entry point for online tabular drift monitoring.

Compares a new production batch (CSV) against the training reference (train.csv)
and reports per-feature drift using statistical tests (KS / chi-square).

    reference = stored training data (train.csv from the trained model version)
    batch     = new incoming production data

Minimal usage (auto-resolves dataset name and latest version from config):
    monitor-drift \\
        --batch-csv data/new_batch.csv \\
        --model-name iris_rf \\
        --config src/config/pipeline_tabular.yaml

Explicit dataset version:
    monitor-drift \\
        --batch-csv data/new_batch.csv \\
        --model-name iris_rf \\
        --config src/config/pipeline_tabular.yaml \\
        --dataset-version abc123hash

Output:
    <output_dir>/<timestamp>.json           — monitoring result
    <output_dir>/<timestamp>_decision.json  — user decision (if interactive)
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config.loader import load_config, load_drift_config
from src.common.io import atomic_write_json
from src.monitoring.drift import load_reference_for_model, monitor_batch
from src.monitoring.reports import print_drift_summary
from src.monitoring.drift_decision import request_drift_decision
from src.drift.interpret import _SEVERITY_ORD
from src.data.prepare_batch import resolve_latest_version

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run online tabular batch drift monitoring against the training reference."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--batch-csv",
        type=str,
        required=True,
        help="Path to the new production data batch CSV file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name (used for output directory naming).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config YAML (e.g. src/config/pipeline_tabular.yaml).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Dataset name. Defaults to the 'dataset' field in the pipeline config."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help=(
            "Dataset version ID (content-addressed hash). "
            "Defaults to the most recently preprocessed version."
        ),
    )
    parser.add_argument(
        "--drift-config",
        type=str,
        default=None,
        help="Optional path to a drift config override YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for monitoring output. Defaults to outputs/drift_monitoring/<model-name>/.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = _parse_args()

    # --- Load pipeline config ---
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load pipeline config: %s", e)
        sys.exit(1)

    drift_config_path = (
        Path(args.drift_config) if args.drift_config else Path(config.configs.drift)
    )
    try:
        drift_config = load_drift_config(drift_config_path.resolve())
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load drift config: %s", e)
        sys.exit(1)

    # --- Load batch ---
    batch_path = Path(args.batch_csv)
    if not batch_path.exists():
        logger.error("Batch CSV not found: %s", batch_path)
        sys.exit(1)
    batch_df = pd.read_csv(batch_path)
    logger.info("Loaded batch: %s  (%d rows)", batch_path.name, len(batch_df))

    # --- Resolve dataset name and version ---
    dataset_name = args.dataset_name or config.dataset
    processed_dir = Path(config.data.processed)

    if args.dataset_version:
        version_id = args.dataset_version
        logger.info("Dataset: %s  version: %s (explicit)", dataset_name, version_id)
    else:
        try:
            version_id = resolve_latest_version(processed_dir, dataset_name)
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)
        logger.info(
            "Dataset: %s  version: %s (auto-resolved, newest)",
            dataset_name, version_id,
        )

    # --- Load reference data ---
    model_info = {
        "dataset_name": dataset_name,
        "dataset_version_id": version_id,
    }
    try:
        reference_df, feature_map = load_reference_for_model(
            model_info=model_info,
            processed_dir=config.data.processed,
            reference_source=drift_config.reference_source,
        )
    except FileNotFoundError as e:
        logger.error("Reference data not found: %s", e)
        sys.exit(1)

    # --- Run drift analysis ---
    try:
        drift_result = monitor_batch(
            batch_df=batch_df,
            reference_df=reference_df,
            feature_map=feature_map,
            drift_config=drift_config,
            model_name=args.model_name,
            interactive=sys.stdin.isatty(),
        )
    except ValueError as e:
        logger.error("Drift monitoring failed: %s", e)
        sys.exit(1)

    if drift_result is None:
        logger.info(
            "Batch has fewer than min_batch_size=%d rows — no drift analysis performed.",
            drift_config.monitoring.min_batch_size,
        )
        sys.exit(0)

    if not sys.stdin.isatty():
        print_drift_summary(drift_result)

    # --- Save results ---
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/drift_monitoring") / args.model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{timestamp}.json"
    atomic_write_json(output_path, drift_result)
    logger.info("Drift result written to %s", output_path)

    # --- Decision gate ---
    overall_severity = drift_result["overall"]["severity"]
    alert_severity = drift_config.monitoring.alert_severity
    if sys.stdin.isatty() and (
        _SEVERITY_ORD.get(overall_severity, 0) >= _SEVERITY_ORD.get(alert_severity, 0)
    ):
        decision = request_drift_decision(
            drift_result,
            is_image_isp=False,
            drift_report_linked=output_path.name,
        )
        if decision is not None:
            decision_path = output_dir / f"{timestamp}_decision.json"
            atomic_write_json(decision_path, decision.to_dict())
            logger.info("Drift decision written to %s", decision_path)

    sys.exit(0)


if __name__ == "__main__":
    main()
