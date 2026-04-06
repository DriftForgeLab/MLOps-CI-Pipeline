"""CLI entry point for offline batch drift monitoring.

Usage:
    monitor-drift --batch-csv data/new_batch.csv --model-name iris_rf \\
                  --config src/config/pipeline_tabular.yaml
    monitor-drift --batch-csv data/new_batch.csv --model-name iris_rf \\
                  --config src/config/pipeline_tabular.yaml \\
                  --drift-config src/config/drift.yaml
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

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline batch drift monitoring against a model's reference data."
    )
    parser.add_argument(
        "--batch-csv",
        type=str,
        required=True,
        help="Path to the new data batch CSV file.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name whose reference data to compare against.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset (e.g. 'iris').",
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        required=True,
        help="Dataset version ID (content-addressed hash).",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to pipeline config file (e.g., src/config/pipeline_tabular.yaml).",
    )
    parser.add_argument(
        "--drift-config",
        type=str,
        default=None,
        help="Path to drift config override. Defaults to the drift config in pipeline config.",
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

    # Load configs
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load pipeline config: %s", e)
        sys.exit(1)

    drift_config_path = Path(args.drift_config) if args.drift_config else Path(config.configs.drift)
    try:
        drift_config = load_drift_config(drift_config_path.resolve())
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load drift config: %s", e)
        sys.exit(1)

    # Load batch data
    batch_path = Path(args.batch_csv)
    if not batch_path.exists():
        logger.error("Batch CSV not found: %s", batch_path)
        sys.exit(1)
    batch_df = pd.read_csv(batch_path)
    logger.info("Loaded batch data: %s (%d rows)", batch_path, len(batch_df))

    # Load reference data
    model_info = {
        "dataset_name": args.dataset_name,
        "dataset_version_id": args.dataset_version,
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

    # Run batch monitoring
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
        logger.info("Batch too small for drift analysis — no output written.")
        sys.exit(0)

    # Always print summary (even in non-interactive mode for CLI)
    if not sys.stdin.isatty():
        print_drift_summary(drift_result)

    # Save results to structured JSON log
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs/drift_monitoring") / args.model_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{timestamp}.json"
    atomic_write_json(output_path, drift_result)
    logger.info("Monitoring results written to %s", output_path)

    sys.exit(0)


if __name__ == "__main__":
    main()
