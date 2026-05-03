# =============================================================================
# src/monitoring/prepare_drift_tabular_cli.py — prepare-drift-training-tabular
# =============================================================================
# Prepares a labeled drifted CSV batch for drift-adaptive retraining by:
#
#   1. Validating the drifted CSV schema against the current dataset.
#   2. Splitting rows into a training portion and a held-out evaluation set.
#   3. Evaluating the current Production model on the holdout → baseline metrics.
#   4. Saving the holdout CSV and baseline metrics for post-retrain comparison.
#   5. Appending the training portion to data/raw/<dataset>/data.csv, which
#      triggers a new content-addressed version_id on the next pipeline run.
#
# Incoming drifted batch convention
# ----------------------------------
# The drifted CSV must have the same column schema as the original dataset:
#   - Same feature column names (same order is not required but is recommended)
#   - Same target column name
#   - Raw (un-preprocessed) values — the preprocessing pipeline is applied
#     automatically when evaluating and during the next pipeline run
#
# Example:
#   data/batches/tabular/drifted_housing.csv  (8 features + MedHouseVal target)
#
# Usage:
#   prepare-drift-training-tabular \
#       --drifted-csv data/batches/tabular/drifted_housing.csv \
#       --config src/config/pipeline_tabular_regression.yaml
#
# After running this command:
#   - Holdout rows are saved to data/evaluation/drifted_holdout/<dataset>/holdout.csv
#   - Baseline metrics are saved alongside the holdout.
#   - Training rows are appended to data/raw/<dataset>/data.csv.
#
# Next step:
#   run-pipeline --config src/config/pipeline_tabular_regression.yaml
#
# The pipeline will automatically evaluate the retrained model on the holdout
# and print a before/after performance comparison at the end of the run.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a labeled drifted CSV batch for drift-adaptive retraining. "
            "Splits it into a training portion (appended to the raw dataset) and a "
            "held-out evaluation set (used to compare performance before/after retraining)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--drifted-csv",
        type=str,
        required=True,
        help=(
            "Path to the labeled drifted CSV batch. Must have the same column schema "
            "(feature names + target column) as the original dataset. Values should be "
            "raw (un-preprocessed) — the pipeline applies preprocessing automatically. "
            "Conventional location: data/batches/tabular/drifted.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the pipeline config YAML (e.g. src/config/pipeline_tabular_regression.yaml).",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.3,
        help=(
            "Fraction of drifted rows to hold out for evaluation (default: 0.3). "
            "The remainder is appended to the training dataset."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help=(
            "Explicit dataset version ID (content-addressed hash). "
            "Auto-resolved to the most recent preprocessed version if not given."
        ),
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
    from src.config.loader import load_config

    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load pipeline config: %s", e)
        sys.exit(1)

    if config.task_type not in ("classification", "regression"):
        logger.error(
            "prepare-drift-training-tabular only supports task_type 'classification' "
            "or 'regression', but config has task_type='%s'. "
            "For image pipelines use prepare-drift-training instead.",
            config.task_type,
        )
        sys.exit(1)

    # --- Resolve dataset version ---
    from src.data.prepare_batch import resolve_latest_version

    processed_dir = Path(config.data.processed)
    dataset_name = config.dataset

    if args.dataset_version:
        version_id = args.dataset_version
        logger.info("Using explicit dataset version: %s", version_id)
    else:
        try:
            version_id = resolve_latest_version(processed_dir, dataset_name)
        except FileNotFoundError as e:
            logger.error(
                "Could not resolve dataset version: %s\n"
                "Run 'run-pipeline --config %s' at least once to create a processed version.",
                e, args.config,
            )
            sys.exit(1)

    # --- Load feature contract (input features, target, pipeline) ---
    from src.data.tabular_drift_adaptation import (
        load_feature_contract,
        split_holdout_tabular,
        evaluate_on_holdout_tabular,
        save_holdout_csv,
        save_baseline_metrics,
        append_training_rows_to_dataset,
        load_baseline_metrics,
    )

    try:
        feature_contract = load_feature_contract(processed_dir, dataset_name, version_id)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load feature contract: %s", e)
        sys.exit(1)

    input_features: list[str] = feature_contract["input_features"]
    target: str = feature_contract["target"]

    # --- Load and validate drifted CSV ---
    drifted_csv = Path(args.drifted_csv)
    if not drifted_csv.exists():
        logger.error("Drifted CSV not found: %s", drifted_csv)
        sys.exit(1)

    import pandas as pd

    try:
        drifted_df = pd.read_csv(drifted_csv)
    except Exception as e:
        logger.error("Failed to read drifted CSV '%s': %s", drifted_csv, e)
        sys.exit(1)

    required_cols = set(input_features) | {target}
    missing_cols = required_cols - set(drifted_df.columns)
    if missing_cols:
        logger.error(
            "Drifted CSV '%s' is missing required columns: %s\n"
            "Expected columns: %s",
            drifted_csv, sorted(missing_cols), sorted(required_cols),
        )
        sys.exit(1)

    if len(drifted_df) < 2:
        logger.error(
            "Drifted CSV has only %d row(s) — need at least 2 to split into "
            "training and holdout portions.",
            len(drifted_df),
        )
        sys.exit(1)

    # --- Load current Production model ---
    model_path = Path("artifacts/runs") / version_id / "model" / "model.joblib"
    if not model_path.exists():
        logger.error(
            "Model artifact not found at '%s'. "
            "Run the pipeline at least once so a trained model exists.",
            model_path,
        )
        sys.exit(1)

    try:
        import joblib
        model = joblib.load(model_path)
        logger.info("Loaded Production model: %s", model_path)
    except Exception as e:
        logger.error("Failed to load model from '%s': %s", model_path, e)
        sys.exit(1)

    # --- Stratified / random split ---
    holdout_ratio = max(0.0, min(args.holdout_ratio, 0.9))
    try:
        train_df, holdout_df = split_holdout_tabular(
            drifted_df,
            target=target,
            task_type=config.task_type,
            holdout_ratio=holdout_ratio,
            seed=config.random_seed,
        )
    except ValueError as e:
        logger.error("Split failed: %s", e)
        sys.exit(1)

    n_total = len(drifted_df)
    n_train = len(train_df)
    n_holdout = len(holdout_df)

    # --- Check for existing holdout and warn ---
    holdout_dir = Path(config.data.evaluation) / "drifted_holdout" / dataset_name
    existing_baseline = load_baseline_metrics(holdout_dir)
    if existing_baseline is not None:
        generated = existing_baseline.get("generated_at", "unknown")
        print()
        print("WARNING: A holdout set already exists at:")
        print(f"  {holdout_dir}")
        print(f"  Baseline created: {generated}")
        print()
        try:
            answer = input("Overwrite the existing holdout and baseline? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            answer = "n"
        if answer not in ("y", "yes"):
            print("Aborted — existing holdout unchanged.")
            sys.exit(0)
        import shutil
        shutil.rmtree(holdout_dir)
        logger.info("Removed existing holdout directory.")

    # --- Save holdout CSV and marker ---
    save_holdout_csv(holdout_df, holdout_dir)
    (holdout_dir / ".drift_prepared").touch()

    # --- Evaluate Production model on holdout → baseline ---
    if n_holdout == 0:
        logger.warning(
            "No holdout rows. Before/after comparison will not be available."
        )
        baseline_metrics = None
    else:
        try:
            baseline_metrics = evaluate_on_holdout_tabular(
                model, holdout_df, feature_contract, config.task_type
            )
            save_baseline_metrics(baseline_metrics, holdout_dir)
        except Exception as e:
            logger.error("Baseline evaluation failed: %s", e, exc_info=True)
            sys.exit(1)

    # --- Append training rows to raw dataset CSV ---
    raw_csv_path = Path(config.data.raw) / dataset_name / "data.csv"
    try:
        append_training_rows_to_dataset(train_df[list(drifted_df.columns)], raw_csv_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to append training rows: %s", e)
        sys.exit(1)

    # --- Print summary ---
    _print_summary(
        drifted_csv=drifted_csv,
        holdout_dir=holdout_dir,
        raw_csv_path=raw_csv_path,
        n_total=n_total,
        n_train=n_train,
        n_holdout=n_holdout,
        baseline_metrics=baseline_metrics,
        task_type=config.task_type,
        config_path=args.config,
    )


def _print_summary(
    drifted_csv: Path,
    holdout_dir: Path,
    raw_csv_path: Path,
    n_total: int,
    n_train: int,
    n_holdout: int,
    baseline_metrics: dict | None,
    task_type: str,
    config_path: str,
) -> None:
    print()
    print("=" * 62)
    print("  DRIFT TRAINING PREPARATION (TABULAR) — COMPLETE")
    print("=" * 62)
    print()
    print(f"  Drifted rows loaded:     {n_total}")
    print(f"  -> Added to training:    {n_train}")
    print(f"  -> Held out for eval:    {n_holdout}")
    print()

    if baseline_metrics:
        m = baseline_metrics
        print("  Baseline (Production model on held-out drifted rows):")
        if task_type == "classification":
            print(f"    Accuracy:   {m.get('accuracy', 'N/A')}")
            print(f"    F1 score:   {m.get('f1_score', 'N/A')}")
            print(f"    Precision:  {m.get('precision', 'N/A')}")
            print(f"    Recall:     {m.get('recall', 'N/A')}")
        else:
            print(f"    R2:         {m.get('r2', 'N/A')}")
            print(f"    MAE:        {m.get('mae', 'N/A')}")
            print(f"    RMSE:       {m.get('rmse', 'N/A')}")
        print(f"    N samples:  {m.get('n_samples', 'N/A')}")
    else:
        print("  Baseline: N/A (no holdout rows)")

    print()
    print("  Holdout saved to:")
    print(f"    {holdout_dir}")
    print()
    print("  Training rows appended to:")
    print(f"    {raw_csv_path}")
    print()
    print("  NEXT STEP — retrain the model:")
    print(f"    run-pipeline --config {config_path}")
    print()
    print("  The pipeline will automatically evaluate the retrained model on")
    print("  the holdout and print a before/after comparison at the end.")
    print()
    print("=" * 62)
