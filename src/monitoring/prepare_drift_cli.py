# =============================================================================
# src/monitoring/prepare_drift_cli.py — prepare-drift-training CLI
# =============================================================================
# Prepares a set of labeled drifted images for drift-adaptive fine-tuning by:
#
#   1. Scanning a labeled (ImageFolder) drifted image directory.
#   2. Evaluating the current Production model on the held-out split → baseline.
#   3. Splitting drifted images into a training portion and a held-out set.
#   4. Copying the training portion into the raw dataset (triggers a new
#      content-addressed version on the next pipeline run).
#   5. Saving baseline metrics so the post-fine-tune pipeline step can
#      compare before and after automatically.
#
# Project path convention
# -----------------------
# Incoming drifted production batches live in:
#   data/batches/images/<batch_name>/
#
# To use a batch for drift-adaptive fine-tuning, the images must be organised
# into class subdirectories (ImageFolder layout) matching the dataset's class
# names exactly. For example, for the sample_images dataset:
#
#   data/batches/images/drifted/
#     cats/            ← must match class name in data/raw/sample_images/images/
#       cat_000.png
#     dogs/
#       dog_000.png
#
# The monitoring CLI (monitor-drift-image) already works with both flat and
# ImageFolder batch directories — adding class subdirs does not break monitoring.
#
# Usage:
#   prepare-drift-training \
#       --drifted-dir data/batches/images/drifted \
#       --config src/config/pipeline_image.yaml
#
# After running this command:
#   - Holdout images are saved to data/evaluation/drifted_holdout/<dataset>/
#   - Baseline metrics are saved alongside the holdout images.
#   - Training images are added to data/raw/<dataset>/images/<class>/.
#
# Next step:
#   run-pipeline --config src/config/pipeline_image.yaml --fine-tune
#
# The pipeline will automatically evaluate the fine-tuned model on the holdout
# and print a before/after comparison at the end of the run.
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
            "Prepare labeled drifted images for drift-adaptive fine-tuning. "
            "Splits them into a training portion (added to the dataset) and a "
            "held-out evaluation set (used to compare performance before/after fine-tuning)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--drifted-dir",
        type=str,
        required=True,
        help=(
            "Path to a directory of labeled drifted images in ImageFolder layout: "
            "one subdirectory per class, images inside. "
            "Subdirectory names must match the dataset's class names exactly. "
            "The conventional location is data/batches/images/<batch_name>/ — "
            "the same directory used with monitor-drift-image, but with class "
            "subdirectories added (e.g. data/batches/images/drifted/cats/, "
            "data/batches/images/drifted/dogs/). "
            "The monitoring CLI already scans recursively, so adding class subdirs "
            "does not break monitor-drift-image on the same directory."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the pipeline config YAML (e.g. src/config/pipeline_image.yaml).",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.3,
        help=(
            "Fraction of drifted images per class to hold out for evaluation "
            "(default: 0.3). The remainder is added to the training dataset."
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

    # --- Load configs ---
    from src.config.loader import load_config, load_preprocessing_config

    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load pipeline config: %s", e)
        sys.exit(1)

    try:
        prep_config = load_preprocessing_config(Path(config.configs.preprocessing))
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load preprocessing config: %s", e)
        sys.exit(1)

    if prep_config.image is None:
        logger.error(
            "Preprocessing config has no 'image' section. "
            "prepare-drift-training only supports image pipelines."
        )
        sys.exit(1)

    # --- Resolve dataset version and load training stats ---
    from src.data.prepare_batch import resolve_latest_version, load_training_stats
    from src.data.drift_adaptation import (
        scan_labeled_drifted_dir,
        split_holdout_stratified,
        copy_training_images_to_dataset,
        save_holdout_images,
        evaluate_on_holdout_dir,
        load_class_to_index,
        save_baseline_metrics,
        load_baseline_metrics,
        save_normalization_stats,
    )

    processed_dir = Path(config.data.processed)
    dataset_name = config.dataset

    if args.dataset_version:
        version_id = args.dataset_version
        logger.info("Using explicit dataset version: %s", version_id)
    else:
        try:
            version_id = resolve_latest_version(processed_dir, dataset_name)
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)

    try:
        norm_mean, norm_std, _ = load_training_stats(processed_dir, dataset_name, version_id)
    except FileNotFoundError as e:
        logger.error("Training stats not found: %s", e)
        sys.exit(1)

    try:
        class_to_index = load_class_to_index(processed_dir, dataset_name, version_id)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Class mapping not found: %s", e)
        sys.exit(1)

    # --- Scan drifted images ---
    drifted_dir = Path(args.drifted_dir)
    if not drifted_dir.exists():
        logger.error("Drifted image directory not found: %s", drifted_dir)
        sys.exit(1)

    try:
        class_images = scan_labeled_drifted_dir(drifted_dir)
    except ValueError as e:
        logger.error("%s", e)
        sys.exit(1)

    # --- Validate class names match the dataset ---
    known_classes = set(class_to_index.keys())
    drifted_classes = set(class_images.keys())
    unknown = drifted_classes - known_classes
    if unknown:
        logger.error(
            "Class mismatch: drifted subdirectory name(s) %s do not match any "
            "class in the dataset (known classes: %s). "
            "Rename the subdirectories in '%s' to match exactly.",
            sorted(unknown), sorted(known_classes), drifted_dir,
        )
        sys.exit(1)

    # --- Load current Production model ---
    model_pt = Path("artifacts/runs") / version_id / "model" / "model.pt"
    if not model_pt.exists():
        logger.error(
            "Model artifact not found at '%s'. "
            "Run the pipeline at least once so a trained model exists.",
            model_pt,
        )
        sys.exit(1)

    try:
        import torch
        model = torch.load(str(model_pt), weights_only=False, map_location="cpu")
        logger.info("Loaded Production model: %s", model_pt)
    except Exception as e:
        logger.error("Failed to load model from '%s': %s", model_pt, e)
        sys.exit(1)

    # --- Stratified split ---
    holdout_ratio = max(0.0, min(args.holdout_ratio, 0.9))
    train_images, holdout_images = split_holdout_stratified(
        class_images, holdout_ratio, seed=config.random_seed
    )

    n_train = sum(len(v) for v in train_images.values())
    n_holdout = sum(len(v) for v in holdout_images.values())
    n_total = n_train + n_holdout

    # --- Check whether a holdout already exists and warn the user ---
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
        import shutil as _shutil
        _shutil.rmtree(holdout_dir)
        logger.info("Removed existing holdout directory.")

    # --- Save holdout images and normalization stats ---
    # Normalization stats are saved alongside the holdout so both the baseline
    # evaluation (now) and the post-fine-tuning evaluation (later) use the
    # same scaling, even if the dataset version changes after adding drifted images.
    save_holdout_images(holdout_images, holdout_dir)
    save_normalization_stats(norm_mean, norm_std, holdout_dir)

    # --- Evaluate current model on holdout → baseline ---
    if n_holdout == 0:
        logger.warning(
            "No holdout images (all classes have only 1 image each). "
            "Before/after comparison will not be available. "
            "Continuing without saving baseline metrics."
        )
        baseline_metrics = None
    else:
        try:
            baseline_metrics = evaluate_on_holdout_dir(
                model, holdout_dir, prep_config, norm_mean, norm_std, class_to_index
            )
            save_baseline_metrics(baseline_metrics, holdout_dir)
        except Exception as e:
            logger.error("Baseline evaluation failed: %s", e, exc_info=True)
            sys.exit(1)

    # --- Copy training images into the raw dataset ---
    raw_images_dir = Path(config.data.raw) / dataset_name / "images"
    copy_training_images_to_dataset(train_images, raw_images_dir)

    # --- Print summary ---
    _print_summary(
        drifted_dir=drifted_dir,
        holdout_dir=holdout_dir,
        raw_images_dir=raw_images_dir,
        n_total=n_total,
        n_train=n_train,
        n_holdout=n_holdout,
        train_images=train_images,
        holdout_images=holdout_images,
        baseline_metrics=baseline_metrics,
        config_path=args.config,
    )


def _print_summary(
    drifted_dir: Path,
    holdout_dir: Path,
    raw_images_dir: Path,
    n_total: int,
    n_train: int,
    n_holdout: int,
    train_images: dict,
    holdout_images: dict,
    baseline_metrics: dict | None,
    config_path: str,
) -> None:
    print()
    print("=" * 62)
    print("  DRIFT TRAINING PREPARATION — COMPLETE")
    print("=" * 62)
    print()
    print(f"  Drifted images scanned:  {n_total}")
    print(f"  → Added to training:     {n_train}  ({_class_counts(train_images)})")
    print(f"  → Held out for eval:     {n_holdout}  ({_class_counts(holdout_images)})")
    print()

    if baseline_metrics:
        m = baseline_metrics
        print("  Baseline (Production model on held-out drifted images):")
        print(f"    Accuracy:   {m['accuracy']:.4f}")
        print(f"    F1 score:   {m['f1_score']:.4f}")
        print(f"    Precision:  {m['precision']:.4f}")
        print(f"    Recall:     {m['recall']:.4f}")
        print(f"    N samples:  {m['n_samples']}")
    else:
        print("  Baseline: N/A (no holdout images)")
    print()
    print("  Holdout saved to:")
    print(f"    {holdout_dir}")
    print()
    print("  Training images added to dataset:")
    print(f"    {raw_images_dir}")
    print()
    print("  NEXT STEP — run fine-tuning:")
    print(f"    run-pipeline --config {config_path} --fine-tune")
    print()
    print("  The pipeline will automatically evaluate the fine-tuned model")
    print("  on the holdout and print a before/after comparison at the end.")
    print()
    print("=" * 62)


def _class_counts(class_images: dict) -> str:
    return ", ".join(f"{cls}: {len(paths)}" for cls, paths in class_images.items())
