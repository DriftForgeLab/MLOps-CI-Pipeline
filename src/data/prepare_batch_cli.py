"""CLI for preprocessing new production images into a batch NPZ for drift monitoring.

This is the first step in the drift monitoring workflow. Run this command
whenever you have collected new images and want to check for data drift.

Usage — minimal (auto-resolves dataset name and latest version from config):
    prepare-image-batch \\
        --input-dir data/incoming/my_new_images \\
        --config src/config/pipeline_image_cnn.yaml

Usage — raw DNG images:
    prepare-image-batch \\
        --input-dir data/incoming/drone_batch_001 \\
        --config src/config/pipeline_image_raw.yaml

Usage — explicit dataset version:
    prepare-image-batch \\
        --input-dir data/incoming/my_new_images \\
        --config src/config/pipeline_image_cnn.yaml \\
        --dataset-version abc123hash

Usage — explicit output path:
    prepare-image-batch \\
        --input-dir data/incoming/my_new_images \\
        --config src/config/pipeline_image_cnn.yaml \\
        --output-npz data/batches/my_batch.npz

After running this command, pass the output NPZ to monitor-drift-image:
    monitor-drift-image \\
        --batch-npz data/batches/my_batch.npz \\
        --config src/config/pipeline_image_cnn.yaml

Input directory structure:
    The input directory can be flat (all images in one folder) or ImageFolder
    (class subdirectories). Class labels are not required — only pixel data
    is used for drift analysis.

    Flat:
        data/incoming/batch/
            frame_001.jpg
            frame_002.jpg

    ImageFolder:
        data/incoming/batch/
            hund/frame_001.jpg
            katt/frame_002.jpg

Output:
    A single NPZ file with an 'X' array of shape (N, H, W, C), preprocessed
    exactly the same way as the training data.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config.loader import load_config, load_preprocessing_config
from src.common.io import atomic_write_npz
from src.data.prepare_batch import (
    load_training_stats,
    prepare_image_batch,
    resolve_latest_version,
)

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess new production images into a batch NPZ file "
            "for use with monitor-drift-image."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help=(
            "Directory containing new production images. "
            "Flat or ImageFolder structure — class labels are ignored."
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help=(
            "Path to the pipeline config YAML used during training "
            "(e.g. src/config/pipeline_image_cnn.yaml or pipeline_image_raw.yaml). "
            "Used to resolve dataset name, processed data dir, and preprocessing config."
        ),
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Dataset name. Defaults to the 'dataset' field in the pipeline config. "
            "Override only if you trained on a different dataset name."
        ),
    )
    parser.add_argument(
        "--dataset-version",
        type=str,
        default=None,
        help=(
            "Version ID (hash) of the training run to use as normalization reference. "
            "Defaults to the most recently preprocessed version."
        ),
    )
    parser.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help=(
            "Output path for the batch NPZ file. "
            "Defaults to data/batches/<timestamp>.npz."
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
    config_path = Path(args.config).resolve()
    try:
        config = load_config(config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load pipeline config: %s", e)
        sys.exit(1)

    # --- Resolve dataset name ---
    dataset_name = args.dataset_name or config.dataset
    logger.info("Dataset: %s", dataset_name)

    # --- Resolve dataset version ---
    processed_dir = Path(config.data.processed)
    if args.dataset_version:
        version_id = args.dataset_version
        logger.info("Version: %s (explicit)", version_id)
    else:
        try:
            version_id = resolve_latest_version(processed_dir, dataset_name)
        except FileNotFoundError as e:
            logger.error("%s", e)
            sys.exit(1)
        logger.info("Version: %s (auto-resolved, newest)", version_id)

    # --- Load training normalization stats ---
    try:
        norm_mean, norm_std, expected_formats = load_training_stats(
            processed_dir, dataset_name, version_id
        )
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(1)

    if norm_mean is not None:
        logger.info(
            "Normalization stats loaded (mean channels: %s)",
            [round(v, 4) for v in norm_mean],
        )
    else:
        logger.info("No normalization stats found — batch will not be z-score normalized.")

    # --- Load preprocessing config ---
    prep_config_path = Path(config.configs.preprocessing).resolve()
    try:
        prep_config = load_preprocessing_config(prep_config_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Failed to load preprocessing config: %s", e)
        sys.exit(1)

    is_raw = bool(
        prep_config.image and prep_config.image.raw_input and prep_config.image.isp
    )
    pipeline_type = "raw DNG (ISP)" if is_raw else "standard JPG/PNG"
    logger.info("Pipeline type: %s", pipeline_type)

    # --- Preprocess images ---
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        sys.exit(1)

    logger.info("Preprocessing images from '%s'...", input_dir)
    try:
        X = prepare_image_batch(
            input_dir=input_dir,
            prep_config=prep_config,
            norm_mean=norm_mean,
            norm_std=norm_std,
            expected_formats=expected_formats,
        )
    except (ValueError, ImportError) as e:
        logger.error("Preprocessing failed: %s", e)
        sys.exit(1)

    logger.info("Batch shape: %s  dtype: %s", X.shape, X.dtype)

    # --- Save output NPZ ---
    if args.output_npz:
        output_path = Path(args.output_npz)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = Path("data/batches") / f"{timestamp}.npz"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Store version_id so monitor-drift-image can verify batch and reference match
    atomic_write_npz(output_path, X=X, version_id=np.array(version_id))

    print("\n" + "=" * 60)
    print("  BATCH PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n  Images processed: {X.shape[0]}")
    print(f"  Image shape:      {X.shape[1:]}")
    print(f"  Pipeline type:    {pipeline_type}")
    print(f"  Dataset version:  {version_id}")
    print(f"\n  Output: {output_path}")
    print("\n  Next step — run drift analysis:")
    print(f"\n    monitor-drift-image --batch-npz {output_path} --config {args.config}")
    print("\n" + "=" * 60 + "\n")

    logger.info("Batch NPZ written to %s", output_path)


if __name__ == "__main__":
    main()
