# =============================================================================
# src/data/drift_adaptation.py — Drift-adaptive fine-tuning data preparation
# =============================================================================
# Provides pure data operations for the drift-adaptive fine-tuning workflow:
#
#   1. Scan a labeled (ImageFolder) directory of drifted images.
#   2. Split the images into a training portion and a held-out evaluation set.
#   3. Copy the training portion into the raw dataset (collision-safe naming).
#   4. Save the holdout images for repeated post-fine-tuning evaluation.
#   5. Evaluate a trained model on the holdout and return before/after metrics.
#
# This module is intentionally stateless — functions take paths in, produce
# files out, and return results. All orchestration lives in the CLI.
# =============================================================================

from __future__ import annotations

import json
import logging
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common.io import atomic_write_json
from src.data.image_utils import scan_image_folder, SUPPORTED_IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# Fixed suffix appended to drifted image filenames when they are copied into
# the training dataset. Prevents silent overwrites of original files that
# share the same name.
_DRIFTED_SUFFIX = "_drifted"

# Directory name for the baseline metrics file inside the holdout directory.
_BASELINE_FILENAME = "baseline_metrics.json"


# =============================================================================
# Public API
# =============================================================================

def scan_labeled_drifted_dir(drifted_dir: Path) -> dict[str, list[Path]]:
    """Scan a labeled drifted image directory and return per-class image paths.

    Expects ImageFolder structure — one subdirectory per class, images inside:
        drifted_dir/cats/cat_drifted_000.png
        drifted_dir/dogs/dog_drifted_000.png

    Args:
        drifted_dir: Path to the labeled drifted image directory.

    Returns:
        Dict mapping class name → sorted list of image Paths.

    Raises:
        ValueError: If no class subdirectories with images are found.
    """
    drifted_dir = Path(drifted_dir)
    entries = scan_image_folder(drifted_dir)

    if not entries:
        # Give a helpful error distinguishing flat vs. empty directories.
        flat_images = [
            p for p in drifted_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if flat_images:
            raise ValueError(
                f"Found {len(flat_images)} image(s) directly in '{drifted_dir}' "
                "but no class subdirectories were found. "
                "Organise drifted images into class subdirectories that match "
                "your dataset's class names:\n"
                f"  {drifted_dir}/<class_name>/<image>.png\n"
                "The subdirectory name is treated as the class label."
            )
        raise ValueError(
            f"No labeled images found in '{drifted_dir}'. "
            "Ensure the directory contains class subdirectories with image files."
        )

    class_images: dict[str, list[Path]] = {}
    for path, class_name in entries:
        class_images.setdefault(class_name, []).append(path)

    total = sum(len(v) for v in class_images.values())
    logger.info(
        "Scanned drifted dir '%s': %d images across %d class(es): %s",
        drifted_dir, total, len(class_images),
        {c: len(p) for c, p in class_images.items()},
    )
    return class_images


def split_holdout_stratified(
    class_images: dict[str, list[Path]],
    holdout_ratio: float,
    seed: int,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Stratified split of drifted images into training and held-out sets.

    Preserves class proportions in both splits. Each class is guaranteed to
    have at least one training image (holdout cannot take all images from a
    class). Classes with only one image contribute that image to training only.

    Args:
        class_images:  Dict mapping class name → list of Paths.
        holdout_ratio: Fraction of each class to hold out (0 < ratio < 1).
        seed:          Random seed for reproducibility.

    Returns:
        Tuple (train_images, holdout_images), each a dict of the same form.
    """
    rng = random.Random(seed)
    train: dict[str, list[Path]] = {}
    holdout: dict[str, list[Path]] = {}

    for class_name, paths in class_images.items():
        shuffled = list(paths)
        rng.shuffle(shuffled)

        n_holdout = max(1, round(len(shuffled) * holdout_ratio))
        # Ensure at least one image remains for training.
        n_holdout = min(n_holdout, len(shuffled) - 1)

        if n_holdout == 0:
            logger.warning(
                "Class '%s' has only %d image(s) — all assigned to training; "
                "none held out for evaluation.",
                class_name, len(shuffled),
            )

        holdout[class_name] = shuffled[:n_holdout]
        train[class_name] = shuffled[n_holdout:]

    return train, holdout


def safe_copy_with_suffix(
    src: Path,
    dest_dir: Path,
    suffix: str = _DRIFTED_SUFFIX,
) -> Path:
    """Copy src to dest_dir, appending suffix to the stem to avoid collisions.

    If <stem><suffix><ext> already exists, appends an integer counter:
      cat_000_drifted.png → cat_000_drifted_1.png → cat_000_drifted_2.png ...

    Args:
        src:      Source image path.
        dest_dir: Destination directory (created if missing).
        suffix:   String appended to the filename stem.

    Returns:
        Path of the copied file.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem
    ext = src.suffix
    dest = dest_dir / f"{stem}{suffix}{ext}"
    counter = 0
    while dest.exists():
        counter += 1
        dest = dest_dir / f"{stem}{suffix}_{counter}{ext}"
    shutil.copy2(src, dest)
    return dest


def copy_training_images_to_dataset(
    train_images: dict[str, list[Path]],
    raw_dataset_images_dir: Path,
) -> dict[str, list[Path]]:
    """Copy the training portion of drifted images into the raw dataset.

    Images are placed under raw_dataset_images_dir/<class_name>/ with
    collision-safe naming (_drifted suffix). This makes them visible to the
    pipeline's versioning step, which will detect the new content, generate a
    new version hash, and include the images in the next training run.

    Args:
        train_images:           Dict mapping class name → training image paths.
        raw_dataset_images_dir: Path to data/raw/<dataset>/images/.

    Returns:
        Dict mapping class name → list of destination paths (for logging).
    """
    copied: dict[str, list[Path]] = {}
    for class_name, paths in train_images.items():
        class_dir = raw_dataset_images_dir / class_name
        copied[class_name] = []
        for src in paths:
            dest = safe_copy_with_suffix(src, class_dir)
            copied[class_name].append(dest)
            logger.debug("  Copied to dataset: %s → %s", src.name, dest.name)
    total = sum(len(v) for v in copied.values())
    logger.info(
        "Copied %d training image(s) to dataset directory '%s'",
        total, raw_dataset_images_dir,
    )
    return copied


def save_holdout_images(
    holdout_images: dict[str, list[Path]],
    holdout_dir: Path,
) -> None:
    """Copy holdout images to holdout_dir/<class>/ with collision-safe naming.

    The holdout directory uses the same ImageFolder layout as the raw dataset,
    so evaluate_on_holdout_dir() can scan it with scan_image_folder().

    Args:
        holdout_images: Dict mapping class name → holdout image paths.
        holdout_dir:    Root holdout directory (created if missing).
    """
    for class_name, paths in holdout_images.items():
        class_dir = holdout_dir / class_name
        for src in paths:
            safe_copy_with_suffix(src, class_dir)
    total = sum(len(v) for v in holdout_images.values())
    logger.info("Saved %d holdout image(s) to '%s'", total, holdout_dir)


def evaluate_on_holdout_dir(
    model,
    holdout_dir: Path,
    prep_config,
    norm_mean: list[float] | None,
    norm_std: list[float] | None,
    class_to_index: dict[str, int],
) -> dict:
    """Evaluate a CNN model on the held-out drifted images.

    Applies the same preprocessing as the training pipeline (resize, colour
    mode conversion, z-score normalisation) and then runs model.predict().
    Only handles standard JPG/PNG images (raw DNG pipelines are not supported
    for holdout evaluation in this release).

    Args:
        model:         Trained PyTorch CNN with a .predict(X_nchw) method.
        holdout_dir:   Path to ImageFolder holdout directory.
        prep_config:   PreprocessingConfig with an .image section.
        norm_mean:     Channel-wise mean from training (None = no z-score).
        norm_std:      Channel-wise std from training.
        class_to_index: Class name → integer index mapping from feature_map.json.

    Returns:
        Dict with accuracy, f1_score, precision, recall, n_samples,
        class_distribution.

    Raises:
        ValueError: If no valid images are found after label filtering.
    """
    from PIL import Image
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    img_config = prep_config.image
    if img_config is None:
        raise ValueError(
            "Preprocessing config has no 'image' section — "
            "holdout evaluation requires an image pipeline."
        )

    target_h, target_w = img_config.target_size
    entries = scan_image_folder(holdout_dir)

    if not entries:
        raise ValueError(
            f"No images found in holdout directory '{holdout_dir}'. "
            "Run prepare-drift-training first."
        )

    arrays: list[np.ndarray] = []
    labels: list[int] = []

    for img_path, class_name in entries:
        if class_name not in class_to_index:
            logger.warning(
                "Holdout image '%s' has unknown class '%s' — skipping.",
                img_path.name, class_name,
            )
            continue
        try:
            img = Image.open(img_path)
            if img_config.color_mode == "grayscale":
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            img = img.resize((target_w, target_h))  # PIL uses (W, H)
            arr = np.array(img, dtype=np.float64) / 255.0
            arrays.append(arr)
            labels.append(class_to_index[class_name])
        except Exception as exc:
            logger.warning("Could not load holdout image '%s': %s", img_path.name, exc)

    if not arrays:
        raise ValueError(
            "No valid images loaded from holdout directory. "
            "Check that class subdirectory names match the dataset's class_to_index."
        )

    X = np.stack(arrays)   # (N, H, W, C) or (N, H, W)
    y_true = np.array(labels, dtype=np.int64)

    if img_config.normalize and norm_mean is not None and norm_std is not None:
        mean = np.array(norm_mean, dtype=np.float64)
        std = np.array(norm_std, dtype=np.float64)
        std = np.where(std == 0, 1.0, std)
        X = (X - mean) / std

    # NHWC → NCHW for PyTorch
    X_nchw = X.transpose(0, 3, 1, 2).astype(np.float32) if X.ndim == 4 else X.astype(np.float32)

    y_pred = model.predict(X_nchw)

    # Build reverse index for readable distribution
    index_to_class = {v: k for k, v in class_to_index.items()}

    return {
        "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score":  round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "recall":    round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "n_samples": int(len(y_true)),
        "class_distribution": {
            index_to_class.get(int(idx), str(idx)): int(np.sum(y_true == idx))
            for idx in np.unique(y_true)
        },
    }


def load_class_to_index(
    processed_dir: Path,
    dataset_name: str,
    version_id: str,
) -> dict[str, int]:
    """Load the class-name-to-integer-index mapping from a processed dataset.

    Args:
        processed_dir: Base directory for processed data.
        dataset_name:  Dataset identifier.
        version_id:    Content-addressed version hash.

    Returns:
        Dict mapping class name → integer index.

    Raises:
        FileNotFoundError: If feature_map.json is missing.
        ValueError: If class_to_index is absent or empty.
    """
    fm_path = (
        Path(processed_dir) / dataset_name / version_id / "preprocessed" / "feature_map.json"
    )
    if not fm_path.exists():
        raise FileNotFoundError(
            f"feature_map.json not found at '{fm_path}'. "
            "Run the preprocessing stage first."
        )
    with open(fm_path) as f:
        fm = json.load(f)
    mapping = fm.get("class_to_index", {})
    if not mapping:
        raise ValueError(
            f"class_to_index is empty in '{fm_path}'. "
            "The dataset may not have been fully preprocessed."
        )
    return mapping


def save_baseline_metrics(metrics: dict, holdout_dir: Path) -> Path:
    """Persist baseline (pre-fine-tuning) evaluation metrics to the holdout dir.

    Args:
        metrics:     Output of evaluate_on_holdout_dir().
        holdout_dir: Root holdout directory.

    Returns:
        Path to the written baseline_metrics.json.
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Baseline evaluation of the Production model on the held-out "
            "drifted images, measured before fine-tuning."
        ),
        "metrics": metrics,
    }
    path = Path(holdout_dir) / _BASELINE_FILENAME
    atomic_write_json(path, payload)
    logger.info("Baseline metrics saved to '%s'", path)
    return path


def save_normalization_stats(
    norm_mean: list[float] | None,
    norm_std: list[float] | None,
    holdout_dir: Path,
) -> Path:
    """Save the training normalization statistics alongside the holdout.

    These stats are used to preprocess holdout images consistently for both
    the baseline evaluation (before fine-tuning) and the post-fine-tuning
    evaluation. Saving them ensures both evaluations use identical scaling
    regardless of whether the dataset version changes between the two runs.

    Args:
        norm_mean:   Channel-wise mean from the training feature_map.json.
        norm_std:    Channel-wise std from the training feature_map.json.
        holdout_dir: Root holdout directory.

    Returns:
        Path to the written normalization_stats.json.
    """
    payload = {"mean": norm_mean, "std": norm_std}
    path = Path(holdout_dir) / "normalization_stats.json"
    atomic_write_json(path, payload)
    return path


def load_normalization_stats(
    holdout_dir: Path,
) -> tuple[list[float] | None, list[float] | None]:
    """Load previously saved normalization statistics from the holdout directory.

    Args:
        holdout_dir: Root holdout directory.

    Returns:
        (mean, std) as loaded, or (None, None) if the file does not exist.
    """
    path = Path(holdout_dir) / "normalization_stats.json"
    if not path.exists():
        return None, None
    with open(path) as f:
        stats = json.load(f)
    return stats.get("mean"), stats.get("std")


def load_baseline_metrics(holdout_dir: Path) -> dict | None:
    """Load previously saved baseline metrics, or None if not found.

    Args:
        holdout_dir: Root holdout directory.

    Returns:
        Baseline payload dict (with "metrics" key), or None.
    """
    path = Path(holdout_dir) / _BASELINE_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
