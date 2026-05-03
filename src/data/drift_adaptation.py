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

# Subdirectory under data/raw/<dataset>/ where per-attempt manifests live.
# Each manifest records exactly which files were copied so a rejected
# fine-tuning experiment can be rolled back cleanly.
_DRIFT_ATTEMPTS_SUBDIR = ".drift_attempts"


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


# =============================================================================
# Drift-attempt manifests
# =============================================================================
# Each prepare-drift-training invocation writes a manifest under
# data/raw/<dataset>/.drift_attempts/<timestamp>.json listing every file it
# copied into the raw dataset, the source drifted_dir, the holdout dir, and
# the baseline metrics path. A rejected or aborted fine-tune can then be
# rolled back cleanly with rollback-drift-training, which deletes exactly
# those files (and optionally the holdout) without touching unrelated data.


def _attempts_dir(raw_dataset_dir: Path) -> Path:
    """Resolve data/raw/<dataset>/.drift_attempts/."""
    return Path(raw_dataset_dir) / _DRIFT_ATTEMPTS_SUBDIR


def write_drift_attempt_manifest(
    raw_dataset_dir: Path,
    drifted_dir: Path,
    raw_images_dir: Path,
    holdout_dir: Path,
    copied: dict[str, list[Path]],
    baseline_metrics: dict | None,
    holdout_ratio: float,
    random_seed: int,
) -> Path:
    """Persist a per-attempt manifest of drifted files added to the dataset.

    The manifest file is named with a UTC timestamp (``<YYYYMMDD_HHMMSS>.json``)
    and includes:
      - source drifted_dir, raw_images_dir, holdout_dir
      - the holdout_ratio and random_seed used for the split
      - the baseline metrics summary (acc / f1 / precision / recall) at write time
      - per-class destination file lists (relative to raw_images_dir)

    Args:
        raw_dataset_dir:  ``data/raw/<dataset>/`` (NOT the images/ subdir).
        drifted_dir:      Source directory passed to prepare-drift-training.
        raw_images_dir:   ``data/raw/<dataset>/images/`` — destination root.
        holdout_dir:      Holdout directory under data/evaluation/.
        copied:           Output of ``copy_training_images_to_dataset`` (dict
                          mapping class name -> absolute destination Paths).
        baseline_metrics: Output of ``evaluate_on_holdout_dir``, or None if
                          no holdout was carved out.
        holdout_ratio:    Holdout fraction used.
        random_seed:      Seed used for the stratified split.

    Returns:
        Path to the written manifest file.
    """
    attempts_dir = _attempts_dir(raw_dataset_dir)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    manifest_path = attempts_dir / f"{timestamp}.json"

    raw_images_dir = Path(raw_images_dir).resolve()
    files_by_class: dict[str, list[str]] = {}
    total = 0
    for class_name, paths in copied.items():
        rel_paths: list[str] = []
        for p in paths:
            try:
                rel = Path(p).resolve().relative_to(raw_images_dir).as_posix()
            except ValueError:
                # Fall back to the bare filename if the path is somehow not
                # under raw_images_dir (shouldn't happen in normal flow).
                rel = Path(p).name
            rel_paths.append(rel)
            total += 1
        files_by_class[class_name] = rel_paths

    payload = {
        "attempt_id": timestamp,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "drifted_dir": str(Path(drifted_dir)),
        "raw_images_dir": str(raw_images_dir),
        "holdout_dir": str(Path(holdout_dir)),
        "holdout_ratio": holdout_ratio,
        "random_seed": random_seed,
        "n_files_added": total,
        "files_by_class": files_by_class,
        "baseline_metrics": baseline_metrics,
    }
    atomic_write_json(manifest_path, payload)
    logger.info(
        "Drift attempt manifest written to '%s' (tracked %d file(s) for rollback).",
        manifest_path, total,
    )
    return manifest_path


def list_drift_attempts(raw_dataset_dir: Path) -> list[dict]:
    """Return all drift-attempt manifests under raw_dataset_dir, oldest first.

    Each entry carries the manifest payload with an extra ``manifest_path``
    field so callers can pass it straight to ``rollback_drift_attempt``.
    Returns an empty list if no attempts have been recorded.
    """
    attempts_dir = _attempts_dir(raw_dataset_dir)
    if not attempts_dir.exists():
        return []
    out: list[dict] = []
    for manifest_path in sorted(attempts_dir.glob("*.json")):
        try:
            with open(manifest_path) as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning("Skipping unreadable manifest '%s': %s", manifest_path, exc)
            continue
        payload["manifest_path"] = str(manifest_path)
        out.append(payload)
    return out


def latest_drift_attempt(raw_dataset_dir: Path) -> dict | None:
    """Return the most recent drift-attempt manifest, or None if there are none."""
    attempts = list_drift_attempts(raw_dataset_dir)
    return attempts[-1] if attempts else None


def rollback_drift_attempt(
    manifest_path: Path,
    remove_holdout: bool = False,
) -> dict:
    """Delete every file recorded in a drift-attempt manifest.

    Walks ``files_by_class`` and unlinks each entry under ``raw_images_dir``,
    then removes the manifest itself. Files that have already been deleted
    (e.g. by a prior cleanup) are counted as "missing" and do not raise.
    Empty per-class directories left behind are removed; the parent
    ``images/`` directory is preserved.

    Args:
        manifest_path:  Path to the manifest JSON written by
                        ``write_drift_attempt_manifest``.
        remove_holdout: If True, also recursively remove the holdout directory
                        recorded in the manifest. Defaults to False so callers
                        can keep historical baseline measurements.

    Returns:
        Dict with ``removed``, ``missing``, ``manifest_path``, ``holdout_removed``.

    Raises:
        FileNotFoundError: If the manifest does not exist.
        ValueError:        If the manifest is malformed.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: '{manifest_path}'")
    with open(manifest_path) as f:
        payload = json.load(f)

    raw_images_dir = payload.get("raw_images_dir")
    files_by_class = payload.get("files_by_class")
    if not raw_images_dir or not isinstance(files_by_class, dict):
        raise ValueError(
            f"Manifest '{manifest_path}' is missing required fields "
            "(raw_images_dir, files_by_class)."
        )
    raw_images_dir = Path(raw_images_dir)

    removed = 0
    missing = 0
    touched_class_dirs: set[Path] = set()
    for class_name, rel_paths in files_by_class.items():
        class_dir = raw_images_dir / class_name
        touched_class_dirs.add(class_dir)
        for rel in rel_paths:
            target = (raw_images_dir / rel).resolve()
            try:
                target.relative_to(raw_images_dir.resolve())
            except ValueError:
                logger.warning(
                    "Refusing to delete '%s' — outside raw_images_dir '%s'.",
                    target, raw_images_dir,
                )
                continue
            if target.exists():
                try:
                    target.unlink()
                    removed += 1
                except OSError as exc:
                    logger.warning("Could not remove '%s': %s", target, exc)
            else:
                missing += 1

    # Tidy up empty per-class directories created solely by drifted images.
    for class_dir in touched_class_dirs:
        try:
            if class_dir.exists() and not any(class_dir.iterdir()):
                class_dir.rmdir()
        except OSError:
            pass

    holdout_removed = False
    if remove_holdout:
        holdout_dir = payload.get("holdout_dir")
        if holdout_dir:
            holdout_path = Path(holdout_dir)
            if holdout_path.exists():
                shutil.rmtree(holdout_path, ignore_errors=True)
                holdout_removed = True

    try:
        manifest_path.unlink()
    except OSError as exc:
        logger.warning("Could not remove manifest '%s': %s", manifest_path, exc)

    logger.info(
        "Rolled back drift attempt '%s': removed=%d, missing=%d, holdout_removed=%s.",
        manifest_path.name, removed, missing, holdout_removed,
    )
    return {
        "manifest_path": str(manifest_path),
        "removed": removed,
        "missing": missing,
        "holdout_removed": holdout_removed,
    }


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

    Applies the same preprocessing as the training pipeline (ISP or PIL resize,
    colour mode conversion, z-score normalisation) and then runs model.predict().
    Supports both standard JPG/PNG pipelines and raw DNG/ISP pipelines:

    - Standard (raw_input=False): PIL → resize → [0,1] → z-score
    - RAW DNG  (raw_input=True):  rawpy → ISP → skimage resize → [0,1] → z-score

    The ISP config used for RAW images is the *baseline* config from the
    preprocessing config (img_config.isp), matching the config used during
    normal training preprocessing.

    Args:
        model:          Trained PyTorch CNN with a .predict(X_nchw) method.
        holdout_dir:    Path to ImageFolder holdout directory.
        prep_config:    PreprocessingConfig with an .image section.
        norm_mean:      Channel-wise mean from training (None = no z-score).
        norm_std:       Channel-wise std from training.
        class_to_index: Class name → integer index mapping from feature_map.json.

    Returns:
        Dict with accuracy, f1_score, precision, recall, n_samples,
        class_distribution.

    Raises:
        ValueError: If no valid images are found after label filtering.
        ImportError: If rawpy/scikit-image are missing for a RAW pipeline.
    """
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    img_config = prep_config.image
    if img_config is None:
        raise ValueError(
            "Preprocessing config has no 'image' section — "
            "holdout evaluation requires an image pipeline."
        )

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

        if img_config.raw_input:
            arr = _load_holdout_image_raw(img_path, img_config)
        else:
            arr = _load_holdout_image_standard(img_path, img_config)

        if arr is None:
            continue
        arrays.append(arr)
        labels.append(class_to_index[class_name])

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


def _load_holdout_image_standard(
    img_path: Path,
    img_config,
) -> np.ndarray | None:
    """Load one standard JPG/PNG holdout image using PIL.

    Mirrors prepare_batch._load_standard_images for a single image.
    Returns a float64 array scaled to [0, 1] when normalize=True,
    or [0, 255] when normalize=False.  Returns None on load failure.
    """
    try:
        from PIL import Image
        target_h, target_w = img_config.target_size
        img = Image.open(img_path)
        img = img.convert("L" if img_config.color_mode == "grayscale" else "RGB")
        img = img.resize((target_w, target_h))  # PIL uses (W, H)
        arr = np.array(img, dtype=np.float64)
        if img_config.normalize:
            arr = arr / 255.0
        return arr
    except Exception as exc:
        logger.warning("Could not load holdout image '%s': %s", img_path.name, exc)
        return None


def _load_holdout_image_raw(
    img_path: Path,
    img_config,
) -> np.ndarray | None:
    """Load one RAW DNG holdout image through the ISP pipeline.

    Mirrors prepare_batch._load_raw_images for a single image.
    Uses the baseline ISP config (img_config.isp) — the same ISP parameters
    that were used during normal training preprocessing, not any drift scenario
    variant.  Returns float64 in [0, 1] (skimage_resize preserves this range).
    Returns None on load failure.

    Raises:
        ImportError: If rawpy or scikit-image are not installed.
    """
    try:
        import rawpy
        from skimage.transform import resize as skimage_resize
    except ImportError:
        raise ImportError(
            "rawpy and scikit-image are required for RAW DNG holdout evaluation. "
            "Install with: pip install rawpy scikit-image"
        )

    from src.data.isp_pipeline import run_isp, read_camera_params

    try:
        target_h, target_w = img_config.target_size
        with rawpy.imread(str(img_path)) as raw:
            raw_array = raw.raw_image_visible.copy().astype(np.float32)
        camera_params = read_camera_params(img_path)
        rgb = run_isp(raw_array, img_config.isp, camera_params)
        if img_config.color_mode == "grayscale":
            luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
            return skimage_resize(luma, (target_h, target_w), anti_aliasing=True)
        return skimage_resize(rgb, (target_h, target_w, 3), anti_aliasing=True)
    except Exception as exc:
        logger.warning("Could not load RAW holdout image '%s': %s", img_path.name, exc)
        return None


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
