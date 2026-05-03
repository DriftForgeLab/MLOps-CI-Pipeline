# =============================================================================
# src/data/prepare_batch.py — Batch preprocessing for drift monitoring
# =============================================================================
# Prepares a batch of new production images into an NPZ file that can be
# compared against the training reference by monitor-drift-image.
#
# Applies exactly the same preprocessing as the original training pipeline:
#   - Raw DNG: ISP pipeline → resize → z-score normalization
#   - Standard JPG/PNG: resize → /255 → z-score normalization
#
# Normalization statistics (mean, std) are loaded from the training run's
# feature_map.json so the batch is on the same scale as the reference.
#
# Input directory structure — either flat or ImageFolder:
#   Flat:        data/incoming/batch/frame_001.dng
#   ImageFolder: data/incoming/batch/class_a/frame_001.dng
# Class labels are not required — only pixel data is used for drift analysis.
# =============================================================================

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)

# Image file extensions recognized by the batch scanner
_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".jpg", ".jpeg", ".png", ".dng", ".tiff", ".tif", ".bmp",
})


# =============================================================================
# Public API
# =============================================================================

def resolve_latest_version(processed_dir: Path, dataset_name: str) -> str:
    """Find the most recently preprocessed version for a dataset.

    Scans data/processed/<dataset_name>/ for version directories that contain
    a preprocessed/feature_map.json, and returns the name of the newest one
    (by filesystem modification time).

    Args:
        processed_dir: Base directory for processed data (e.g. data/processed/).
        dataset_name:  Dataset identifier.

    Returns:
        Version ID string (content-addressed hash).

    Raises:
        FileNotFoundError: If no preprocessed versions are found.
    """
    dataset_dir = processed_dir / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"No processed data found for dataset '{dataset_name}' at '{dataset_dir}'. "
            "Run the preprocessing stage first."
        )

    candidates = [
        d for d in dataset_dir.iterdir()
        if d.is_dir() and (d / "preprocessed" / "feature_map.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No preprocessed versions found for dataset '{dataset_name}' in '{dataset_dir}'. "
            "Run the preprocessing stage first."
        )

    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    version_id = candidates[0].name
    logger.info("Auto-resolved version: %s (newest of %d)", version_id, len(candidates))
    return version_id


def load_training_stats(
    processed_dir: Path,
    dataset_name: str,
    version_id: str,
) -> tuple[list[float] | None, list[float] | None, list[str] | None]:
    """Load normalization stats and expected formats from a training run.

    Args:
        processed_dir: Base directory for processed data.
        dataset_name:  Dataset identifier.
        version_id:    Version hash from the training run.

    Returns:
        (mean, std, expected_formats) where mean and std are channel-wise
        lists (or None if normalize=False was used), and expected_formats is
        a list of file extensions (e.g. [".dng"]) from dataset.yaml, or None.

    Raises:
        FileNotFoundError: If feature_map.json is not found.
    """
    feature_map_path = (
        processed_dir / dataset_name / version_id / "preprocessed" / "feature_map.json"
    )
    if not feature_map_path.exists():
        raise FileNotFoundError(
            f"feature_map.json not found at '{feature_map_path}'. "
            "Ensure the dataset has been preprocessed."
        )

    with open(feature_map_path) as f:
        fm = json.load(f)

    stats = fm.get("normalization_stats") or {}
    mean = stats.get("mean")
    std = stats.get("std")

    # Load expected image formats from dataset.yaml
    dataset_yaml = processed_dir / dataset_name / version_id / "dataset.yaml"
    expected_formats: list[str] | None = None
    if dataset_yaml.exists():
        with open(dataset_yaml) as f:
            meta = yaml.safe_load(f)
        image_props = (meta or {}).get("image_properties") or {}
        expected_formats = image_props.get("expected_formats")

    return mean, std, expected_formats


def prepare_image_batch(
    input_dir: Path,
    prep_config,
    norm_mean: list[float] | None,
    norm_std: list[float] | None,
    expected_formats: list[str] | None = None,
) -> np.ndarray:
    """Preprocess new images into a batch array using the same steps as training.

    Scans input_dir recursively for image files, applies the same resize,
    color mode conversion, and normalization as the original training pipeline,
    and returns a stacked array ready for drift analysis.

    Args:
        input_dir:        Directory with new production images. Flat or
                          ImageFolder structure — class labels are ignored.
        prep_config:      Preprocessing config (ImagePreprocessingConfig).
        norm_mean:        Channel-wise mean from training feature_map.json.
                          None = no z-score (normalize=False during training).
        norm_std:         Channel-wise std from training feature_map.json.
        expected_formats: Allowed file extensions (e.g. [".dng"]). None = all.

    Returns:
        float64 array of shape (N, H, W, C) or (N, H, W), normalized the same
        way as the training data.

    Raises:
        ValueError: If no readable images are found in input_dir.
    """
    img_config = prep_config.image
    if img_config is None:
        raise ValueError(
            "Preprocessing config has no 'image' section. "
            "Use an image pipeline config (pipeline_image_cnn.yaml or pipeline_image_raw.yaml)."
        )

    image_paths = _scan_images(input_dir, expected_formats)
    if not image_paths:
        raise ValueError(
            f"No images found in '{input_dir}'. "
            f"Expected formats: {expected_formats or list(_IMAGE_EXTENSIONS)}"
        )

    logger.info("Found %d image(s) in '%s'", len(image_paths), input_dir)

    if img_config.raw_input:
        arrays = _load_raw_images(image_paths, img_config)
    else:
        arrays = _load_standard_images(image_paths, img_config)

    if not arrays:
        raise ValueError(f"No readable images could be processed from '{input_dir}'.")

    X = np.stack(arrays)  # (N, H, W, C) or (N, H, W), float64, [0, 1]

    if img_config.normalize and norm_mean is not None and norm_std is not None:
        mean = np.array(norm_mean, dtype=np.float64)
        std = np.array(norm_std, dtype=np.float64)
        std = np.where(std == 0, 1.0, std)
        X = (X - mean) / std
        logger.info("Applied z-score normalization (training stats).")
    elif img_config.normalize:
        logger.warning(
            "normalize=True in config but no training stats found — "
            "batch will not be z-score normalized. Drift scores may be off."
        )

    return X


# =============================================================================
# Private helpers
# =============================================================================

def _scan_images(
    input_dir: Path,
    expected_formats: list[str] | None,
) -> list[Path]:
    """Scan input_dir recursively for image files."""
    if expected_formats:
        extensions = frozenset(
            fmt.lower() if fmt.startswith(".") else f".{fmt.lower()}"
            for fmt in expected_formats
        )
    else:
        extensions = _IMAGE_EXTENSIONS

    paths = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )
    return paths


def _load_raw_images(image_paths: list[Path], img_config) -> list[np.ndarray]:
    """Load DNG files through the ISP pipeline and resize."""
    try:
        import rawpy
        from skimage.transform import resize as skimage_resize
    except ImportError as exc:
        raise ImportError(
            "rawpy and scikit-image are required for raw DNG images. "
            "Install with: pip install rawpy scikit-image"
        ) from exc

    from src.data.isp_pipeline import run_isp, read_camera_params

    target_h, target_w = img_config.target_size
    arrays: list[np.ndarray] = []

    for img_path in image_paths:
        try:
            with rawpy.imread(str(img_path)) as raw:
                raw_array = raw.raw_image_visible.copy().astype(np.float32)
            camera_params = read_camera_params(img_path)
            rgb = run_isp(raw_array, img_config.isp, camera_params)

            if img_config.color_mode == "grayscale":
                luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                arr = skimage_resize(luma, (target_h, target_w), anti_aliasing=True)
            else:
                arr = skimage_resize(rgb, (target_h, target_w, 3), anti_aliasing=True)
            # arr is float64 in [0, 1]
            arrays.append(arr)
            logger.debug("  Processed: %s", img_path.name)
        except Exception as e:
            logger.warning("  Skipping '%s': %s", img_path.name, e)

    return arrays


def _load_standard_images(image_paths: list[Path], img_config) -> list[np.ndarray]:
    """Load standard JPG/PNG images, resize, and scale to [0, 1]."""
    from PIL import Image

    target_h, target_w = img_config.target_size
    arrays: list[np.ndarray] = []

    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            if img_config.color_mode == "grayscale":
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            img = img.resize((target_w, target_h))  # PIL uses (width, height)
            arr = np.array(img, dtype=np.float64)
            if img_config.normalize:
                arr = arr / 255.0  # scale to [0, 1] before z-score
            arrays.append(arr)
            logger.debug("  Processed: %s", img_path.name)
        except Exception as e:
            logger.warning("  Skipping '%s': %s", img_path.name, e)

    return arrays
