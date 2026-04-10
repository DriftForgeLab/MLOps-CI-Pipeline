# =============================================================================
# src/data/image_preprocess.py — Image preprocessing pipeline
# =============================================================================
# Resizes, normalizes, and optionally augments images from split directories
# into NPZ archives (train.npz, val.npz, test.npz) for CNN training.
#
# Leak-proof: normalization statistics (mean, std) are computed ONLY from
# training images and applied to all splits.
#
# Determinism: same config + seed = identical output.
# Idempotency: SHA-256 hash of (image config + manifest + PIPELINE_VERSION).
# =============================================================================

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.common.io import atomic_write_json, atomic_write_npz, atomic_write_pickle
from src.config.loader import PreprocessingConfig, load_preprocessing_config
from src.data.image_utils import compute_folder_hash, scan_image_folder
from src.config.schema import ISPConfig

logger = logging.getLogger(__name__)

# Bump this whenever the ISP pipeline logic changes (not just config) to
# invalidate cached preprocessed NPZ files and force re-preprocessing.
PIPELINE_VERSION = "1.0.0"
PREPROCESSED_SUBDIR = "preprocessed"


def _load_and_transform_images(
    images_dir: Path,
    target_size: tuple[int, int],
    color_mode: str,
    expected_formats: list[str] | None,
) -> tuple[np.ndarray, list[str], list[Path]]:
    """Load images from an ImageFolder split, resize, and convert to numpy arrays.

    Returns:
        (images_array, labels, paths) where images_array has shape (N, H, W, C) or (N, H, W).
    """
    from PIL import Image

    entries = scan_image_folder(images_dir, expected_formats)
    if not entries:
        raise ValueError(f"No images found in {images_dir}")

    arrays = []
    labels = []
    paths = []

    for img_path, class_name in entries:
        try:
            img = Image.open(img_path)
            if color_mode == "grayscale":
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            img = img.resize((target_size[1], target_size[0]))  # PIL uses (width, height)
            arr = np.array(img, dtype=np.float64)
            arrays.append(arr)
            labels.append(class_name)
            paths.append(img_path)
        except Exception as e:
            logger.warning("Skipping unreadable image '%s': %s", img_path, e)

    if not arrays:
        raise ValueError(f"No readable images found in {images_dir}")

    return np.stack(arrays), labels, paths


def _load_and_transform_raw_images(
    images_dir: Path,
    img_config,
    expected_formats: list[str] | None,
) -> tuple[np.ndarray, list[str], list[Path]]:
    """Load DNG raw images, run ISP pipeline, resize, return numpy arrays.

    This is the raw-input counterpart of _load_and_transform_images.
    Resize is performed on the float array (no intermediate uint8 conversion)
    to avoid unnecessary precision loss.

    Unlike _load_and_transform_images (which returns [0, 255] from PIL), this
    function returns values in [0, 1] float64 — the natural output range of the
    ISP pipeline. The normalization step in the caller handles the two paths
    differently: JPG/PNG divides by 255 first; raw images skip that step since
    they are already in [0, 1].

    Args:
        images_dir:       Path to an ImageFolder split's images/ directory.
        img_config:       Validated ImagePreprocessingConfig (raw_input=True).
        expected_formats: Allowed file extensions from dataset.yaml.

    Returns:
        (images_array, labels, paths) where images_array has shape (N, H, W, C)
        with values in [0, 1] float64.
    """
    try:
        import rawpy
    except ImportError:
        raise ImportError(
            "rawpy is required for DNG image loading. Install with: pip install rawpy"
        )

    try:
        from skimage.transform import resize as skimage_resize
    except ImportError:
        raise ImportError(
            "scikit-image is required for float-preserving resize. "
            "Install with: pip install scikit-image"
        )

    from src.data.isp_pipeline import run_isp, read_camera_params

    entries = scan_image_folder(images_dir, expected_formats)
    if not entries:
        raise ValueError(f"No images found in {images_dir}")

    target_h, target_w = img_config.target_size
    arrays = []
    labels = []
    paths = []

    for img_path, class_name in entries:
        try:
            # Read raw Bayer data — copy immediately to free rawpy object
            with rawpy.imread(str(img_path)) as raw:
                raw_array = raw.raw_image_visible.copy().astype(np.float32)

            # Read camera parameters from DNG metadata
            camera_params = read_camera_params(img_path)

            # Run ISP pipeline → float64 RGB in [0, 1]
            rgb = run_isp(raw_array, img_config.isp, camera_params)

            # Resize on float array (no precision loss from uint8 conversion)
            if img_config.color_mode == "grayscale":
                # Convert to luminance using ITU-R BT.601 coefficients
                luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                arr = skimage_resize(luma, (target_h, target_w), anti_aliasing=True)
            else:
                arr = skimage_resize(
                    rgb, (target_h, target_w, 3), anti_aliasing=True
                )

            # arr is float64 in [0, 1] (ISP + skimage_resize preserve this range)
            arrays.append(arr)
            labels.append(class_name)
            paths.append(img_path)
        except Exception as e:
            logger.warning("Skipping unreadable raw image '%s': %s", img_path, e)

    if not arrays:
        raise ValueError(f"No readable raw images found in {images_dir}")

    return np.stack(arrays), labels, paths


def _apply_augmentation(
    X: np.ndarray,
    y: np.ndarray,
    aug_config,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply offline augmentation to training images.

    Returns augmented copies appended to the original arrays.
    """
    if not aug_config.enabled or aug_config.augmentation_factor <= 1:
        return X, y

    rng = np.random.RandomState(random_seed)
    augmented_X = [X]
    augmented_y = [y]

    for _ in range(aug_config.augmentation_factor - 1):
        batch = X.copy()

        if aug_config.horizontal_flip:
            # Flip along width axis
            flip_mask = rng.rand(len(batch)) > 0.5
            if batch.ndim == 4:  # (N, H, W, C)
                batch[flip_mask] = batch[flip_mask, :, ::-1, :]
            else:  # (N, H, W) grayscale
                batch[flip_mask] = batch[flip_mask, :, ::-1]

        if aug_config.rotation_degrees > 0:
            # Simple 90-degree rotation approximation
            for i in range(len(batch)):
                k = rng.randint(0, 4)  # 0, 90, 180, 270 degrees
                if k > 0:
                    batch[i] = np.rot90(batch[i], k=k, axes=(0, 1))

        augmented_X.append(batch)
        augmented_y.append(y.copy())

    return np.concatenate(augmented_X), np.concatenate(augmented_y)


def run_image_preprocessing(
    dataset_name: str,
    version_id: str,
    prep_config_path: Path,
    processed_dir: Path = Path("data/processed"),
    random_seed: int | None = None,
) -> None:
    """Run image preprocessing: resize, normalize, augment, save as NPZ."""
    prep_config = load_preprocessing_config(prep_config_path)
    img_config = prep_config.image

    if img_config is None:
        raise ValueError(
            "Image preprocessing config ('image' section) is missing from "
            f"'{prep_config_path}'. Required for image_classification datasets."
        )

    version_dir = processed_dir / dataset_name / version_id
    if not version_dir.exists():
        raise FileNotFoundError(
            f"No versioned dataset found at '{version_dir}'. "
            "Ensure the dataset has been ingested before running preprocessing."
        )

    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR

    # Load dataset metadata
    yaml_path = version_dir / "dataset.yaml"
    with open(yaml_path) as f:
        metadata = yaml.safe_load(f)

    image_props = metadata.get("image_properties", {}) or {}
    expected_formats = image_props.get("expected_formats")
    target_name = metadata.get("target", "label")

    # Idempotency check (includes training manifest hash for content sensitivity)
    train_images_dir = version_dir / "train" / "images"
    train_manifest_hash = (
        compute_folder_hash(train_images_dir, expected_formats)
        if train_images_dir.exists()
        else ""
    )
    preprocess_hash = _compute_image_preprocess_hash(img_config, metadata, train_manifest_hash)
    existing_meta = _load_existing_metadata(preprocessed_dir)
    if existing_meta and existing_meta.get("preprocess_hash") == preprocess_hash and _outputs_exist(preprocessed_dir):
        logger.info("  Image preprocessing up-to-date for version '%s' — skipping.", version_id)
        return

    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Validate raw_input requires isp config
    if img_config.raw_input and img_config.isp is None:
        raise ValueError(
            "image.raw_input is True but no 'image.isp' block is configured. "
            "Add an 'isp:' section to your preprocessing config, or use "
            "preprocessing_raw_image.yaml as a starting point."
        )

    # Build class mapping from training split (cached for reuse in the loop)
    if img_config.raw_input:
        X_train_raw, train_labels, _ = _load_and_transform_raw_images(
            train_images_dir, img_config, expected_formats
        )
    else:
        X_train_raw, train_labels, _ = _load_and_transform_images(
            train_images_dir, img_config.target_size, img_config.color_mode, expected_formats
        )

    class_names = sorted(set(train_labels))
    class_to_index = {name: idx for idx, name in enumerate(class_names)}
    index_to_class = {str(idx): name for idx, name in enumerate(class_names)}

    # Scale to [0, 1] before computing stats.
    # JPG/PNG images come from PIL as [0, 255] and need dividing.
    # Raw ISP images are already [0, 1] — skip the division.
    if img_config.normalize and not img_config.raw_input:
        X_train_raw = X_train_raw / 255.0

    # Compute normalization stats from training set ONLY (leak-proof)
    if img_config.normalize:
        if X_train_raw.ndim == 4:  # (N, H, W, C)
            mean = X_train_raw.mean(axis=(0, 1, 2))
            std = X_train_raw.std(axis=(0, 1, 2))
        else:  # (N, H, W) grayscale
            mean = np.array([X_train_raw.mean()])
            std = np.array([X_train_raw.std()])
        std = np.where(std == 0, 1.0, std)  # avoid division by zero
    else:
        mean = None
        std = None

    # Resolve random seed: prefer explicit parameter, fall back to split metadata
    if random_seed is None:
        train_meta_path = version_dir / "train" / "metadata.json"
        if train_meta_path.exists():
            with open(train_meta_path) as f:
                train_meta = json.load(f)
            random_seed = train_meta.get("random_seed", 42)
        else:
            logger.warning("No random_seed provided and no split metadata found — defaulting to 42")
            random_seed = 42

    # Process all splits
    norm_stats = {"mean": mean.tolist() if mean is not None else None,
                  "std": std.tolist() if std is not None else None}

    for split_name in ("train", "val", "test"):
        if split_name == "train":
            # Reuse cached training data (already loaded above)
            X_raw, labels = X_train_raw.copy(), list(train_labels)
        else:
            split_images_dir = version_dir / split_name / "images"
            if img_config.raw_input:
                X_raw, labels, _ = _load_and_transform_raw_images(
                    split_images_dir, img_config, expected_formats
                )
            else:
                X_raw, labels, _ = _load_and_transform_images(
                    split_images_dir, img_config.target_size, img_config.color_mode, expected_formats
                )

        # Normalize using training statistics.
        # For JPG/PNG: divide by 255 first (PIL returns [0, 255]).
        # For raw ISP: already [0, 1] — skip the division.
        if img_config.normalize:
            if split_name != "train" and not img_config.raw_input:
                X_raw = X_raw / 255.0
            X_raw = (X_raw - mean) / std

        # Encode labels (with descriptive error for unseen classes)
        unknown = set(labels) - set(class_to_index)
        if unknown:
            raise ValueError(
                f"[{split_name}] contains classes not in training: {sorted(unknown)}"
            )
        y = np.array([class_to_index[lbl] for lbl in labels])

        # Augment training set only
        if split_name == "train":
            X_raw, y = _apply_augmentation(X_raw, y, img_config.augmentation, random_seed)

        # Flatten (legacy option — ignored for CNN pipelines; flatten defaults to False)
        if img_config.flatten:
            X_raw = X_raw.reshape(X_raw.shape[0], -1)

        atomic_write_npz(preprocessed_dir / f"{split_name}.npz", X=X_raw, y=y)

    # Determine image shape and feature count
    if img_config.color_mode == "grayscale":
        image_shape = list(img_config.target_size)
        num_features = img_config.target_size[0] * img_config.target_size[1]
    else:
        image_shape = list(img_config.target_size) + [3]
        num_features = img_config.target_size[0] * img_config.target_size[1] * 3

    # Write feature_map.json
    feature_map = {
        "output_features": [f"pixel_{i}" for i in range(num_features)] if img_config.flatten else [],
        "target": target_name,
        "image_shape": image_shape,
        "num_features": num_features,
        "class_names": class_names,
        "class_to_index": class_to_index,
        "index_to_class": index_to_class,
        "normalization_stats": norm_stats,
    }
    atomic_write_json(preprocessed_dir / "feature_map.json", feature_map)

    # Write normalization pipeline artifact
    pipeline_artifact = {
        "normalization_stats": norm_stats,
        "target_size": list(img_config.target_size),
        "color_mode": img_config.color_mode,
        "normalize": img_config.normalize,
        "flatten": img_config.flatten,
        "class_to_index": class_to_index,
    }
    atomic_write_pickle(pipeline_artifact, preprocessed_dir / "pipeline.pkl")

    # Write metadata
    meta_payload = {
        "dataset_version_id": version_id,
        "preprocess_hash": preprocess_hash,
        "pipeline_version": PIPELINE_VERSION,
        "preprocessed_at": datetime.now(timezone.utc).isoformat(),
        "transform_description": "image_preprocessing",
        "output_features": feature_map["output_features"],
        "target": target_name,
        "num_classes": len(class_names),
        "class_names": class_names,
        "image_shape": image_shape,
    }
    atomic_write_json(preprocessed_dir / "metadata.json", meta_payload)

    # For raw-image runs: save the ISP config that was used, so the exact
    # processing chain is auditable alongside the preprocessed outputs.
    # None values mean "read from DNG metadata at runtime" — preserved as-is.
    if img_config.raw_input and img_config.isp is not None:
        import dataclasses
        atomic_write_json(
            preprocessed_dir / "isp_config.json",
            dataclasses.asdict(img_config.isp),
        )

    logger.info(
        "  Image preprocessing complete: %d classes, %s shape → %s/{train,val,test}.npz",
        len(class_names),
        image_shape,
        PREPROCESSED_SUBDIR,
    )


def _compute_image_preprocess_hash(img_config, metadata: dict, train_manifest_hash: str = "") -> str:
    import dataclasses
    canonical_data: dict = {
        "pipeline_version": PIPELINE_VERSION,
        "target_size": list(img_config.target_size),
        "color_mode": img_config.color_mode,
        "normalize": img_config.normalize,
        "flatten": img_config.flatten,
        "augmentation_enabled": img_config.augmentation.enabled,
        "augmentation_factor": img_config.augmentation.augmentation_factor,
        "horizontal_flip": img_config.augmentation.horizontal_flip,
        "rotation_degrees": img_config.augmentation.rotation_degrees,
        "raw_input": img_config.raw_input,
        "task_type": metadata.get("task_type"),
        "target": metadata.get("target"),
        "train_manifest_hash": train_manifest_hash,
    }
    if img_config.isp is not None:
        # dataclasses.asdict recursively converts nested dataclasses to dicts
        canonical_data["isp"] = dataclasses.asdict(img_config.isp)
    canonical = json.dumps(canonical_data, sort_keys=True)
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _load_existing_metadata(preprocessed_dir: Path) -> dict | None:
    meta_path = preprocessed_dir / "metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        return meta if isinstance(meta, dict) else None
    except Exception:
        return None


def _outputs_exist(preprocessed_dir: Path) -> bool:
    return (
        all((preprocessed_dir / f"{s}.npz").exists() for s in ("train", "val", "test"))
        and (preprocessed_dir / "metadata.json").exists()
        and (preprocessed_dir / "pipeline.pkl").exists()
        and (preprocessed_dir / "feature_map.json").exists()
    )
