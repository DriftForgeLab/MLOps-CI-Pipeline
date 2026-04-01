# =============================================================================
# src/data/image_utils.py — Image dataset utility functions
# =============================================================================
# Common utilities for scanning, manifesting, hashing, and validating image
# datasets in ImageFolder format (images/<class_name>/<file>).
# Reused by versioning, splitting, and preprocessing modules.
# =============================================================================

import hashlib
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".dng"})


def scan_image_folder(
    images_dir: Path,
    expected_formats: list[str] | None = None,
) -> list[tuple[Path, str]]:
    """Scan an ImageFolder structure and return (image_path, class_name) tuples.

    Args:
        images_dir: Path to the images/ directory containing class subdirectories.
        expected_formats: List of allowed file extensions (e.g. [".jpg", ".png"]).
                          Defaults to SUPPORTED_IMAGE_EXTENSIONS.

    Returns:
        Sorted list of (absolute_image_path, class_name) tuples.
    """
    if expected_formats is None:
        allowed = SUPPORTED_IMAGE_EXTENSIONS
    else:
        allowed = frozenset(ext.lower() for ext in expected_formats)

    entries: list[tuple[Path, str]] = []
    for class_dir in sorted(images_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in allowed:
                entries.append((img_path, class_name))

    return entries


def build_manifest_df(image_entries: list[tuple[Path, str]]) -> pd.DataFrame:
    """Build a manifest DataFrame from scan results.

    Args:
        image_entries: List of (image_path, class_name) tuples from scan_image_folder().

    Returns:
        DataFrame with columns: file_path, label.
    """
    return pd.DataFrame(
        [(str(path), label) for path, label in image_entries],
        columns=["file_path", "label"],
    )


def compute_folder_hash(
    images_dir: Path,
    expected_formats: list[str] | None = None,
) -> str:
    """Compute a SHA-256 hash of a sorted manifest (relative paths + file sizes).

    Hashing all pixel data is too slow for large datasets. Instead, hash the
    sorted list of (relative_path, file_size) pairs as a practical proxy.

    Args:
        images_dir:       Path to the images/ directory.
        expected_formats: List of allowed extensions (e.g. [".dng", ".png"]).
                          Defaults to SUPPORTED_IMAGE_EXTENSIONS when None.

    Returns:
        First 12 characters of the SHA-256 hex digest.
    """
    allowed = (
        frozenset(ext.lower() for ext in expected_formats)
        if expected_formats is not None
        else SUPPORTED_IMAGE_EXTENSIONS
    )
    entries: list[str] = []
    for img_path in sorted(images_dir.rglob("*")):
        if img_path.is_file() and img_path.suffix.lower() in allowed:
            rel = img_path.relative_to(images_dir)
            entries.append(f"{rel}:{img_path.stat().st_size}")

    manifest = "\n".join(entries)
    return hashlib.sha256(manifest.encode()).hexdigest()[:12]


def validate_image_readable(path: Path) -> bool:
    """Check if an image file can be opened with Pillow.

    Args:
        path: Path to the image file.

    Returns:
        True if the image can be opened, False otherwise.
    """
    try:
        from PIL import Image
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False
