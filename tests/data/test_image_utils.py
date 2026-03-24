# =============================================================================
# tests/data/test_image_utils.py
# =============================================================================

import pytest
from pathlib import Path
from PIL import Image
import numpy as np

from src.data.image_utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    build_manifest_df,
    compute_folder_hash,
    scan_image_folder,
    validate_image_readable,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_folder(tmp_path: Path, classes: dict[str, int] | None = None) -> Path:
    """Create a minimal ImageFolder structure with synthetic images."""
    if classes is None:
        classes = {"cats": 3, "dogs": 3}

    images_dir = tmp_path / "images"
    rng = np.random.RandomState(42)
    for class_name, count in classes.items():
        class_dir = images_dir / class_name
        class_dir.mkdir(parents=True)
        for i in range(count):
            arr = rng.randint(0, 255, (4, 4, 3), dtype=np.uint8)
            Image.fromarray(arr).save(class_dir / f"img_{i:03d}.png")

    return images_dir


# ===========================================================================
# scan_image_folder
# ===========================================================================

class TestScanImageFolder:
    def test_returns_sorted_tuples(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"a": 2, "b": 2})
        entries = scan_image_folder(images_dir)

        assert len(entries) == 4
        # Sorted by class then by filename
        assert [e[1] for e in entries] == ["a", "a", "b", "b"]
        assert all(isinstance(e[0], Path) for e in entries)

    def test_filters_by_supported_extensions(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"cats": 2})
        # Add a non-image file
        (images_dir / "cats" / "notes.txt").write_text("not an image")
        (images_dir / "cats" / "data.csv").write_text("a,b\n1,2")

        entries = scan_image_folder(images_dir)
        assert len(entries) == 2
        assert all(e[0].suffix == ".png" for e in entries)

    def test_custom_expected_formats(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"cats": 2})
        # With .png filter — should find the images
        entries_png = scan_image_folder(images_dir, expected_formats=[".png"])
        assert len(entries_png) == 2

        # With .jpg filter — should find nothing (images are .png)
        entries_jpg = scan_image_folder(images_dir, expected_formats=[".jpg"])
        assert len(entries_jpg) == 0

    def test_case_insensitive_extensions(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"cats": 1})
        # Rename to uppercase extension
        src = images_dir / "cats" / "img_000.png"
        dest = images_dir / "cats" / "img_000.PNG"
        src.rename(dest)

        entries = scan_image_folder(images_dir)
        assert len(entries) == 1

    def test_skips_non_directory_entries(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"cats": 2})
        # Add a file at the top level (not a class directory)
        (images_dir / "readme.txt").write_text("should be ignored")

        entries = scan_image_folder(images_dir)
        assert len(entries) == 2

    def test_empty_class_directory(self, tmp_path):
        images_dir = tmp_path / "images"
        (images_dir / "empty_class").mkdir(parents=True)

        entries = scan_image_folder(images_dir)
        assert entries == []

    def test_empty_images_directory(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()

        entries = scan_image_folder(images_dir)
        assert entries == []


# ===========================================================================
# build_manifest_df
# ===========================================================================

class TestBuildManifestDf:
    def test_returns_dataframe_with_correct_columns(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"cats": 2})
        entries = scan_image_folder(images_dir)
        df = build_manifest_df(entries)

        assert list(df.columns) == ["file_path", "label"]
        assert len(df) == 2
        assert all(df["label"] == "cats")

    def test_empty_entries(self):
        df = build_manifest_df([])
        assert list(df.columns) == ["file_path", "label"]
        assert len(df) == 0

    def test_paths_are_strings(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"a": 1})
        entries = scan_image_folder(images_dir)
        df = build_manifest_df(entries)

        assert isinstance(df["file_path"].iloc[0], str)


# ===========================================================================
# compute_folder_hash
# ===========================================================================

class TestComputeFolderHash:
    def test_deterministic(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"cats": 3})
        hash1 = compute_folder_hash(images_dir)
        hash2 = compute_folder_hash(images_dir)
        assert hash1 == hash2

    def test_returns_12_char_hex(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"cats": 2})
        h = compute_folder_hash(images_dir)
        assert len(h) == 12
        assert all(c in "0123456789abcdef" for c in h)

    def test_different_content_different_hash(self, tmp_path):
        dir1 = tmp_path / "d1"
        dir2 = tmp_path / "d2"
        _make_image_folder(dir1, {"cats": 2})
        _make_image_folder(dir2, {"cats": 3})
        images_dir1 = dir1 / "images"
        images_dir2 = dir2 / "images"

        assert compute_folder_hash(images_dir1) != compute_folder_hash(images_dir2)

    def test_ignores_non_image_files(self, tmp_path):
        images_dir = _make_image_folder(tmp_path, {"cats": 2})
        hash_before = compute_folder_hash(images_dir)

        # Add non-image files that should be ignored
        (images_dir / "cats" / ".DS_Store").write_text("junk")
        (images_dir / "cats" / "Thumbs.db").write_bytes(b"\x00" * 100)
        (images_dir / "cats" / "readme.txt").write_text("notes")

        hash_after = compute_folder_hash(images_dir)
        assert hash_before == hash_after

    def test_empty_directory(self, tmp_path):
        images_dir = tmp_path / "images"
        images_dir.mkdir()
        h = compute_folder_hash(images_dir)
        # Should still produce a valid 12-char hash (of empty string)
        assert len(h) == 12


# ===========================================================================
# validate_image_readable
# ===========================================================================

class TestValidateImageReadable:
    def test_valid_image(self, tmp_path):
        img_path = tmp_path / "valid.png"
        arr = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        assert validate_image_readable(img_path) is True

    def test_corrupt_image(self, tmp_path):
        img_path = tmp_path / "corrupt.png"
        img_path.write_bytes(b"not a real image at all")

        assert validate_image_readable(img_path) is False

    def test_nonexistent_file(self, tmp_path):
        assert validate_image_readable(tmp_path / "missing.png") is False

    def test_truncated_image(self, tmp_path):
        # Create a valid image, then truncate it
        img_path = tmp_path / "truncated.png"
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)
        data = img_path.read_bytes()
        img_path.write_bytes(data[:len(data) // 2])

        assert validate_image_readable(img_path) is False
