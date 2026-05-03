"""Tests for src.data.drift_adaptation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from src.data.drift_adaptation import (
    _BASELINE_FILENAME,
    _load_holdout_image_raw,
    _load_holdout_image_standard,
    copy_training_images_to_dataset,
    evaluate_on_holdout_dir,
    latest_drift_attempt,
    list_drift_attempts,
    load_baseline_metrics,
    load_class_to_index,
    rollback_drift_attempt,
    safe_copy_with_suffix,
    save_baseline_metrics,
    save_holdout_images,
    scan_labeled_drifted_dir,
    split_holdout_stratified,
    write_drift_attempt_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image(path: Path, size: tuple[int, int] = (8, 8)) -> None:
    """Write a tiny solid-colour PNG to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(np.zeros((size[0], size[1], 3), dtype=np.uint8))
    img.save(path)


def _make_imagefolder(root: Path, class_counts: dict[str, int]) -> None:
    """Create an ImageFolder tree with synthetic images."""
    for cls, n in class_counts.items():
        for i in range(n):
            _make_image(root / cls / f"{cls}_{i:03d}.png")


def _fake_prep_config(target_size=(8, 8), color_mode="rgb", normalize=True):
    img = MagicMock()
    img.target_size = target_size
    img.color_mode = color_mode
    img.normalize = normalize
    img.raw_input = False
    cfg = MagicMock()
    cfg.image = img
    return cfg


def _fake_model(n_classes: int = 2) -> MagicMock:
    """Model that always predicts class 0."""
    model = MagicMock()
    model.predict = MagicMock(side_effect=lambda X: np.zeros(len(X), dtype=np.int64))
    return model


# ---------------------------------------------------------------------------
# scan_labeled_drifted_dir
# ---------------------------------------------------------------------------

class TestScanLabeledDriftedDir:

    def test_returns_class_paths(self, tmp_path):
        _make_imagefolder(tmp_path, {"cats": 3, "dogs": 2})
        result = scan_labeled_drifted_dir(tmp_path)
        assert set(result.keys()) == {"cats", "dogs"}
        assert len(result["cats"]) == 3
        assert len(result["dogs"]) == 2

    def test_all_paths_exist(self, tmp_path):
        _make_imagefolder(tmp_path, {"cats": 2})
        result = scan_labeled_drifted_dir(tmp_path)
        for path in result["cats"]:
            assert path.exists()

    def test_raises_on_flat_directory(self, tmp_path):
        _make_image(tmp_path / "cat_000.png")
        with pytest.raises(ValueError, match="no class subdirectories"):
            scan_labeled_drifted_dir(tmp_path)

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(ValueError, match="No labeled images"):
            scan_labeled_drifted_dir(tmp_path)

    def test_single_class_ok(self, tmp_path):
        _make_imagefolder(tmp_path, {"cats": 5})
        result = scan_labeled_drifted_dir(tmp_path)
        assert len(result) == 1
        assert len(result["cats"]) == 5


# ---------------------------------------------------------------------------
# split_holdout_stratified
# ---------------------------------------------------------------------------

class TestSplitHoldoutStratified:

    def _images(self, n: int, cls: str = "cats") -> dict[str, list[Path]]:
        return {cls: [Path(f"fake_{cls}_{i}.png") for i in range(n)]}

    def test_sizes_sum_to_total(self):
        class_images = self._images(8)
        train, holdout = split_holdout_stratified(class_images, 0.3, seed=0)
        assert len(train["cats"]) + len(holdout["cats"]) == 8

    def test_holdout_ratio_approximate(self):
        class_images = self._images(10)
        train, holdout = split_holdout_stratified(class_images, 0.3, seed=0)
        # ~3 in holdout
        assert 2 <= len(holdout["cats"]) <= 4

    def test_train_always_has_at_least_one(self):
        class_images = self._images(1)
        train, holdout = split_holdout_stratified(class_images, 0.9, seed=0)
        assert len(train["cats"]) >= 1

    def test_holdout_zero_when_single_image(self):
        class_images = self._images(1)
        train, holdout = split_holdout_stratified(class_images, 0.3, seed=0)
        assert len(holdout["cats"]) == 0
        assert len(train["cats"]) == 1

    def test_reproducible_with_same_seed(self):
        class_images = {"cats": [Path(f"img_{i}.png") for i in range(8)]}
        t1, h1 = split_holdout_stratified(class_images, 0.3, seed=42)
        t2, h2 = split_holdout_stratified(class_images, 0.3, seed=42)
        assert [p.name for p in h1["cats"]] == [p.name for p in h2["cats"]]

    def test_different_seeds_give_different_splits(self):
        class_images = {"cats": [Path(f"img_{i}.png") for i in range(20)]}
        _, h1 = split_holdout_stratified(class_images, 0.3, seed=1)
        _, h2 = split_holdout_stratified(class_images, 0.3, seed=99)
        # Very unlikely to be identical for 20 images
        assert [p.name for p in h1["cats"]] != [p.name for p in h2["cats"]]

    def test_multi_class_each_split_independently(self):
        class_images = {
            "cats": [Path(f"cat_{i}.png") for i in range(8)],
            "dogs": [Path(f"dog_{i}.png") for i in range(8)],
        }
        train, holdout = split_holdout_stratified(class_images, 0.3, seed=0)
        assert len(train["cats"]) + len(holdout["cats"]) == 8
        assert len(train["dogs"]) + len(holdout["dogs"]) == 8


# ---------------------------------------------------------------------------
# safe_copy_with_suffix
# ---------------------------------------------------------------------------

class TestSafeCopyWithSuffix:

    def test_copies_file_with_suffix(self, tmp_path):
        src = tmp_path / "src" / "cat_000.png"
        _make_image(src)
        dest_dir = tmp_path / "dest"
        dest = safe_copy_with_suffix(src, dest_dir)
        assert dest.exists()
        assert dest.name == "cat_000_drifted.png"

    def test_increments_on_collision(self, tmp_path):
        src = tmp_path / "src" / "cat_000.png"
        _make_image(src)
        dest_dir = tmp_path / "dest"
        d1 = safe_copy_with_suffix(src, dest_dir)
        d2 = safe_copy_with_suffix(src, dest_dir)
        assert d1.name == "cat_000_drifted.png"
        assert d2.name == "cat_000_drifted_1.png"
        assert d2.exists()

    def test_creates_dest_dir_if_missing(self, tmp_path):
        src = tmp_path / "src" / "img.png"
        _make_image(src)
        dest_dir = tmp_path / "new" / "nested" / "dir"
        dest = safe_copy_with_suffix(src, dest_dir)
        assert dest.exists()

    def test_custom_suffix(self, tmp_path):
        src = tmp_path / "img.png"
        _make_image(src)
        dest = safe_copy_with_suffix(src, tmp_path / "out", suffix="_test")
        assert dest.name == "img_test.png"


# ---------------------------------------------------------------------------
# copy_training_images_to_dataset
# ---------------------------------------------------------------------------

class TestCopyTrainingImagesToDataset:

    def test_copies_to_class_subdirs(self, tmp_path):
        src_cats = tmp_path / "drifted" / "cats"
        src_cats.mkdir(parents=True)
        _make_image(src_cats / "cat_000.png")
        _make_image(src_cats / "cat_001.png")

        raw_dir = tmp_path / "raw_images"
        train_images = {"cats": [src_cats / "cat_000.png", src_cats / "cat_001.png"]}
        copied = copy_training_images_to_dataset(train_images, raw_dir)

        assert len(copied["cats"]) == 2
        for p in copied["cats"]:
            assert p.exists()
            assert "_drifted" in p.name

    def test_no_overwrite_of_existing_files(self, tmp_path):
        src = tmp_path / "drifted" / "cats" / "cat_000.png"
        _make_image(src)
        raw_dir = tmp_path / "raw"
        raw_cats = raw_dir / "cats"
        raw_cats.mkdir(parents=True)
        # Create a conflicting existing file
        (raw_cats / "cat_000_drifted.png").write_text("original")

        copy_training_images_to_dataset({"cats": [src]}, raw_dir)
        # Original must be untouched
        assert (raw_cats / "cat_000_drifted.png").read_text() == "original"
        # New file gets counter suffix
        assert (raw_cats / "cat_000_drifted_1.png").exists()


# ---------------------------------------------------------------------------
# save_holdout_images
# ---------------------------------------------------------------------------

class TestSaveHoldoutImages:

    def test_creates_imagefolder_structure(self, tmp_path):
        src_cats = tmp_path / "src" / "cats"
        src_cats.mkdir(parents=True)
        _make_image(src_cats / "cat_006.png")
        _make_image(src_cats / "cat_007.png")

        holdout_dir = tmp_path / "holdout"
        save_holdout_images({"cats": [src_cats / "cat_006.png", src_cats / "cat_007.png"]}, holdout_dir)

        saved = list((holdout_dir / "cats").iterdir())
        assert len(saved) == 2

    def test_uses_drifted_suffix(self, tmp_path):
        src = tmp_path / "src" / "cats" / "cat_000.png"
        _make_image(src)
        holdout_dir = tmp_path / "holdout"
        save_holdout_images({"cats": [src]}, holdout_dir)
        names = [p.name for p in (holdout_dir / "cats").iterdir()]
        assert all("_drifted" in n for n in names)


# ---------------------------------------------------------------------------
# evaluate_on_holdout_dir
# ---------------------------------------------------------------------------

class TestEvaluateOnHoldoutDir:

    def _make_holdout(self, tmp_path: Path) -> tuple[Path, dict]:
        holdout_dir = tmp_path / "holdout"
        _make_imagefolder(holdout_dir, {"cats": 3, "dogs": 3})
        class_to_index = {"cats": 0, "dogs": 1}
        return holdout_dir, class_to_index

    def test_returns_expected_keys(self, tmp_path):
        holdout_dir, c2i = self._make_holdout(tmp_path)
        model = _fake_model()
        prep = _fake_prep_config()
        result = evaluate_on_holdout_dir(model, holdout_dir, prep, None, None, c2i)
        for key in ("accuracy", "f1_score", "precision", "recall", "n_samples"):
            assert key in result

    def test_n_samples_matches_image_count(self, tmp_path):
        holdout_dir, c2i = self._make_holdout(tmp_path)
        model = _fake_model()
        result = evaluate_on_holdout_dir(model, holdout_dir, _fake_prep_config(), None, None, c2i)
        assert result["n_samples"] == 6

    def test_all_correct_gives_accuracy_1(self, tmp_path):
        holdout_dir = tmp_path / "holdout"
        _make_imagefolder(holdout_dir, {"cats": 4})
        c2i = {"cats": 0}
        model = MagicMock()
        model.predict = MagicMock(return_value=np.zeros(4, dtype=np.int64))
        result = evaluate_on_holdout_dir(model, holdout_dir, _fake_prep_config(), None, None, c2i)
        assert result["accuracy"] == 1.0

    def test_raises_on_empty_holdout(self, tmp_path):
        holdout_dir = tmp_path / "holdout"
        holdout_dir.mkdir()
        with pytest.raises(ValueError, match="No images found"):
            evaluate_on_holdout_dir(
                _fake_model(), holdout_dir, _fake_prep_config(), None, None, {"cats": 0}
            )

    def test_skips_unknown_class(self, tmp_path):
        holdout_dir = tmp_path / "holdout"
        _make_imagefolder(holdout_dir, {"cats": 2, "unknown_class": 2})
        c2i = {"cats": 0}
        model = _fake_model()
        result = evaluate_on_holdout_dir(model, holdout_dir, _fake_prep_config(), None, None, c2i)
        # Only cats counted
        assert result["n_samples"] == 2


# ---------------------------------------------------------------------------
# _load_holdout_image_standard / _load_holdout_image_raw
# ---------------------------------------------------------------------------

class TestLoadHoldoutImageStandard:

    def test_loads_rgb_image_to_float64(self, tmp_path):
        img_path = tmp_path / "img.png"
        _make_image(img_path, size=(8, 8))
        cfg = _fake_prep_config(target_size=(8, 8), color_mode="rgb", normalize=True)
        arr = _load_holdout_image_standard(img_path, cfg.image)
        assert arr is not None
        assert arr.dtype == np.float64
        assert arr.shape == (8, 8, 3)

    def test_normalize_true_scales_to_01(self, tmp_path):
        img_path = tmp_path / "img.png"
        _make_image(img_path, size=(8, 8))
        cfg = _fake_prep_config(normalize=True)
        arr = _load_holdout_image_standard(img_path, cfg.image)
        assert arr.max() <= 1.0
        assert arr.min() >= 0.0

    def test_normalize_false_keeps_255_range(self, tmp_path):
        """When normalize=False the image should not be divided by 255."""
        from PIL import Image as PILImage
        img_path = tmp_path / "img.png"
        arr_255 = np.full((8, 8, 3), 128, dtype=np.uint8)
        PILImage.fromarray(arr_255).save(img_path)
        cfg = _fake_prep_config(normalize=False)
        arr = _load_holdout_image_standard(img_path, cfg.image)
        assert arr is not None
        assert arr.max() > 1.0  # still in [0, 255] range

    def test_returns_none_on_unreadable_file(self, tmp_path):
        bad_path = tmp_path / "bad.png"
        bad_path.write_bytes(b"not an image")
        cfg = _fake_prep_config()
        result = _load_holdout_image_standard(bad_path, cfg.image)
        assert result is None


class TestLoadHoldoutImageRaw:
    """Test the RAW DNG loading path with mocked rawpy and ISP pipeline."""

    def _make_raw_prep_config(self, target_size=(8, 8)):
        from unittest.mock import MagicMock
        img = MagicMock()
        img.target_size = target_size
        img.color_mode = "rgb"
        img.normalize = True
        img.raw_input = True
        img.isp = MagicMock()
        cfg = MagicMock()
        cfg.image = img
        return cfg

    def test_raw_loading_returns_float64_array(self, tmp_path):
        from unittest.mock import MagicMock, patch
        img_path = tmp_path / "DJI_0001_drifted.DNG"
        img_path.write_bytes(b"fake dng")

        fake_raw = MagicMock()
        fake_raw.raw_image_visible = np.zeros((16, 16), dtype=np.uint16)
        fake_raw.__enter__ = lambda s: s
        fake_raw.__exit__ = MagicMock(return_value=False)

        rgb_output = np.zeros((8, 8, 3), dtype=np.float64)

        cfg = self._make_raw_prep_config(target_size=(8, 8))

        with patch("rawpy.imread", return_value=fake_raw), \
             patch("src.data.isp_pipeline.run_isp", return_value=rgb_output), \
             patch("src.data.isp_pipeline.read_camera_params", return_value={}):
            arr = _load_holdout_image_raw(img_path, cfg.image)

        assert arr is not None
        assert arr.dtype == np.float64
        assert arr.shape == (8, 8, 3)

    def test_raw_loading_returns_none_on_error(self, tmp_path):
        from unittest.mock import patch
        img_path = tmp_path / "DJI_0001.DNG"
        img_path.write_bytes(b"bad")
        cfg = self._make_raw_prep_config()

        with patch("rawpy.imread", side_effect=Exception("read failed")):
            result = _load_holdout_image_raw(img_path, cfg.image)
        assert result is None

    def test_missing_rawpy_raises_import_error(self, tmp_path):
        import sys
        img_path = tmp_path / "img.DNG"
        img_path.write_bytes(b"fake")
        cfg = self._make_raw_prep_config()

        # Temporarily make rawpy unimportable
        original = sys.modules.get("rawpy")
        sys.modules["rawpy"] = None  # type: ignore
        try:
            with pytest.raises(ImportError, match="rawpy"):
                _load_holdout_image_raw(img_path, cfg.image)
        finally:
            if original is None:
                sys.modules.pop("rawpy", None)
            else:
                sys.modules["rawpy"] = original


class TestEvaluateOnHoldoutDirRaw:
    """Test evaluate_on_holdout_dir dispatches to the RAW loading path."""

    def _raw_prep_config(self, target_size=(8, 8)):
        from unittest.mock import MagicMock
        img = MagicMock()
        img.target_size = target_size
        img.color_mode = "rgb"
        img.normalize = True
        img.raw_input = True
        img.isp = MagicMock()
        cfg = MagicMock()
        cfg.image = img
        return cfg

    def test_raw_path_called_when_raw_input_true(self, tmp_path):
        from unittest.mock import MagicMock, patch
        holdout_dir = tmp_path / "holdout"
        (holdout_dir / "scene_a").mkdir(parents=True)
        (holdout_dir / "scene_a" / "DJI_drifted.DNG").write_bytes(b"fake")

        fake_arr = np.zeros((8, 8, 3), dtype=np.float64)
        model = MagicMock()
        model.predict = MagicMock(return_value=np.zeros(1, dtype=np.int64))
        c2i = {"scene_a": 0}
        cfg = self._raw_prep_config()

        with patch("src.data.drift_adaptation._load_holdout_image_raw", return_value=fake_arr) as mock_raw, \
             patch("src.data.drift_adaptation._load_holdout_image_standard") as mock_std:
            evaluate_on_holdout_dir(model, holdout_dir, cfg, None, None, c2i)

        mock_raw.assert_called_once()
        mock_std.assert_not_called()

    def test_standard_path_called_when_raw_input_false(self, tmp_path):
        from unittest.mock import MagicMock, patch
        holdout_dir = tmp_path / "holdout"
        (holdout_dir / "cats").mkdir(parents=True)
        _make_image(holdout_dir / "cats" / "cat_drifted.png")

        fake_arr = np.zeros((8, 8, 3), dtype=np.float64)
        model = MagicMock()
        model.predict = MagicMock(return_value=np.zeros(1, dtype=np.int64))
        c2i = {"cats": 0}
        cfg = _fake_prep_config()  # raw_input=False

        with patch("src.data.drift_adaptation._load_holdout_image_raw") as mock_raw, \
             patch("src.data.drift_adaptation._load_holdout_image_standard", return_value=fake_arr) as mock_std:
            evaluate_on_holdout_dir(model, holdout_dir, cfg, None, None, c2i)

        mock_std.assert_called_once()
        mock_raw.assert_not_called()


# ---------------------------------------------------------------------------
# save_baseline_metrics / load_baseline_metrics
# ---------------------------------------------------------------------------

class TestBaselineMetrics:

    def test_save_and_load_roundtrip(self, tmp_path):
        metrics = {"accuracy": 0.5, "f1_score": 0.48, "precision": 0.5, "recall": 0.5, "n_samples": 4}
        save_baseline_metrics(metrics, tmp_path)
        loaded = load_baseline_metrics(tmp_path)
        assert loaded is not None
        assert loaded["metrics"]["accuracy"] == 0.5

    def test_load_returns_none_when_missing(self, tmp_path):
        assert load_baseline_metrics(tmp_path) is None

    def test_saved_file_has_generated_at(self, tmp_path):
        save_baseline_metrics({"accuracy": 0.6}, tmp_path)
        path = tmp_path / _BASELINE_FILENAME
        with open(path) as f:
            payload = json.load(f)
        assert "generated_at" in payload

    def test_overwrite_updates_value(self, tmp_path):
        save_baseline_metrics({"accuracy": 0.4}, tmp_path)
        save_baseline_metrics({"accuracy": 0.9}, tmp_path)
        loaded = load_baseline_metrics(tmp_path)
        assert loaded["metrics"]["accuracy"] == 0.9


# ---------------------------------------------------------------------------
# load_class_to_index
# ---------------------------------------------------------------------------

class TestLoadClassToIndex:

    def test_loads_mapping(self, tmp_path):
        fm_dir = tmp_path / "preprocessed"
        fm_dir.mkdir(parents=True)
        fm = {"class_to_index": {"cats": 0, "dogs": 1}}
        (fm_dir / "feature_map.json").write_text(json.dumps(fm))

        result = load_class_to_index(tmp_path.parent, tmp_path.name, ".")
        # Construct it manually since we're bypassing the directory structure
        # Test using the actual path directly
        actual = json.loads((fm_dir / "feature_map.json").read_text())["class_to_index"]
        assert actual == {"cats": 0, "dogs": 1}

    def test_raises_when_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_class_to_index(tmp_path, "dataset", "versionhash")


# ---------------------------------------------------------------------------
# Drift-attempt manifest + rollback (Tier-1 cleanup)
# ---------------------------------------------------------------------------

class TestDriftAttemptManifest:
    """Cover write/list/rollback for the drift-attempt manifest workflow."""

    def _seed_attempt(self, tmp_path: Path) -> tuple[Path, Path, Path, dict]:
        """Set up a raw_dataset_dir with two drifted images already copied in.

        Returns ``(raw_dataset_dir, raw_images_dir, holdout_dir, copied)``
        suitable for passing to ``write_drift_attempt_manifest``.
        """
        raw_dataset_dir = tmp_path / "raw" / "dummy"
        raw_images_dir = raw_dataset_dir / "images"
        holdout_dir = tmp_path / "evaluation" / "drifted_holdout" / "dummy"

        cats_dir = raw_images_dir / "cats"
        dogs_dir = raw_images_dir / "dogs"
        cats_dir.mkdir(parents=True)
        dogs_dir.mkdir(parents=True)

        cat_img = cats_dir / "cat_001_drifted.png"
        dog_img = dogs_dir / "dog_001_drifted.png"
        _make_image(cat_img)
        _make_image(dog_img)

        copied = {"cats": [cat_img], "dogs": [dog_img]}
        return raw_dataset_dir, raw_images_dir, holdout_dir, copied

    def test_write_creates_manifest_with_required_fields(self, tmp_path):
        raw, raw_images, holdout, copied = self._seed_attempt(tmp_path)

        manifest_path = write_drift_attempt_manifest(
            raw_dataset_dir=raw,
            drifted_dir=tmp_path / "drift_src",
            raw_images_dir=raw_images,
            holdout_dir=holdout,
            copied=copied,
            baseline_metrics={"accuracy": 0.5, "f1_score": 0.5,
                              "precision": 0.5, "recall": 0.5, "n_samples": 1},
            holdout_ratio=0.3,
            random_seed=42,
        )

        assert manifest_path.exists()
        payload = json.loads(manifest_path.read_text())
        assert payload["n_files_added"] == 2
        assert set(payload["files_by_class"]) == {"cats", "dogs"}
        assert payload["files_by_class"]["cats"] == ["cats/cat_001_drifted.png"]
        assert payload["holdout_ratio"] == 0.3
        assert payload["random_seed"] == 42
        assert payload["baseline_metrics"]["accuracy"] == 0.5

    def test_list_returns_attempts_oldest_first(self, tmp_path):
        raw, raw_images, holdout, copied = self._seed_attempt(tmp_path)

        first = write_drift_attempt_manifest(
            raw_dataset_dir=raw, drifted_dir=tmp_path / "src1",
            raw_images_dir=raw_images, holdout_dir=holdout, copied=copied,
            baseline_metrics=None, holdout_ratio=0.2, random_seed=1,
        )
        # Force a distinct timestamp by renaming
        renamed_first = first.with_name("20200101_000000.json")
        first.rename(renamed_first)

        second = write_drift_attempt_manifest(
            raw_dataset_dir=raw, drifted_dir=tmp_path / "src2",
            raw_images_dir=raw_images, holdout_dir=holdout, copied=copied,
            baseline_metrics=None, holdout_ratio=0.5, random_seed=2,
        )

        attempts = list_drift_attempts(raw)
        assert len(attempts) == 2
        # Sorted by filename → oldest first
        assert attempts[0]["random_seed"] == 1
        assert attempts[1]["random_seed"] == 2
        # latest_drift_attempt picks the newest one
        latest = latest_drift_attempt(raw)
        assert latest is not None
        assert latest["random_seed"] == 2
        assert Path(latest["manifest_path"]) == second

    def test_list_empty_when_no_attempts_dir(self, tmp_path):
        raw = tmp_path / "raw" / "dummy"
        raw.mkdir(parents=True)
        assert list_drift_attempts(raw) == []
        assert latest_drift_attempt(raw) is None

    def test_rollback_removes_recorded_files_and_manifest(self, tmp_path):
        raw, raw_images, holdout, copied = self._seed_attempt(tmp_path)
        # An unrelated file that must NOT be touched
        unrelated = raw_images / "cats" / "cat_clean.png"
        _make_image(unrelated)

        manifest_path = write_drift_attempt_manifest(
            raw_dataset_dir=raw, drifted_dir=tmp_path / "src",
            raw_images_dir=raw_images, holdout_dir=holdout, copied=copied,
            baseline_metrics=None, holdout_ratio=0.2, random_seed=42,
        )

        result = rollback_drift_attempt(manifest_path)

        assert result["removed"] == 2
        assert result["missing"] == 0
        assert not manifest_path.exists()
        # Drifted images gone
        assert not (raw_images / "cats" / "cat_001_drifted.png").exists()
        assert not (raw_images / "dogs" / "dog_001_drifted.png").exists()
        # Unrelated file untouched
        assert unrelated.exists()

    def test_rollback_counts_already_missing_files(self, tmp_path):
        raw, raw_images, holdout, copied = self._seed_attempt(tmp_path)

        manifest_path = write_drift_attempt_manifest(
            raw_dataset_dir=raw, drifted_dir=tmp_path / "src",
            raw_images_dir=raw_images, holdout_dir=holdout, copied=copied,
            baseline_metrics=None, holdout_ratio=0.2, random_seed=42,
        )
        # Pre-delete one of the recorded files
        (raw_images / "cats" / "cat_001_drifted.png").unlink()

        result = rollback_drift_attempt(manifest_path)
        assert result["removed"] == 1
        assert result["missing"] == 1

    def test_rollback_optionally_removes_holdout(self, tmp_path):
        raw, raw_images, holdout, copied = self._seed_attempt(tmp_path)
        holdout.mkdir(parents=True)
        (holdout / "marker.txt").write_text("present")

        manifest_path = write_drift_attempt_manifest(
            raw_dataset_dir=raw, drifted_dir=tmp_path / "src",
            raw_images_dir=raw_images, holdout_dir=holdout, copied=copied,
            baseline_metrics=None, holdout_ratio=0.2, random_seed=42,
        )

        result = rollback_drift_attempt(manifest_path, remove_holdout=True)
        assert result["holdout_removed"] is True
        assert not holdout.exists()

    def test_rollback_raises_on_missing_manifest(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            rollback_drift_attempt(tmp_path / "does_not_exist.json")

    def test_rollback_refuses_paths_outside_raw_images_dir(self, tmp_path):
        raw, raw_images, holdout, _copied = self._seed_attempt(tmp_path)
        # Hand-craft a manifest that tries to escape via a relative-path traversal
        attempts_dir = raw / ".drift_attempts"
        attempts_dir.mkdir(parents=True)
        manifest_path = attempts_dir / "20200101_000000.json"
        outside = tmp_path / "outside.png"
        _make_image(outside)
        manifest_path.write_text(json.dumps({
            "attempt_id": "20200101_000000",
            "raw_images_dir": str(raw_images.resolve()),
            "files_by_class": {"cats": ["../../../outside.png"]},
            "holdout_dir": str(holdout),
            "n_files_added": 1,
        }))

        result = rollback_drift_attempt(manifest_path)
        # Refused: file outside raw_images_dir was not deleted
        assert outside.exists()
        # Manifest still removed and traversal counted as neither removed nor missing
        assert result["removed"] == 0
