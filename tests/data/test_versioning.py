import yaml

import pytest

from src.data.versioning import _compute_version_id, create_dataset_version


# ── _compute_version_id determinism ──────────────────────────────────────────

class TestComputeVersionId:
    def test_same_content_same_hash(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        assert _compute_version_id(csv) == _compute_version_id(csv)

    def test_different_content_different_hash(self, tmp_path):
        csv1 = tmp_path / "a.csv"
        csv2 = tmp_path / "b.csv"
        csv1.write_text("a,b\n1,2\n", encoding="utf-8")
        csv2.write_text("a,b\n9,9\n", encoding="utf-8")
        assert _compute_version_id(csv1) != _compute_version_id(csv2)

    def test_hash_is_12_hex_chars(self, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("x\n1\n", encoding="utf-8")
        vid = _compute_version_id(csv)
        assert len(vid) == 12
        assert all(c in "0123456789abcdef" for c in vid)


# ── create_dataset_version (tabular) ─────────────────────────────────────────

class TestCreateTabularVersion:
    def _setup_raw(self, tmp_path, dataset_name="iris", task_type="classification"):
        raw_dir = tmp_path / "raw"
        ds_dir = raw_dir / dataset_name
        ds_dir.mkdir(parents=True)
        (ds_dir / "data.csv").write_text("a,b,label\n1,2,0\n3,4,1\n", encoding="utf-8")
        meta = {"task_type": task_type, "target_column": "label"}
        with open(ds_dir / "dataset.yaml", "w") as f:
            yaml.dump(meta, f)
        return raw_dir

    def test_creates_version_directory(self, tmp_path):
        raw_dir = self._setup_raw(tmp_path)
        processed = tmp_path / "processed"
        result = create_dataset_version("iris", raw_dir=raw_dir, processed_dir=processed)
        assert result.exists()
        assert (result / "data.csv").exists()
        assert (result / "dataset.yaml").exists()

    def test_version_id_in_path(self, tmp_path):
        raw_dir = self._setup_raw(tmp_path)
        processed = tmp_path / "processed"
        result = create_dataset_version("iris", raw_dir=raw_dir, processed_dir=processed)
        # path is processed/iris/<version_id>
        assert result.parent.name == "iris"
        vid = result.name
        assert len(vid) == 12

    def test_idempotent_second_call_skips(self, tmp_path):
        raw_dir = self._setup_raw(tmp_path)
        processed = tmp_path / "processed"
        v1 = create_dataset_version("iris", raw_dir=raw_dir, processed_dir=processed)
        v2 = create_dataset_version("iris", raw_dir=raw_dir, processed_dir=processed)
        assert v1 == v2

    def test_missing_dataset_yaml_raises(self, tmp_path):
        raw_dir = tmp_path / "raw"
        (raw_dir / "nope").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="dataset.yaml"):
            create_dataset_version("nope", raw_dir=raw_dir, processed_dir=tmp_path / "out")

    def test_missing_csv_raises(self, tmp_path):
        raw_dir = tmp_path / "raw"
        ds_dir = raw_dir / "ds"
        ds_dir.mkdir(parents=True)
        with open(ds_dir / "dataset.yaml", "w") as f:
            yaml.dump({"task_type": "classification"}, f)
        with pytest.raises(FileNotFoundError, match="data.csv"):
            create_dataset_version("ds", raw_dir=raw_dir, processed_dir=tmp_path / "out")


# ── create_dataset_version (image) ───────────────────────────────────────────

class TestCreateImageVersion:
    def _setup_image_raw(self, tmp_path, dataset_name="cifar"):
        raw_dir = tmp_path / "raw"
        ds_dir = raw_dir / dataset_name
        images = ds_dir / "images" / "cat"
        images.mkdir(parents=True)
        (images / "img1.png").write_bytes(b"\x89PNG fake image bytes")
        meta = {"task_type": "image_classification_cnn", "target_column": "label"}
        with open(ds_dir / "dataset.yaml", "w") as f:
            yaml.dump(meta, f)
        return raw_dir

    def test_image_version_creates_directory(self, tmp_path):
        raw_dir = self._setup_image_raw(tmp_path)
        processed = tmp_path / "processed"
        result = create_dataset_version("cifar", raw_dir=raw_dir, processed_dir=processed)
        assert result.exists()
        assert (result / "images").exists()
        assert (result / "dataset.yaml").exists()

    def test_missing_images_dir_raises(self, tmp_path):
        raw_dir = tmp_path / "raw"
        ds_dir = raw_dir / "noimg"
        ds_dir.mkdir(parents=True)
        with open(ds_dir / "dataset.yaml", "w") as f:
            yaml.dump({"task_type": "image_classification_cnn"}, f)
        with pytest.raises(FileNotFoundError, match="images"):
            create_dataset_version("noimg", raw_dir=raw_dir, processed_dir=tmp_path / "out")
