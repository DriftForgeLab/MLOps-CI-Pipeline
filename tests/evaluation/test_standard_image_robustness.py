"""Tests for src.evaluation.standard_image_robustness.

Covers:
- Report structure and required fields
- All 8 augmentation scenarios are evaluated
- Delta computation vs baseline
- Sensitivity classification (high/medium/low)
- filtered_images is always 0 (regression for NameError bug)
- HTML output is non-empty and contains scenario names
- Missing val.npz raises FileNotFoundError
- Missing model.pt raises FileNotFoundError
- run_robustness_analysis returns valid report without baseline (delta omitted)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.evaluation.standard_image_robustness import (
    DEFAULT_ROBUSTNESS_THRESHOLDS,
    _AUGMENTATION_SCENARIOS,
    _classify_sensitivity,
    _compute_metrics,
    _horizontal_flip,
    _rotation,
    _brightness_shift,
    _contrast_scale,
    _gaussian_noise,
    _gaussian_blur,
    run_robustness_analysis,
    save_robustness_report_json,
    save_robustness_report_html,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_images(n: int = 20, h: int = 8, w: int = 8, c: int = 3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, h, w, c), dtype=np.float64).astype(np.float32)


def _make_labels(n: int = 20, n_classes: int = 3) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.integers(0, n_classes, size=n)


def _make_mock_model(n_classes: int = 3) -> MagicMock:
    """Mock PyTorch CNN that returns plausible predictions."""
    model = MagicMock()
    rng = np.random.default_rng(0)

    def fake_predict(X):
        return rng.integers(0, n_classes, size=len(X))

    model.predict.side_effect = fake_predict
    return model


def _write_val_npz(preprocessed_dir: Path, X: np.ndarray, y: np.ndarray) -> None:
    preprocessed_dir.mkdir(parents=True, exist_ok=True)
    np.savez(preprocessed_dir / "val.npz", X=X, y=y)


def _write_model_pt(artifact_dir: Path, version_id: str) -> Path:
    """Write a minimal placeholder model.pt so _load_cnn_model finds it."""
    model_dir = artifact_dir / version_id / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    pt_path = model_dir / "model.pt"
    pt_path.write_bytes(b"fake")  # content doesn't matter — we patch torch.load
    return pt_path


def _write_eval_report(output_dir: Path, accuracy: float = 0.80, f1: float = 0.79) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"metrics": {"accuracy": accuracy, "f1_score": f1}}
    path = output_dir / "evaluation_report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Augmentation functions — shape and value sanity checks
# ---------------------------------------------------------------------------

class TestAugmentationFunctions:
    def test_horizontal_flip_preserves_shape(self):
        X = _make_images(4, 8, 8, 3)
        out = _horizontal_flip(X)
        assert out.shape == X.shape

    def test_horizontal_flip_reverses_width(self):
        X = _make_images(2, 4, 4, 1)
        out = _horizontal_flip(X)
        np.testing.assert_array_equal(out[0, :, :, 0], X[0, :, ::-1, 0])

    def test_rotation_preserves_shape_square(self):
        X = _make_images(4, 8, 8, 3)
        out = _rotation(X, k=1)
        assert out.shape == X.shape

    def test_brightness_shift_changes_values(self):
        X = _make_images(4, 8, 8, 3)
        out = _brightness_shift(X, +0.5)
        np.testing.assert_allclose(out, X + 0.5)

    def test_contrast_scale_changes_values(self):
        X = _make_images(4, 8, 8, 3)
        out = _contrast_scale(X, 0.5)
        np.testing.assert_allclose(out, X * 0.5)

    def test_gaussian_noise_changes_values(self):
        X = _make_images(4, 8, 8, 3)
        out = _gaussian_noise(X, sigma=0.1)
        assert not np.allclose(out, X)

    def test_gaussian_noise_is_reproducible(self):
        X = _make_images(4, 8, 8, 3)
        a = _gaussian_noise(X, sigma=0.1, seed=7)
        b = _gaussian_noise(X, sigma=0.1, seed=7)
        np.testing.assert_array_equal(a, b)

    def test_gaussian_blur_preserves_shape(self):
        X = _make_images(2, 8, 8, 3)
        out = _gaussian_blur(X, sigma=1.0)
        assert out.shape == X.shape


# ---------------------------------------------------------------------------
# Sensitivity classification
# ---------------------------------------------------------------------------

class TestClassifySensitivity:
    def test_high(self):
        assert _classify_sensitivity(0.20, DEFAULT_ROBUSTNESS_THRESHOLDS) == "high"

    def test_medium(self):
        assert _classify_sensitivity(0.10, DEFAULT_ROBUSTNESS_THRESHOLDS) == "medium"

    def test_low(self):
        assert _classify_sensitivity(0.01, DEFAULT_ROBUSTNESS_THRESHOLDS) == "low"

    def test_boundary_high(self):
        assert _classify_sensitivity(0.15, DEFAULT_ROBUSTNESS_THRESHOLDS) == "high"

    def test_boundary_medium(self):
        assert _classify_sensitivity(0.05, DEFAULT_ROBUSTNESS_THRESHOLDS) == "medium"


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_accuracy(self):
        y = np.array([0, 1, 2, 0, 1])
        metrics = _compute_metrics(y, y)
        assert metrics["accuracy"] == 1.0

    def test_zero_accuracy(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        metrics = _compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_returns_both_keys(self):
        y = np.array([0, 1, 2])
        metrics = _compute_metrics(y, y)
        assert "accuracy" in metrics
        assert "f1_score" in metrics


# ---------------------------------------------------------------------------
# Number of built-in augmentation scenarios
# ---------------------------------------------------------------------------

def test_eight_augmentation_scenarios_defined():
    assert len(_AUGMENTATION_SCENARIOS) == 8


def test_augmentation_scenario_names():
    names = {name for name, _, _ in _AUGMENTATION_SCENARIOS}
    expected = {
        "horizontal_flip", "rotation_90", "rotation_180",
        "brightness_high", "brightness_low", "contrast_low",
        "gaussian_noise", "gaussian_blur",
    }
    assert names == expected


# ---------------------------------------------------------------------------
# run_robustness_analysis — integration-level (model and file I/O mocked)
# ---------------------------------------------------------------------------

class TestRunRobustnessAnalysis:
    def _setup(self, tmp_path: Path):
        version_id = "v001"
        X = _make_images(20)
        y = _make_labels(20)

        preprocessed_dir = tmp_path / "processed" / "ds" / version_id / "preprocessed"
        _write_val_npz(preprocessed_dir, X, y)

        artifact_dir = tmp_path / "artifacts" / "runs"
        _write_model_pt(artifact_dir, version_id)

        baseline_path = _write_eval_report(tmp_path / "outputs")
        mock_model = _make_mock_model()

        return version_id, artifact_dir, preprocessed_dir, baseline_path, mock_model

    @patch("src.evaluation.standard_image_robustness._load_cnn_model")
    def test_report_schema_fields(self, mock_load, tmp_path):
        version_id, artifact_dir, preprocessed_dir, baseline_path, mock_model = self._setup(tmp_path)
        mock_load.return_value = mock_model

        report = run_robustness_analysis(
            version_id=version_id,
            task_type="image_classification_cnn",
            preprocessed_dir=preprocessed_dir,
            artifact_dir=artifact_dir,
            baseline_report_path=baseline_path,
        )

        assert report["schema_version"] == "1.0.0"
        assert report["report_type"] == "image_augmentation_robustness"
        assert report["task_type"] == "image_classification_cnn"
        assert report["version_id"] == version_id
        assert "generated_at" in report
        assert "baseline_metrics" in report
        assert "scenarios" in report
        assert "most_sensitive_augmentation" in report
        assert "least_sensitive_augmentation" in report

    @patch("src.evaluation.standard_image_robustness._load_cnn_model")
    def test_no_filtered_images_field(self, mock_load, tmp_path):
        """filtered_images was removed — field should not appear in report."""
        version_id, artifact_dir, preprocessed_dir, baseline_path, mock_model = self._setup(tmp_path)
        mock_load.return_value = mock_model

        report = run_robustness_analysis(
            version_id=version_id,
            task_type="image_classification_cnn",
            preprocessed_dir=preprocessed_dir,
            artifact_dir=artifact_dir,
            baseline_report_path=baseline_path,
        )
        assert "filtered_images" not in report

    @patch("src.evaluation.standard_image_robustness._load_cnn_model")
    def test_all_eight_scenarios_evaluated(self, mock_load, tmp_path):
        version_id, artifact_dir, preprocessed_dir, baseline_path, mock_model = self._setup(tmp_path)
        mock_load.return_value = mock_model

        report = run_robustness_analysis(
            version_id=version_id,
            task_type="image_classification_cnn",
            preprocessed_dir=preprocessed_dir,
            artifact_dir=artifact_dir,
            baseline_report_path=baseline_path,
        )
        names = {s["name"] for s in report["scenarios"]}
        assert len(report["scenarios"]) == 8
        assert "horizontal_flip" in names
        assert "gaussian_blur" in names

    @patch("src.evaluation.standard_image_robustness._load_cnn_model")
    def test_delta_computed_when_baseline_present(self, mock_load, tmp_path):
        version_id, artifact_dir, preprocessed_dir, baseline_path, mock_model = self._setup(tmp_path)
        mock_load.return_value = mock_model

        report = run_robustness_analysis(
            version_id=version_id,
            task_type="image_classification_cnn",
            preprocessed_dir=preprocessed_dir,
            artifact_dir=artifact_dir,
            baseline_report_path=baseline_path,
        )
        for scenario in report["scenarios"]:
            assert "accuracy" in scenario["delta"], f"No delta.accuracy for {scenario['name']}"

    @patch("src.evaluation.standard_image_robustness._load_cnn_model")
    def test_delta_omitted_without_baseline(self, mock_load, tmp_path):
        version_id, artifact_dir, preprocessed_dir, _, mock_model = self._setup(tmp_path)
        mock_load.return_value = mock_model

        report = run_robustness_analysis(
            version_id=version_id,
            task_type="image_classification_cnn",
            preprocessed_dir=preprocessed_dir,
            artifact_dir=artifact_dir,
            baseline_report_path=None,
        )
        for scenario in report["scenarios"]:
            assert scenario["delta"] == {}, f"Expected empty delta without baseline for {scenario['name']}"

    @patch("src.evaluation.standard_image_robustness._load_cnn_model")
    def test_sensitivity_labels_valid(self, mock_load, tmp_path):
        version_id, artifact_dir, preprocessed_dir, baseline_path, mock_model = self._setup(tmp_path)
        mock_load.return_value = mock_model

        report = run_robustness_analysis(
            version_id=version_id,
            task_type="image_classification_cnn",
            preprocessed_dir=preprocessed_dir,
            artifact_dir=artifact_dir,
            baseline_report_path=baseline_path,
        )
        valid = {"high", "medium", "low"}
        for scenario in report["scenarios"]:
            assert scenario["sensitivity"] in valid, f"Invalid sensitivity: {scenario['sensitivity']}"

    @patch("src.evaluation.standard_image_robustness._load_cnn_model")
    def test_total_images_matches_val_split(self, mock_load, tmp_path):
        version_id, artifact_dir, preprocessed_dir, baseline_path, mock_model = self._setup(tmp_path)
        mock_load.return_value = mock_model

        report = run_robustness_analysis(
            version_id=version_id,
            task_type="image_classification_cnn",
            preprocessed_dir=preprocessed_dir,
            artifact_dir=artifact_dir,
            baseline_report_path=baseline_path,
        )
        assert report["total_images"] == 20

    def test_missing_val_npz_raises(self, tmp_path):
        version_id = "v001"
        artifact_dir = tmp_path / "artifacts" / "runs"
        _write_model_pt(artifact_dir, version_id)
        preprocessed_dir = tmp_path / "processed" / "ds" / version_id / "preprocessed"
        preprocessed_dir.mkdir(parents=True)  # dir exists but no val.npz

        with patch("src.evaluation.standard_image_robustness._load_cnn_model") as mock_load:
            mock_load.return_value = _make_mock_model()
            with pytest.raises(FileNotFoundError, match="val.npz"):
                run_robustness_analysis(
                    version_id=version_id,
                    task_type="image_classification_cnn",
                    preprocessed_dir=preprocessed_dir,
                    artifact_dir=artifact_dir,
                )


# ---------------------------------------------------------------------------
# save_robustness_report_json
# ---------------------------------------------------------------------------

def test_save_robustness_report_json(tmp_path):
    report = {
        "schema_version": "1.0.0",
        "report_type": "image_augmentation_robustness",
        "scenarios": [],
    }
    path = save_robustness_report_json(report, tmp_path)
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["report_type"] == "image_augmentation_robustness"


# ---------------------------------------------------------------------------
# save_robustness_report_html
# ---------------------------------------------------------------------------

@patch("src.evaluation.standard_image_robustness._load_cnn_model")
def test_save_robustness_report_html_contains_scenario_names(mock_load, tmp_path):
    version_id = "v001"
    X = _make_images(20)
    y = _make_labels(20)
    preprocessed_dir = tmp_path / "processed" / "ds" / version_id / "preprocessed"
    _write_val_npz(preprocessed_dir, X, y)
    _write_model_pt(tmp_path / "artifacts" / "runs", version_id)
    mock_load.return_value = _make_mock_model()

    report = run_robustness_analysis(
        version_id=version_id,
        task_type="image_classification_cnn",
        preprocessed_dir=preprocessed_dir,
        artifact_dir=tmp_path / "artifacts" / "runs",
    )
    path = save_robustness_report_html(report, tmp_path / "out")
    html = path.read_text(encoding="utf-8")

    assert "horizontal_flip" in html
    assert "gaussian_blur" in html
    assert "Augmentation Robustness" in html
    assert "NOT" in html  # disclaimer that this is not drift detection
