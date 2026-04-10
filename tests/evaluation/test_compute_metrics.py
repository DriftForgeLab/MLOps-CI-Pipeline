import numpy as np
import pytest

from src.config.loader import ClassificationEvalConfig, EvaluationConfig, RegressionEvalConfig
from src.evaluation.evaluate import _compute_metrics


# ── helpers ─────────────────────────────────────────────────────────────────

def _eval_config(averaging="weighted"):
    return EvaluationConfig(
        classification=ClassificationEvalConfig(averaging=averaging),
        regression=RegressionEvalConfig(),
    )


# ── classification metrics ──────────────────────────────────────────────────

class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 1, 0, 1])
        metrics = _compute_metrics(y, y, "classification", _eval_config())
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        metrics = _compute_metrics(y_true, y_pred, "classification", _eval_config())
        assert metrics["accuracy"] == 0.0

    def test_metric_keys_present(self):
        y = np.array([0, 1, 0, 1])
        metrics = _compute_metrics(y, y, "classification", _eval_config())
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score", "averaging"}

    def test_averaging_passed_through(self):
        y = np.array([0, 1, 0, 1])
        metrics = _compute_metrics(y, y, "classification", _eval_config(averaging="macro"))
        assert metrics["averaging"] == "macro"

    def test_values_are_rounded_to_4_decimals(self):
        y_true = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        metrics = _compute_metrics(y_true, y_pred, "classification", _eval_config())
        for key in ("accuracy", "precision", "recall", "f1_score"):
            val = metrics[key]
            assert val == round(val, 4)

    def test_zero_division_handled(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])
        metrics = _compute_metrics(y_true, y_pred, "classification", _eval_config())
        assert metrics["accuracy"] == 1.0
        assert isinstance(metrics["precision"], float)


# ── CNN classification metrics ──────────────────────────────────────────────

class TestCNNClassificationMetrics:
    def test_includes_confusion_matrix(self):
        y = np.array([0, 1, 2, 0, 1])
        metrics = _compute_metrics(y, y, "image_classification_cnn", _eval_config())
        assert "confusion_matrix" in metrics
        assert isinstance(metrics["confusion_matrix"], list)

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        metrics = _compute_metrics(y_true, y_pred, "image_classification_cnn", _eval_config())
        cm = metrics["confusion_matrix"]
        assert len(cm) == 3
        assert all(len(row) == 3 for row in cm)


# ── regression metrics ──────────────────────────────────────────────────────

class TestRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = _compute_metrics(y, y, "regression", _eval_config())
        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0
        assert metrics["r2"] == 1.0

    def test_metric_keys_present(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = _compute_metrics(y, y, "regression", _eval_config())
        assert set(metrics.keys()) == {"mae", "mse", "rmse", "r2"}

    def test_known_error_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        metrics = _compute_metrics(y_true, y_pred, "regression", _eval_config())
        assert metrics["mae"] == 0.5
        assert metrics["mse"] == 0.25
        assert metrics["rmse"] == 0.5

    def test_values_are_rounded_to_4_decimals(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.2, 2.8, 4.3, 4.7])
        metrics = _compute_metrics(y_true, y_pred, "regression", _eval_config())
        for key in ("mae", "mse", "rmse", "r2"):
            val = metrics[key]
            assert val == round(val, 4)

    def test_rmse_is_sqrt_of_mse(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.2, 1.8, 3.3, 3.7])
        metrics = _compute_metrics(y_true, y_pred, "regression", _eval_config())
        assert metrics["rmse"] == pytest.approx(round(np.sqrt(metrics["mse"]), 4), abs=1e-4)
