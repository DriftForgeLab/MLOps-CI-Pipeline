import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.config.loader import (
    TrainingConfig,
    ModelConfig,
    RandomForestHyperparams,
    LogisticRegressionHyperparams,
)
from src.training.model_factory import create_model


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rf_config(**overrides):
    hp = RandomForestHyperparams(**overrides)
    return TrainingConfig(model=ModelConfig(algorithm="random_forest", hyperparameters=hp))


def _lr_config(**overrides):
    hp = LogisticRegressionHyperparams(**overrides)
    return TrainingConfig(model=ModelConfig(algorithm="logistic_regression", hyperparameters=hp))


def _linreg_config():
    # linear_regression has no hyperparams but model_factory expects hp attribute;
    # the factory only reads algorithm, so use a dummy LogisticRegressionHyperparams.
    hp = LogisticRegressionHyperparams()
    return TrainingConfig(model=ModelConfig(algorithm="linear_regression", hyperparameters=hp))


SEED = 42


# ── Valid algorithm + task_type combinations ─────────────────────────────────

class TestValidCombinations:
    @pytest.mark.parametrize("task_type", ["classification"])
    def test_random_forest_classifier(self, task_type):
        model = create_model(_rf_config(), task_type, SEED)
        assert isinstance(model, RandomForestClassifier)
        assert model.random_state == SEED

    def test_random_forest_regressor(self):
        model = create_model(_rf_config(), "regression", SEED)
        assert isinstance(model, RandomForestRegressor)
        assert model.random_state == SEED

    @pytest.mark.parametrize("task_type", ["classification"])
    def test_logistic_regression_classifier(self, task_type):
        model = create_model(_lr_config(), task_type, SEED)
        assert isinstance(model, LogisticRegression)
        assert model.random_state == SEED

    def test_linear_regression(self):
        model = create_model(_linreg_config(), "regression", SEED)
        assert isinstance(model, LinearRegression)


# ── Hyperparameter passthrough ───────────────────────────────────────────────

class TestHyperparameterPassthrough:
    def test_random_forest_hyperparams(self):
        cfg = _rf_config(n_estimators=200, max_depth=10, min_samples_split=5, class_weight="balanced")
        model = create_model(cfg, "classification", SEED)
        assert model.n_estimators == 200
        assert model.max_depth == 10
        assert model.min_samples_split == 5
        assert model.class_weight == "balanced"

    def test_logistic_regression_hyperparams(self):
        cfg = _lr_config(C=0.5, solver="saga", max_iter=500, class_weight="balanced")
        model = create_model(cfg, "classification", SEED)
        assert model.C == 0.5
        assert model.solver == "saga"
        assert model.max_iter == 500
        assert model.class_weight == "balanced"


# ── Invalid combinations ─────────────────────────────────────────────────────

class TestInvalidCombinations:
    @pytest.mark.parametrize(
        "algorithm, task_type",
        [
            ("logistic_regression", "regression"),
            ("linear_regression", "classification"),
            ("random_forest", "unknown_task"),
        ],
    )
    def test_unsupported_combination_raises(self, algorithm, task_type):
        if algorithm == "linear_regression":
            cfg = _linreg_config()
        elif algorithm == "logistic_regression":
            cfg = _lr_config()
        else:
            cfg = _rf_config()
        with pytest.raises(ValueError, match="Unsupported combination"):
            create_model(cfg, task_type, SEED)
