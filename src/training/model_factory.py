# =============================================================================
# src/training/model_factory.py — Model factory
# =============================================================================
# Responsibility: Instantiate a scikit-learn estimator from a TrainingConfig.
# The factory is the single place that maps algorithm names to sklearn classes.
# All other modules receive a ready-to-use estimator — they never import sklearn
# model classes directly.
#
# CNN models (PyTorch) are NOT handled here — they bypass the factory because
# they require fundamentally different construction (architecture config, layers).
# =============================================================================

from collections.abc import Callable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.config.schema import (
    TrainingConfig,
    RandomForestHyperparams,
    LogisticRegressionHyperparams,
    CLASSIFICATION_TASK_TYPES,
    SKLEARN_TASK_TYPES,
)


# ---------------------------------------------------------------------------
# Builder functions — one per (algorithm, task_type_group) combination
# ---------------------------------------------------------------------------

def _build_rf_classifier(hp: RandomForestHyperparams, seed: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=hp.n_estimators,
        max_depth=hp.max_depth,
        min_samples_split=hp.min_samples_split,
        class_weight=hp.class_weight,
        random_state=seed,
    )


def _build_rf_regressor(hp: RandomForestHyperparams, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=hp.n_estimators,
        max_depth=hp.max_depth,
        min_samples_split=hp.min_samples_split,
        random_state=seed,
    )


def _build_logistic_regression(hp: LogisticRegressionHyperparams, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=hp.C,
        solver=hp.solver,
        max_iter=hp.max_iter,
        class_weight=hp.class_weight,
        random_state=seed,
    )


def _build_linear_regression(hp, seed: int) -> LinearRegression:
    return LinearRegression()


# ---------------------------------------------------------------------------
# Dispatch table — derived from schema constants, not hardcoded strings
# ---------------------------------------------------------------------------

_SKLEARN_CLASSIFICATION_TYPES = SKLEARN_TASK_TYPES & CLASSIFICATION_TASK_TYPES
_SKLEARN_REGRESSION_TYPES = SKLEARN_TASK_TYPES - CLASSIFICATION_TASK_TYPES

_DISPATCH: dict[tuple[str, str], Callable] = {}

for _tt in _SKLEARN_CLASSIFICATION_TYPES:
    _DISPATCH[("random_forest", _tt)] = _build_rf_classifier
    _DISPATCH[("logistic_regression", _tt)] = _build_logistic_regression

for _tt in _SKLEARN_REGRESSION_TYPES:
    _DISPATCH[("random_forest", _tt)] = _build_rf_regressor
    _DISPATCH[("linear_regression", _tt)] = _build_linear_regression


def create_model(training_config: TrainingConfig, task_type: str, random_seed: int):
    """
    Instantiate and return a scikit-learn estimator from a TrainingConfig.

    Supported task types: classification, regression, image_classification.
    CNN task types are handled separately outside this factory.

    Args:
        training_config: Validated TrainingConfig from load_training_config()
        task_type:       One of SKLEARN_TASK_TYPES from schema.py
        random_seed:     From config.random_seed — used for reproducibility

    Returns:
        A configured scikit-learn estimator, ready to be fitted.

    Raises:
        ValueError: If the algorithm and task_type combination is not supported.
    """
    algorithm = training_config.model.algorithm
    hp = training_config.model.hyperparameters
    key = (algorithm, task_type)

    builder = _DISPATCH.get(key)
    if builder is None:
        supported = ", ".join(
            f"{alg}+{tt}" for alg, tt in sorted(_DISPATCH)
        )
        raise ValueError(
            f"Unsupported combination: algorithm='{algorithm}', task_type='{task_type}'. "
            f"Supported combinations: {supported}."
        )

    return builder(hp, random_seed)
