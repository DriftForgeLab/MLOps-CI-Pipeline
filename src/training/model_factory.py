# =============================================================================
# src/training/model_factory.py — Model factory
# =============================================================================
# Responsibility: Instantiate a scikit-learn estimator from a TrainingConfig.
# The factory is the single place that maps algorithm names to sklearn classes.
# All other modules receive a ready-to-use estimator — they never import sklearn
# model classes directly.
# =============================================================================

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.config.loader import TrainingConfig, RandomForestHyperparams, LogisticRegressionHyperparams


def create_model(training_config: TrainingConfig, task_type: str, random_seed: int):
    """
    Instantiate and return a scikit-learn estimator from a TrainingConfig.

    Args:
        training_config: Validated TrainingConfig from load_training_config()
        task_type:       From config.task_type — "classification" or "regression"
        random_seed:     From config.random_seed — used for reproducibility

    Returns:
        A configured scikit-learn estimator, ready to be fitted.

    Raises:
        ValueError: If the algorithm and task_type combination is not supported.
    """
    algorithm = training_config.model.algorithm
    hp = training_config.model.hyperparameters

    if algorithm == "random_forest":
        if task_type == "classification":
            return RandomForestClassifier(
                n_estimators=hp.n_estimators,
                max_depth=hp.max_depth,
                min_samples_split=hp.min_samples_split,
                class_weight=hp.class_weight,
                random_state=random_seed,
            )
        elif task_type == "regression":
            return RandomForestRegressor(
                n_estimators=hp.n_estimators,
                max_depth=hp.max_depth,
                min_samples_split=hp.min_samples_split,
                random_state=random_seed,
            )

    if algorithm == "logistic_regression":
        if task_type == "classification":
            return LogisticRegression(
                C=hp.C,
                solver=hp.solver,
                max_iter=hp.max_iter,
                class_weight=hp.class_weight,
                random_state=random_seed,
            )

    if algorithm == "linear_regression":
        if task_type == "regression":
            return LinearRegression()

    raise ValueError(
        f"Unsupported combination: algorithm='{algorithm}', task_type='{task_type}'. "
        f"Supported combinations: random_forest+classification, random_forest+regression, "
        f"logistic_regression+classification, linear_regression+regression."
    )