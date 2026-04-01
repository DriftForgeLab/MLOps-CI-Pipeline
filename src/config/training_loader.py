"""Training config loader — validate, build, and return a TrainingConfig."""

import logging
from pathlib import Path

from src.config.schema import (
    VALID_ALGORITHMS,
    VALID_SOLVERS,
    VALID_CLASS_WEIGHTS,
    REQUIRED_MODEL_KEYS,
    _OPTIONAL_MODEL_KEYS,
    RF_HYPERPARAMS_KEYS,
    LR_HYPERPARAMS_KEYS,
    CNN_HYPERPARAMS_KEYS,
    CNN_ARCHITECTURE_KEYS,
    CNN_CONV_LAYER_KEYS,
    RandomForestHyperparams,
    LogisticRegressionHyperparams,
    LinearRegressionHyperparams,
    CnnConvLayerConfig,
    CnnArchitectureConfig,
    CnnHyperparams,
    ModelConfig,
    TrainingConfig,
)
from src.config.validation import _load_yaml, _validate_positive_int, _validate_enum

logger = logging.getLogger(__name__)


def _validate_training(raw: dict) -> list[str]:
    """
    Validate the training config section of the raw dict.

    Returns a list of error strings. An empty list means the training config is valid.
    """
    errors = []

    if "model" not in raw:
        errors.append("Missing required top-level key: 'model'")
        return errors  # Cannot continue without the model block

    if not isinstance(raw["model"], dict):
        errors.append("'model' must be a mapping")
        return errors

    model = raw["model"]

    missing_model = REQUIRED_MODEL_KEYS - model.keys()
    if missing_model:
        errors.append(f"Missing required keys in 'model': {', '.join(sorted(missing_model))}")
        return errors  # Cannot validate further without required keys

    extra_model = model.keys() - REQUIRED_MODEL_KEYS - _OPTIONAL_MODEL_KEYS
    if extra_model:
        logger.warning("Unknown keys in 'model': %s", ", ".join(sorted(extra_model)))

    algorithm = model.get("algorithm")
    if algorithm is None:
        errors.append("Missing required key in 'model': 'algorithm'")
    else:
        _validate_enum(algorithm, VALID_ALGORITHMS, "algorithm", errors)

    if not isinstance(model["hyperparameters"], dict):
        errors.append("'model.hyperparameters' must be a mapping")
        return errors

    hp = model["hyperparameters"]

    # Algorithm-specific validation — only runs when algorithm is recognised
    if algorithm == "random_forest":
        extra_hp = hp.keys() - RF_HYPERPARAMS_KEYS
        if extra_hp:
            logger.warning(
                "Unknown keys in 'hyperparameters' for random_forest: %s",
                ", ".join(sorted(extra_hp)),
            )

        if "n_estimators" in hp:
            _validate_positive_int(hp["n_estimators"], "hyperparameters.n_estimators", errors)

        if "max_depth" in hp:
            _validate_positive_int(hp["max_depth"], "hyperparameters.max_depth", errors, allow_null=True)

        if "min_samples_split" in hp:
            _validate_positive_int(hp["min_samples_split"], "hyperparameters.min_samples_split", errors, min_val=2)

        if "class_weight" in hp and hp["class_weight"] is not None:
            _validate_enum(hp["class_weight"], VALID_CLASS_WEIGHTS, "class_weight", errors)

    elif algorithm == "logistic_regression":
        extra_hp = hp.keys() - LR_HYPERPARAMS_KEYS
        if extra_hp:
            logger.warning(
                "Unknown keys in 'hyperparameters' for logistic_regression: %s",
                ", ".join(sorted(extra_hp)),
            )

        if "C" in hp:
            v = hp["C"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                errors.append(
                    f"'hyperparameters.C' must be a positive number, got {v!r}"
                )

        if "solver" in hp:
            _validate_enum(hp["solver"], VALID_SOLVERS, "solver", errors)

        if "max_iter" in hp:
            _validate_positive_int(hp["max_iter"], "hyperparameters.max_iter", errors)

        if "class_weight" in hp and hp["class_weight"] is not None:
            _validate_enum(hp["class_weight"], VALID_CLASS_WEIGHTS, "class_weight", errors)

    elif algorithm == "linear_regression":
        if hp:
            logger.warning(
                "Hyperparameters provided for linear_regression will be ignored: %s",
                ", ".join(sorted(hp.keys())),
            )

    elif algorithm == "cnn":
        # --- Validate CNN hyperparameters ---
        extra_hp = hp.keys() - CNN_HYPERPARAMS_KEYS
        if extra_hp:
            logger.warning(
                "Unknown keys in 'hyperparameters' for cnn: %s",
                ", ".join(sorted(extra_hp)),
            )

        if "epochs" in hp:
            _validate_positive_int(hp["epochs"], "hyperparameters.epochs", errors)

        if "batch_size" in hp:
            _validate_positive_int(hp["batch_size"], "hyperparameters.batch_size", errors)

        if "learning_rate" in hp:
            v = hp["learning_rate"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
                errors.append(
                    f"'hyperparameters.learning_rate' must be a positive number, got {v!r}"
                )

        # --- Validate CNN architecture ---
        if "architecture" not in model:
            errors.append("Missing required key in 'model': 'architecture' (required for cnn)")
            return errors

        if not isinstance(model["architecture"], dict):
            errors.append("'model.architecture' must be a mapping")
            return errors

        arch = model["architecture"]
        extra_arch = arch.keys() - CNN_ARCHITECTURE_KEYS
        if extra_arch:
            logger.warning(
                "Unknown keys in 'architecture': %s",
                ", ".join(sorted(extra_arch)),
            )

        if "conv_layers" not in arch:
            errors.append("Missing required key in 'architecture': 'conv_layers'")
        elif not isinstance(arch["conv_layers"], list) or len(arch["conv_layers"]) == 0:
            errors.append("'architecture.conv_layers' must be a non-empty list")
        else:
            for i, layer in enumerate(arch["conv_layers"]):
                if not isinstance(layer, dict):
                    errors.append(f"'architecture.conv_layers[{i}]' must be a mapping")
                    continue
                extra_layer = layer.keys() - CNN_CONV_LAYER_KEYS
                if extra_layer:
                    logger.warning(
                        "Unknown keys in 'conv_layers[%d]': %s", i, ", ".join(sorted(extra_layer))
                    )
                if "out_channels" in layer:
                    _validate_positive_int(layer["out_channels"], f"conv_layers[{i}].out_channels", errors)
                else:
                    errors.append(f"Missing required key in 'conv_layers[{i}]': 'out_channels'")
                if "kernel_size" in layer:
                    _validate_positive_int(layer["kernel_size"], f"conv_layers[{i}].kernel_size", errors)
                else:
                    errors.append(f"Missing required key in 'conv_layers[{i}]': 'kernel_size'")

        if "fc_units" in arch:
            _validate_positive_int(arch["fc_units"], "architecture.fc_units", errors)
        else:
            errors.append("Missing required key in 'architecture': 'fc_units'")

        if "dropout" in arch:
            v = arch["dropout"]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0.0 <= v <= 1.0):
                errors.append(
                    f"'architecture.dropout' must be a number between 0.0 and 1.0, got {v!r}"
                )
        else:
            errors.append("Missing required key in 'architecture': 'dropout'")

    return errors


def _build_training_config(raw: dict) -> TrainingConfig:
    """
    Construct a TrainingConfig from a validated raw dict.

    This function assumes _validate_training has already passed with no errors.
    It performs pure mapping from dict keys to dataclass fields, using dataclass
    defaults for any optional hyperparameter fields absent from the YAML.
    """
    model = raw["model"]
    algorithm = model["algorithm"]
    hp = model.get("hyperparameters", {})

    if algorithm == "random_forest":
        hyperparams = RandomForestHyperparams(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", None),
            min_samples_split=hp.get("min_samples_split", 2),
            class_weight=hp.get("class_weight", None),
        )
    elif algorithm == "logistic_regression":
        hyperparams = LogisticRegressionHyperparams(
            C=hp.get("C", 1.0),
            solver=hp.get("solver", "lbfgs"),
            max_iter=hp.get("max_iter", 200),
            class_weight=hp.get("class_weight", None),
        )
    elif algorithm == "cnn":
        hyperparams = CnnHyperparams(
            epochs=hp.get("epochs", 10),
            batch_size=hp.get("batch_size", 32),
            learning_rate=hp.get("learning_rate", 0.001),
        )
        raw_arch = model["architecture"]
        conv_layers = tuple(
            CnnConvLayerConfig(
                out_channels=layer["out_channels"],
                kernel_size=layer["kernel_size"],
            )
            for layer in raw_arch["conv_layers"]
        )
        architecture = CnnArchitectureConfig(
            conv_layers=conv_layers,
            fc_units=raw_arch["fc_units"],
            dropout=raw_arch["dropout"],
        )
        return TrainingConfig(
            model=ModelConfig(
                algorithm=algorithm,
                hyperparameters=hyperparams,
                architecture=architecture,
            )
        )
    else:  # linear_regression
        hyperparams = LinearRegressionHyperparams()

    return TrainingConfig(
        model=ModelConfig(
            algorithm=algorithm,
            hyperparameters=hyperparams,
        )
    )


def load_training_config(path: Path) -> TrainingConfig:
    """
    Load, validate, and return a TrainingConfig from a YAML file.

    Follows the same contract as load_config().

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Config file is empty, malformed, or fails validation.
    """
    logger.debug("Loading training config from: %s", path)

    raw = _load_yaml(path)

    errors = _validate_training(raw)
    if errors:
        error_report = "\n - ".join(errors)
        raise ValueError(f"Training config validation failed:\n - {error_report}")

    config = _build_training_config(raw)

    logger.debug(
        "Training config loaded successfully: algorithm=%s", config.model.algorithm
    )

    return config
