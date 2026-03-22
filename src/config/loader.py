# =============================================================================
# src/config/loader.py — Configuration loading and validation
# =============================================================================
# Responsibility: Read a YAML config file from disk, validate its contents,
# and return a structured, trustworthy config object to the rest of the system.
#
# This module is the ONLY place that touches raw YAML parsing and validation.
# All other modules receive an already-validated config object.
#
# Design decision: Config is represented as a dataclass (not a raw dict).
# This gives type safety, clear documentation of expected fields,
# IDE autocomplete, and makes it obvious what the config contains.
#
# Design decision: Validation collects ALL errors before raising, so the user
# sees every problem in one pass rather than fixing them one at a time.
# =============================================================================

import logging
from pathlib import Path

# --- Types layer (dataclasses + constants) ---
from src.config.schema import (
    # Pipeline constants
    VALID_TASK_TYPES,
    VALID_LOG_LEVELS,
    VALID_PIPELINE_STAGES,
    VALID_ALGORITHMS,
    VALID_SOLVERS,
    VALID_CLASS_WEIGHTS,
    REQUIRED_MODEL_KEYS,
    RF_HYPERPARAMS_KEYS,
    LR_HYPERPARAMS_KEYS,
    REQUIRED_TOP_LEVEL_KEYS,
    # Preprocessing constants
    VALID_ENCODING_STRATEGIES,
    VALID_HANDLE_UNKNOWN,
    VALID_SCALING_STRATEGIES,
    VALID_MISSING_POLICIES,
    VALID_NUMERIC_IMPUTE_STRATEGIES,
    VALID_CATEGORICAL_IMPUTE_STRATEGIES,
    VALID_IMAGE_COLOR_MODES,
    REQUIRED_PROJECT_KEYS,
    REQUIRED_DATA_KEYS,
    REQUIRED_CONFIGS_KEYS,
    _PREPROCESSING_TOP_LEVEL_KEYS,
    _IMAGE_KEYS,
    _IMAGE_AUGMENTATION_KEYS,
    _ENCODING_KEYS,
    _SCALING_KEYS,
    _MISSING_VALUES_KEYS,
    # Promotion constants
    VALID_OPERATORS,
    _REQUIRED_RULE_KEYS,
    # Deployment constants
    VALID_SERVER_LOG_LEVELS,
    VALID_MODEL_STAGES,
    # Pipeline dataclasses
    ProjectMeta,
    DataPaths,
    SubConfigPaths,
    RandomForestHyperparams,
    LogisticRegressionHyperparams,
    LinearRegressionHyperparams,
    ModelConfig,
    TrainingConfig,
    MLflowConfig,
    PipelineConfig,
    # Preprocessing dataclasses
    EncodingConfig,
    ScalingConfig,
    MissingValuesConfig,
    ImageAugmentationConfig,
    ImagePreprocessingConfig,
    PreprocessingConfig,
    # Evaluation dataclasses
    ClassificationEvalConfig,
    RegressionEvalConfig,
    EvaluationConfig,
    # Promotion dataclasses
    PromotionRule,
    PromotionTaskConfig,
    PromotionConfig,
    # Deployment dataclasses
    ServerConfig,
    ModelServingConfig,
    HealthcheckConfig,
    DeploymentConfig,
)

# --- Helpers layer (validation primitives) ---
from src.config.validation import (
    _load_yaml,
    _validate_positive_int,
    _validate_enum,
    _validate_section,
    _validate_bool,
    _validate_optional_section,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Pipeline domain: validate → build → load
# =============================================================================

def _validate(raw: dict) -> list[str]:
    """
    Validate raw config dict against the schema contract.

    Returns a list of error strings. An empty list means the pipeline config is valid.
    """
    errors = []

    missing_top = REQUIRED_TOP_LEVEL_KEYS - raw.keys()
    if missing_top:
        errors.append(f"Missing required top-level keys: {', '.join(sorted(missing_top))}")

    extra_top = raw.keys() - REQUIRED_TOP_LEVEL_KEYS
    if extra_top:
        logger.warning("Unknown top-level keys: %s", ", ".join(sorted(extra_top)))

    if "task_type" in raw:
        _validate_enum(raw["task_type"], VALID_TASK_TYPES, "task_type", errors)

    if "random_seed" in raw and (isinstance(raw["random_seed"], bool) or not isinstance(raw["random_seed"], int)):
        errors.append(
            f"random_seed must be an integer, got {type(raw['random_seed']).__name__}"
            )

    if "log_level" in raw:
        _validate_enum(raw["log_level"], VALID_LOG_LEVELS, "log_level", errors)

    if "project" in raw:
        _validate_section(raw, "project", REQUIRED_PROJECT_KEYS, errors)

    if "pipeline_stages" in raw:
        stages = raw["pipeline_stages"]
        if not isinstance(stages, list):
            errors.append(
                f"'pipeline_stages' must be a list, got {type(stages).__name__}"
            )
        elif len(stages) == 0:
            errors.append("'pipeline_stages' must contain at least one stage")
        else:
            invalid = [s for s in stages if s not in VALID_PIPELINE_STAGES]
            if invalid:
                errors.append(
                    f"Invalid pipeline stage(s): {', '.join(invalid)}. "
                    f"Allowed: {', '.join(sorted(VALID_PIPELINE_STAGES))}"
                )

    if "output_dir" in raw and not isinstance(raw["output_dir"], str):
        errors.append(
            f"'output_dir' must be a string, got {type(raw['output_dir']).__name__}"
        )

    if "data" in raw:
        _validate_section(raw, "data", REQUIRED_DATA_KEYS, errors)


    if "configs" in raw:
        _validate_section(raw, "configs", REQUIRED_CONFIGS_KEYS, errors)

    return errors

def _build_config(raw: dict) -> PipelineConfig:
    """
    Construct a PipelineConfig from a validated raw dict.

    This function assumes _validate has already passed with no errors.
    It performs pure mapping from dict keys to dataclass fields.
    """
    mlflow_raw = raw.get("mlflow") or {}
    registry_model_name = mlflow_raw.get("registry_model_name")
    if registry_model_name is not None and (
        not isinstance(registry_model_name, str) or not registry_model_name.strip()
    ):
        raise ValueError(
            "mlflow.registry_model_name must be a non-empty string if provided, "
            f"got {registry_model_name!r}"
        )
    mlflow_cfg = MLflowConfig(
        tracking_uri=mlflow_raw.get("tracking_uri"),
        experiment_name=mlflow_raw.get("experiment_name"),
        registry_model_name=registry_model_name,
    )

    return PipelineConfig(
        project=ProjectMeta(
            name = raw["project"]["name"],
            version = raw["project"]["version"]
        ),
        task_type = raw["task_type"],
        random_seed = raw["random_seed"],
        pipeline_stages=tuple(raw["pipeline_stages"]),
        output_dir=raw["output_dir"],
        data = DataPaths(
            raw = raw["data"]["raw"],
            processed = raw["data"]["processed"],
            evaluation = raw["data"]["evaluation"],
            drift_scenarios = raw["data"]["drift_scenarios"]
        ),
        configs = SubConfigPaths(
            preprocessing = raw["configs"]["preprocessing"],
            training = raw["configs"]["training"],
            evaluation = raw["configs"]["evaluation"],
            promotion = raw["configs"]["promotion"],
            deployment = raw["configs"]["deployment"]
        ),
        log_level = raw["log_level"],
        dataset = raw["dataset"],
        mlflow = mlflow_cfg,
    )

def load_config(path: Path) -> PipelineConfig:
    """
    Load, validate, and return a PipelineConfig from a YAML file.

    This is the single entry point for configuration loading. It guarantees
    that the returned object is complete, valid, and immutable.

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Config file is empty, malformed, or fails validation.
    """
    logger.debug("Loading config from: %s", path)

    raw = _load_yaml(path)

    errors = _validate(raw)
    if errors:
        error_report = "\n - ".join(errors)
        raise ValueError(f"Config validation failed:\n - {error_report}")

    config = _build_config(raw)

    logger.debug("Config loaded successfully: task_type=%s", config.task_type)

    return config


# =============================================================================
# Training domain: validate → build → load
# =============================================================================

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

    extra_model = model.keys() - REQUIRED_MODEL_KEYS
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

    This is the single entry point for training configuration loading.
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


# =============================================================================
# Preprocessing domain: validate → build → load
# =============================================================================

def _validate_feature_overrides(raw: dict, errors: list[str]) -> None:
    """Validate numeric_features / categorical_features lists and their overlap."""
    num_feats = raw.get("numeric_features")
    cat_feats = raw.get("categorical_features")
    for key, v in (("numeric_features", num_feats), ("categorical_features", cat_feats)):
        if v is not None:
            if not isinstance(v, list):
                errors.append(f"'{key}' must be null or a list of strings, got {type(v).__name__!r}")
            else:
                bad = [x for x in v if not isinstance(x, str) or not x.strip()]
                if bad:
                    errors.append(
                        f"'{key}' must contain non-empty strings only; bad entries: {bad!r}"
                    )
                elif len(v) != len(set(v)):
                    seen = [x for x in v if v.count(x) > 1]
                    errors.append(f"'{key}' contains duplicate entries: {sorted(set(seen))!r}")

    # Cross-field: warn if both explicit lists are provided and overlap
    if (
        isinstance(num_feats, list)
        and isinstance(cat_feats, list)
        and not errors  # only if both parsed cleanly
    ):
        overlap = set(num_feats) & set(cat_feats)
        if overlap:
            errors.append(
                f"'numeric_features' and 'categorical_features' overlap: {sorted(overlap)!r}. "
                "Each feature must appear in exactly one list."
            )


def _validate_image_augmentation(aug: dict, errors: list[str]) -> None:
    """Validate the image.augmentation sub-section."""
    extra_aug = set(aug.keys()) - _IMAGE_AUGMENTATION_KEYS
    if extra_aug:
        logger.warning(
            "Unknown keys in 'image.augmentation' (possible typo): %s",
            ", ".join(sorted(extra_aug)),
        )
    for key in ("enabled", "horizontal_flip"):
        _validate_bool(aug, key, errors, prefix="image.augmentation.")
    if "rotation_degrees" in aug:
        rd = aug["rotation_degrees"]
        if isinstance(rd, bool) or not isinstance(rd, int) or rd < 0:
            errors.append(
                f"'image.augmentation.rotation_degrees' must be a non-negative integer, got {rd!r}"
            )
    if "augmentation_factor" in aug:
        af = aug["augmentation_factor"]
        if isinstance(af, bool) or not isinstance(af, int) or af < 1:
            errors.append(
                f"'image.augmentation.augmentation_factor' must be an integer >= 1, got {af!r}"
            )


def _validate_image_section(img: dict, errors: list[str]) -> None:
    """Validate the image preprocessing sub-section."""
    extra_img = set(img.keys()) - _IMAGE_KEYS
    if extra_img:
        logger.warning(
            "Unknown keys in 'image' (possible typo): %s",
            ", ".join(sorted(extra_img)),
        )
    if "target_size" in img:
        ts = img["target_size"]
        if (
            not isinstance(ts, list)
            or len(ts) != 2
            or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in ts)
        ):
            errors.append(
                f"'image.target_size' must be a list of two positive integers, got {ts!r}"
            )
    if "color_mode" in img:
        _validate_enum(img["color_mode"], VALID_IMAGE_COLOR_MODES, "image.color_mode", errors)
    for key in ("normalize", "flatten"):
        _validate_bool(img, key, errors, prefix="image.")

    aug = img.get("augmentation")
    if aug is not None:
        if not isinstance(aug, dict):
            errors.append("'image.augmentation' must be a mapping")
        else:
            _validate_image_augmentation(aug, errors)


def _validate_preprocessing(raw: dict) -> list[str]:
    """
    Validate preprocessing.yaml against the PreprocessingConfig schema.

    Collects ALL errors before raising (fail-all-at-once pattern).
    Emits logger.warning() for unknown keys (typo guard) and for
    cross-field invariant violations (e.g. min_frequency on ordinal encoder).
    Returns a list of error strings; empty list means config is valid.
    """
    errors: list[str] = []

    # --- Unknown top-level keys ---
    extra_top = set(raw.keys()) - _PREPROCESSING_TOP_LEVEL_KEYS
    if extra_top:
        logger.warning(
            "Unknown keys in preprocessing.yaml (possible typo): %s",
            ", ".join(sorted(extra_top)),
        )

    # --- Top-level booleans ---
    for key in ("fail_on_nulls", "validate_types", "validate_labels", "validate_on_skip"):
        _validate_bool(raw, key, errors)

    # --- min_rows ---
    if "min_rows" in raw:
        v = raw["min_rows"]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            errors.append(f"'min_rows' must be a positive integer, got {v!r}")

    # --- Feature type overrides ---
    _validate_feature_overrides(raw, errors)

    # --- Encoding ---
    enc = _validate_optional_section(raw, "encoding", _ENCODING_KEYS, errors)
    if enc is not None:
        _validate_bool(enc, "enabled", errors, prefix="encoding.")
        if "strategy" in enc:
            _validate_enum(enc["strategy"], VALID_ENCODING_STRATEGIES, "encoding.strategy", errors)
        if "handle_unknown" in enc:
            _validate_enum(enc["handle_unknown"], VALID_HANDLE_UNKNOWN, "encoding.handle_unknown", errors)
        if "min_frequency" in enc and enc["min_frequency"] is not None:
            v = enc["min_frequency"]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                errors.append(
                    f"'encoding.min_frequency' must be a positive integer or null, got {v!r}"
                )
            # Cross-field: min_frequency is only meaningful for onehot
            effective_strategy = enc.get("strategy", "onehot")
            if effective_strategy != "onehot":
                logger.warning(
                    "'encoding.min_frequency' is set but 'encoding.strategy' is %r. "
                    "min_frequency only applies to 'onehot' — it will be ignored.",
                    effective_strategy,
                )

    # --- Scaling ---
    scl = _validate_optional_section(raw, "scaling", _SCALING_KEYS, errors)
    if scl is not None:
        _validate_bool(scl, "enabled", errors, prefix="scaling.")
        if "strategy" in scl:
            _validate_enum(scl["strategy"], VALID_SCALING_STRATEGIES, "scaling.strategy", errors)

    # --- Missing values ---
    mv = _validate_optional_section(raw, "missing_values", _MISSING_VALUES_KEYS, errors)
    if mv is not None:
        if "policy" in mv:
            _validate_enum(mv["policy"], VALID_MISSING_POLICIES, "missing_values.policy", errors)
        if "numeric_strategy" in mv:
            _validate_enum(
                mv["numeric_strategy"],
                VALID_NUMERIC_IMPUTE_STRATEGIES,
                "missing_values.numeric_strategy",
                errors,
            )
        if "categorical_strategy" in mv:
            _validate_enum(
                mv["categorical_strategy"],
                VALID_CATEGORICAL_IMPUTE_STRATEGIES,
                "missing_values.categorical_strategy",
                errors,
            )
        # Cross-field: warn only when an impute strategy is set to a non-default value
        # while policy != "impute". Default values (mean/most_frequent/null) are always
        # present in the YAML skeleton and should not generate noise.
        _IMPUTE_DEFAULTS = {
            "numeric_strategy": "mean",
            "categorical_strategy": "most_frequent",
            "fill_value": None,
        }
        effective_policy = mv.get("policy", "passthrough")
        if effective_policy != "impute":
            for strat_key, default_val in _IMPUTE_DEFAULTS.items():
                if strat_key in mv and mv[strat_key] != default_val:
                    logger.warning(
                        "'missing_values.%s' is set to a non-default value (%r) but "
                        "'missing_values.policy' is %r. "
                        "Impute strategies are only applied when policy='impute' — "
                        "this value will be ignored.",
                        strat_key,
                        mv[strat_key],
                        effective_policy,
                    )

    # --- Image preprocessing ---
    img = raw.get("image")
    if img is not None:
        if not isinstance(img, dict):
            errors.append("'image' must be a mapping")
        else:
            _validate_image_section(img, errors)

    return errors


def _build_image_preprocessing_config(raw: dict | None) -> ImagePreprocessingConfig | None:
    if raw is None:
        return None
    aug_raw = raw.get("augmentation", {}) or {}
    ts = raw.get("target_size", [64, 64])
    return ImagePreprocessingConfig(
        target_size=tuple(ts),
        color_mode=raw.get("color_mode", "rgb"),
        normalize=raw.get("normalize", True),
        flatten=raw.get("flatten", True),
        augmentation=ImageAugmentationConfig(
            enabled=aug_raw.get("enabled", False),
            horizontal_flip=aug_raw.get("horizontal_flip", False),
            rotation_degrees=aug_raw.get("rotation_degrees", 0),
            augmentation_factor=aug_raw.get("augmentation_factor", 1),
        ),
    )


def _build_preprocessing_config(raw: dict) -> PreprocessingConfig:
    """
    Construct a PreprocessingConfig from a validated raw dict.

    Assumes _validate_preprocessing has already passed. Applies defaults for
    any key absent from the YAML (matching field defaults in the dataclasses).
    """
    enc = raw.get("encoding", {}) or {}
    scl = raw.get("scaling", {}) or {}
    mv = raw.get("missing_values", {}) or {}

    numeric_raw = raw.get("numeric_features")
    categorical_raw = raw.get("categorical_features")

    return PreprocessingConfig(
        fail_on_nulls=raw.get("fail_on_nulls", True),
        min_rows=raw.get("min_rows", 10),
        validate_types=raw.get("validate_types", True),
        validate_labels=raw.get("validate_labels", True),
        validate_on_skip=raw.get("validate_on_skip", False),
        numeric_features=tuple(numeric_raw) if numeric_raw is not None else None,
        categorical_features=tuple(categorical_raw) if categorical_raw is not None else None,
        encoding=EncodingConfig(
            enabled=enc.get("enabled", False),
            strategy=enc.get("strategy", "onehot"),
            handle_unknown=enc.get("handle_unknown", "ignore"),
            min_frequency=enc.get("min_frequency", None),
        ),
        scaling=ScalingConfig(
            enabled=scl.get("enabled", False),
            strategy=scl.get("strategy", "standard"),
        ),
        missing_values=MissingValuesConfig(
            policy=mv.get("policy", "passthrough"),
            numeric_strategy=mv.get("numeric_strategy", "mean"),
            categorical_strategy=mv.get("categorical_strategy", "most_frequent"),
            fill_value=mv.get("fill_value", None),
        ),
        image=_build_image_preprocessing_config(raw.get("image")),
    )


def load_preprocessing_config(path: Path) -> PreprocessingConfig:
    """
    Load, validate, and return a PreprocessingConfig from a YAML file.

    Single entry point for preprocessing configuration. Follows the same
    contract as load_config() and load_training_config().

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Config file is empty, malformed, or fails validation.
    """
    logger.debug("Loading preprocessing config from: %s", path)

    raw = _load_yaml(path)

    errors = _validate_preprocessing(raw)
    if errors:
        error_report = "\n - ".join(errors)
        raise ValueError(f"Preprocessing config validation failed:\n - {error_report}")

    config = _build_preprocessing_config(raw)

    logger.debug(
        "Preprocessing config loaded: encoding=%s, scaling=%s, missing_policy=%s",
        config.encoding.enabled,
        config.scaling.enabled,
        config.missing_values.policy,
    )
    return config


# =============================================================================
# Evaluation domain: load
# =============================================================================

def load_evaluation_config(path: Path) -> EvaluationConfig:
    raw = _load_yaml(path)
    cls_raw = raw.get("classification", {}) or {}
    averaging = cls_raw.get("averaging", "weighted")
    if averaging not in {"weighted", "macro", "binary"}:
        raise ValueError(
            f"Invalid averaging strategy '{averaging}' — must be weighted, macro, or binary."
        )
    return EvaluationConfig(
        classification=ClassificationEvalConfig(averaging=averaging),
        regression=RegressionEvalConfig(),
    )


# =============================================================================
# Promotion domain: build → load
# =============================================================================

def _build_promotion_task_config(raw: dict) -> PromotionTaskConfig:
    errors: list[str] = []
    rules = []
    for i, rule in enumerate(raw.get("rules", [])):
        if not isinstance(rule, dict):
            errors.append(f"Rule at index {i}: must be a mapping, got {type(rule).__name__}")
            continue
        missing = _REQUIRED_RULE_KEYS - rule.keys()
        if missing:
            errors.append(
                f"Rule at index {i} (id={rule.get('id', '?')}): "
                f"missing required keys: {', '.join(sorted(missing))}"
            )
            continue
        if rule["operator"] not in VALID_OPERATORS:
            errors.append(
                f"Rule '{rule['id']}': invalid operator '{rule['operator']}' "
                f"— must be one of {sorted(VALID_OPERATORS)}"
            )
            continue
        rules.append(PromotionRule(
            id=rule["id"],
            metric=rule["metric"],
            threshold=float(rule["threshold"]),
            operator=rule["operator"],
            description=rule.get("description", ""),
        ))
    if errors:
        raise ValueError("Promotion config validation failed:\n  - " + "\n  - ".join(errors))
    return PromotionTaskConfig(rules=tuple(rules))


def load_promotion_config(path: Path) -> PromotionConfig:
    raw = _load_yaml(path)
    return PromotionConfig(
        classification=_build_promotion_task_config(raw.get("classification", {})),
        regression=_build_promotion_task_config(raw.get("regression", {})),
    )


# =============================================================================
# Deployment domain: validate → build → load
# =============================================================================

def _validate_deployment(raw: dict) -> list[str]:
    errors: list[str] = []

    server = raw.get("server", {})
    if not isinstance(server, dict):
        errors.append("'server' must be a mapping")
    else:
        if "port" in server:
            port = server["port"]
            if isinstance(port, bool) or not isinstance(port, int) or port < 1 or port > 65535:
                errors.append(f"'server.port' must be an integer between 1 and 65535, got {port!r}")
        if "log_level" in server:
            _validate_enum(server["log_level"], VALID_SERVER_LOG_LEVELS, "server.log_level", errors)

    model = raw.get("model", {})
    if not isinstance(model, dict):
        errors.append("'model' must be a mapping")
    else:
        if "allowed_stage" in model:
            _validate_enum(model["allowed_stage"], VALID_MODEL_STAGES, "model.allowed_stage", errors)
        _validate_bool(model, "require_production_model", errors, prefix="model.")
        if "startup_timeout_seconds" in model:
            _validate_positive_int(model["startup_timeout_seconds"], "model.startup_timeout_seconds", errors)

    healthcheck = raw.get("healthcheck", {})
    if not isinstance(healthcheck, dict):
        errors.append("'healthcheck' must be a mapping")
    else:
        _validate_bool(healthcheck, "include_model_info", errors, prefix="healthcheck.")

    return errors


def _build_deployment_config(raw: dict) -> DeploymentConfig:
    server_raw = raw.get("server", {}) or {}
    model_raw = raw.get("model", {}) or {}
    hc_raw = raw.get("healthcheck", {}) or {}

    return DeploymentConfig(
        server=ServerConfig(
            host=server_raw.get("host", "0.0.0.0"),
            port=server_raw.get("port", 8000),
            log_level=server_raw.get("log_level", "info"),
        ),
        model=ModelServingConfig(
            allowed_stage=model_raw.get("allowed_stage", "Production"),
            require_production_model=model_raw.get("require_production_model", True),
            startup_timeout_seconds=model_raw.get("startup_timeout_seconds", 120),
        ),
        healthcheck=HealthcheckConfig(
            include_model_info=hc_raw.get("include_model_info", True),
        ),
    )


def load_deployment_config(path: Path) -> DeploymentConfig:
    """Load, validate, and return a DeploymentConfig from a YAML file.

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Config file is empty, malformed, or fails validation.
    """
    logger.debug("Loading deployment config from: %s", path)

    raw = _load_yaml(path)

    errors = _validate_deployment(raw)
    if errors:
        error_report = "\n - ".join(errors)
        raise ValueError(f"Deployment config validation failed:\n - {error_report}")

    config = _build_deployment_config(raw)

    logger.debug(
        "Deployment config loaded: stage=%s, require_model=%s, port=%d",
        config.model.allowed_stage,
        config.model.require_production_model,
        config.server.port,
    )
    return config
