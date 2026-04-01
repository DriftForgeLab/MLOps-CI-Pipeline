"""Pipeline config loader — validate, build, and return a PipelineConfig."""

import logging
from pathlib import Path

from src.config.schema import (
    VALID_TASK_TYPES,
    VALID_LOG_LEVELS,
    VALID_PIPELINE_STAGES,
    REQUIRED_TOP_LEVEL_KEYS,
    REQUIRED_PROJECT_KEYS,
    REQUIRED_DATA_KEYS,
    REQUIRED_CONFIGS_KEYS,
    ProjectMeta,
    DataPaths,
    SubConfigPaths,
    MLflowConfig,
    PipelineConfig,
)
from src.config.validation import _load_yaml, _validate_enum, _validate_section

logger = logging.getLogger(__name__)


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
            name=raw["project"]["name"],
            version=raw["project"]["version"],
        ),
        task_type=raw["task_type"],
        random_seed=raw["random_seed"],
        pipeline_stages=tuple(raw["pipeline_stages"]),
        output_dir=raw["output_dir"],
        data=DataPaths(
            raw=raw["data"]["raw"],
            processed=raw["data"]["processed"],
            evaluation=raw["data"]["evaluation"],
            drift_scenarios=raw["data"]["drift_scenarios"],
        ),
        configs=SubConfigPaths(
            preprocessing=raw["configs"]["preprocessing"],
            training=raw["configs"]["training"],
            evaluation=raw["configs"]["evaluation"],
            promotion=raw["configs"]["promotion"],
            deployment=raw["configs"]["deployment"],
        ),
        log_level=raw["log_level"],
        dataset=raw["dataset"],
        mlflow=mlflow_cfg,
    )


def load_config(path: Path) -> PipelineConfig:
    """
    Load, validate, and return a PipelineConfig from a YAML file.

    This is the single entry point for pipeline configuration loading. It guarantees
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
