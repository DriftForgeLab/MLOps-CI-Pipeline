"""Deployment config loader — validate, build, and return a DeploymentConfig."""

import logging
from pathlib import Path

from src.config.schema import (
    VALID_SERVER_LOG_LEVELS,
    VALID_MODEL_STAGES,
    ServerConfig,
    ModelServingConfig,
    HealthcheckConfig,
    ReloadConfig,
    DeploymentConfig,
)
from src.config.validation import (
    _load_yaml,
    _validate_positive_int,
    _validate_enum,
    _validate_bool,
)

logger = logging.getLogger(__name__)


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

    reload = raw.get("reload", {})
    if not isinstance(reload, dict):
        errors.append("'reload' must be a mapping")
    else:
        _validate_bool(reload, "enabled", errors, prefix="reload.")
        if "timeout_seconds" in reload:
            _validate_positive_int(reload["timeout_seconds"], "reload.timeout_seconds", errors)

    return errors


def _build_deployment_config(raw: dict) -> DeploymentConfig:
    server_raw = raw.get("server", {}) or {}
    model_raw = raw.get("model", {}) or {}
    hc_raw = raw.get("healthcheck", {}) or {}
    reload_raw = raw.get("reload", {}) or {}

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
        reload=ReloadConfig(
            enabled=reload_raw.get("enabled", False),
            url=reload_raw.get("url", "http://localhost:8000/admin/reload"),
            timeout_seconds=reload_raw.get("timeout_seconds", 5),
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
