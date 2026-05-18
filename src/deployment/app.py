# =============================================================================
# src/deployment/app.py — FastAPI application entry point
# =============================================================================
# Responsibility: Define the FastAPI app instance, register routers, and
# expose a CLI entry point for launching the API with uvicorn.
# =============================================================================
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.deployment.startup_checks import validate_environment, load_all_production_models
from src.deployment.routes import router

logger = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate environment and load all Production models once at startup."""
    config, deploy_config = validate_environment()
    models = load_all_production_models(config, deploy_config)
    _state["models"] = models
    _state["config"] = config
    _state["deploy_config"] = deploy_config
    _state["reload_lock"] = asyncio.Lock()
    logger.info(
        "API startup complete — %d model(s) loaded (stage=%s): %s",
        len(models),
        deploy_config.model.allowed_stage,
        list(models.keys()),
    )
    yield
    _state.clear()


app = FastAPI(
    title="MLOps Prediction API",
    description="Serves predictions from the active production model.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.state.model_state = _state


def main() -> None:
    """CLI entry point — launch the API with uvicorn.

    Loads the deployment config to read host, port, and log_level.
    Falls back to defaults if the config cannot be loaded (e.g. missing file).
    """
    try:
        _, deploy_config = validate_environment()
        host = deploy_config.server.host
        port = deploy_config.server.port
        log_level = deploy_config.server.log_level
    except RuntimeError as exc:
        logger.warning("Could not load deployment config for CLI: %s — using defaults.", exc)
        host, port, log_level = "0.0.0.0", 8000, "info"

    uvicorn.run(
        "src.deployment.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
