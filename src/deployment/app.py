# =============================================================================
# src/deployment/app.py — FastAPI application entry point
# =============================================================================
# Responsibility: Define the FastAPI app instance, register routers, and
# expose a CLI entry point for launching the API with uvicorn.
# =============================================================================
from __future__ import annotations

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
    config = validate_environment()
    models = load_all_production_models(config)
    _state["models"] = models
    logger.info(
        "API startup complete — %d Production model(s) loaded: %s",
        len(models),
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
    """CLI entry point — launch the API with uvicorn."""
    uvicorn.run("src.deployment.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()