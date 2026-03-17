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

from src.deployment.load_model import load_production_model
from src.deployment.routes import router

logger = logging.getLogger(__name__)

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model once at startup."""
    model, metadata = load_production_model()
    _state["model"] = model
    _state["metadata"] = metadata
    logger.info("API startup complete — model ready.")
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