# =============================================================================
# src/deployment/app.py — FastAPI application entry point
# =============================================================================
# Responsibility: Define the FastAPI app instance, register routers, and
# expose a CLI entry point for launching the API with uvicorn.
# =============================================================================
from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from src.deployment.routes import router

app = FastAPI(
    title="MLOps Prediction API",
    description="Serves predictions from the active production model.",
    version="0.1.0",
)

app.include_router(router)


def main() -> None:
    """CLI entry point — launch the API with uvicorn."""
    uvicorn.run("src.deployment.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()