# =============================================================================
# src/deployment/routes.py — API route definitions
# =============================================================================
# Responsibility: Define all API endpoints.
# =============================================================================
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/health")
def health() -> JSONResponse:
    """Health check endpoint — confirms the API is running."""
    return JSONResponse(content={"status": "ok"})