# =============================================================================
# src/deployment/schemas.py — Dynamic request and response schemas
# =============================================================================
from __future__ import annotations

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    """Output schema for the /predict endpoint."""
    prediction: str | float
    model_version_id: str
    algorithm: str
    task_type: str