# =============================================================================
# src/deployment/schemas.py — Dynamic request and response schemas
# =============================================================================
from __future__ import annotations

from pydantic import BaseModel


class ModelListItem(BaseModel):
    """Metadata for a single Production model, returned by GET /models."""
    name: str                       # MLflow registry model name
    dataset: str                    # Dataset the model was trained on
    algorithm: str
    task_type: str
    version: str
    trained_at: str
    input_type: str                 # "tabular" or "image"
    feature_names: list[str]        # Empty for image models


class ImagePredictionRequest(BaseModel):
    """Input schema for /predict when the loaded model is an image classifier.

    The ``image`` field must be a base64-encoded image string (PNG, JPEG, etc.).
    Preprocessing (resize, normalize) is applied server-side using parameters
    stored in the model's feature_map.
    """
    image: str


class PredictionResponse(BaseModel):
    """Output schema for the /predict endpoint."""
    prediction: str | float
    model_version_id: str
    algorithm: str
    task_type: str