# =============================================================================
# src/deployment/routes.py — API route definitions
# =============================================================================
from __future__ import annotations

import logging

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
import numpy as np

from src.deployment.schemas import PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def health() -> JSONResponse:
    """Health check endpoint — confirms the API is running."""
    return JSONResponse(content={"status": "ok"})


@router.post("/predict", response_model=PredictionResponse)
def predict(request: Request, body: dict) -> PredictionResponse:
    """Run inference using the active production model.

    Accepts a JSON object with feature names matching the model's feature map.
    Returns prediction result with model metadata.
    """
    state = request.app.state.model_state
    model_info = state["model_info"]

    expected_features = model_info.feature_names

    if not expected_features:
        return JSONResponse(
            status_code=500,
            content={"detail": "Feature map not available — cannot validate input."}
        )

    missing = [f for f in expected_features if f not in body]
    if missing:
        return JSONResponse(
            status_code=422,
            content={"detail": f"Missing required features: {missing}"}
        )

    try:
        features = np.array([[float(body[f]) for f in expected_features]])
    except (ValueError, TypeError) as e:
        return JSONResponse(
            status_code=422,
            content={"detail": f"Invalid feature value: {e}"}
        )

    try:
        prediction = model_info.model.predict(features)[0]
        if hasattr(prediction, "item"):
            prediction = prediction.item()
    except Exception as e:
        logger.error("Inference failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"detail": "Inference failed due to an internal error."}
        )

    return PredictionResponse(
        prediction=str(prediction),
        model_version_id=model_info.dataset_version_id,
        algorithm=model_info.algorithm,
        task_type="classification",
    )