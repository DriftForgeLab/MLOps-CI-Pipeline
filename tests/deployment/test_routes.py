import base64
import io

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from unittest.mock import MagicMock

from src.deployment.routes import router
from src.deployment.startup_checks import ProductionModelInfo

pytestmark = pytest.mark.anyio


# ── Helpers ─────────────────────────────────────────────────────────────────

def _make_model_info(
    *,
    model_name="iris-classifier",
    algorithm="random_forest",
    task_type="classification",
    feature_names=("sepal_len", "sepal_wid", "petal_len", "petal_wid"),
    image_shape=None,
    normalization_stats=None,
    index_to_class=None,
    predict_return=np.array([1.0]),
    predict_side_effect=None,
):
    model = MagicMock()
    if predict_side_effect:
        model.predict.side_effect = predict_side_effect
    else:
        model.predict.return_value = predict_return
    return ProductionModelInfo(
        model=model,
        model_name=model_name,
        model_version="1",
        run_id="run-123",
        stage="Production",
        algorithm=algorithm,
        task_type=task_type,
        trained_at="2026-03-10",
        dataset_version_id="abc123",
        dataset_name="iris",
        promotion_outcome="approved",
        feature_names=list(feature_names),
        image_shape=image_shape,
        normalization_stats=normalization_stats,
        index_to_class=index_to_class,
    )


def _make_app(models: dict | None = None) -> FastAPI:
    """Create a fresh FastAPI app with the router and pre-loaded model state."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.state.model_state = {"models": models or {}}
    return test_app


def _make_image_base64(width=8, height=8, color=(128, 64, 32)):
    """Create a small valid PNG image as a base64 string."""
    from PIL import Image as PILImage

    img = PILImage.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ── Health & UI ─────────────────────────────────────────────────────────────

class TestHealthAndUI:
    async def test_health_returns_ok(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app()), base_url="http://test"
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}

    async def test_ui_returns_html(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app()), base_url="http://test"
        ) as client:
            resp = await client.get("/")
            assert resp.status_code == 200
            assert "text/html" in resp.headers["content-type"]
            assert "<title>" in resp.text


# ── GET /models ─────────────────────────────────────────────────────────────

class TestListModels:
    async def test_returns_loaded_models(self):
        info = _make_model_info()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"iris-classifier": info})),
            base_url="http://test",
        ) as client:
            resp = await client.get("/models")
            assert resp.status_code == 200
            items = resp.json()
            assert len(items) == 1
            assert items[0]["name"] == "iris-classifier"
            assert items[0]["algorithm"] == "random_forest"

    async def test_empty_when_no_models(self):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app()), base_url="http://test"
        ) as client:
            resp = await client.get("/models")
            assert resp.status_code == 200
            assert resp.json() == []

    async def test_tabular_type_when_no_image_shape(self):
        info = _make_model_info(image_shape=None)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            items = (await client.get("/models")).json()
            assert items[0]["input_type"] == "tabular"

    async def test_image_type_when_image_shape_set(self):
        info = _make_model_info(image_shape=[32, 32, 3])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            items = (await client.get("/models")).json()
            assert items[0]["input_type"] == "image"


# ── POST /predict — tabular ────────────────────────────────────────────────

class TestPredictTabular:
    async def test_success(self):
        info = _make_model_info()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"iris-classifier": info})),
            base_url="http://test",
        ) as client:
            body = {"sepal_len": 5.1, "sepal_wid": 3.5, "petal_len": 1.4, "petal_wid": 0.2}
            resp = await client.post("/predict/iris-classifier", json=body)
            assert resp.status_code == 200
            data = resp.json()
            assert data["prediction"] == "1.0"
            assert data["algorithm"] == "random_forest"
            assert data["task_type"] == "classification"

    async def test_unknown_model_404(self):
        info = _make_model_info()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"iris-classifier": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/nonexistent", json={"a": 1})
            assert resp.status_code == 404
            detail = resp.json()["detail"]
            assert "nonexistent" in detail
            assert "iris-classifier" in detail

    async def test_missing_features_422(self):
        info = _make_model_info()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"iris-classifier": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/iris-classifier", json={"sepal_len": 5.1})
            assert resp.status_code == 422
            assert "Missing required features" in resp.json()["detail"]

    async def test_non_numeric_value_422(self):
        info = _make_model_info()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"iris-classifier": info})),
            base_url="http://test",
        ) as client:
            body = {"sepal_len": "abc", "sepal_wid": 3.5, "petal_len": 1.4, "petal_wid": 0.2}
            resp = await client.post("/predict/iris-classifier", json=body)
            assert resp.status_code == 422
            assert "Invalid feature value" in resp.json()["detail"]

    async def test_empty_feature_names_500(self):
        info = _make_model_info(feature_names=[])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/m", json={})
            assert resp.status_code == 500
            assert "Feature map not available" in resp.json()["detail"]

    async def test_inference_error_500(self):
        info = _make_model_info(predict_side_effect=RuntimeError("boom"))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            body = {"sepal_len": 1, "sepal_wid": 2, "petal_len": 3, "petal_wid": 4}
            resp = await client.post("/predict/m", json=body)
            assert resp.status_code == 500
            assert "internal error" in resp.json()["detail"].lower()


# ── POST /predict — image ──────────────────────────────────────────────────

class TestPredictImage:
    async def test_success_sklearn(self):
        info = _make_model_info(
            model_name="cifar",
            algorithm="random_forest",
            task_type="image_classification",
            image_shape=[8, 8, 3],
            feature_names=[],
            index_to_class={"0": "cat", "1": "dog"},
            predict_return=np.array([1]),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"cifar": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/cifar", json={"image": _make_image_base64()})
            assert resp.status_code == 200
            assert resp.json()["prediction"] == "dog"

    async def test_missing_image_field_422(self):
        info = _make_model_info(image_shape=[8, 8, 3], feature_names=[])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/m", json={})
            assert resp.status_code == 422
            assert "image" in resp.json()["detail"].lower()

    async def test_invalid_base64_422(self):
        info = _make_model_info(image_shape=[8, 8, 3], feature_names=[])
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/m", json={"image": "!!!not-base64!!!"})
            assert resp.status_code == 422
            assert "base64" in resp.json()["detail"].lower()

    async def test_corrupt_image_bytes_422(self):
        info = _make_model_info(image_shape=[8, 8, 3], feature_names=[])
        bad_bytes = base64.b64encode(b"not-a-real-image").decode()
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/m", json={"image": bad_bytes})
            assert resp.status_code == 422
            assert "preprocessing failed" in resp.json()["detail"].lower()

    async def test_inference_error_500(self):
        info = _make_model_info(
            image_shape=[8, 8, 3],
            feature_names=[],
            predict_side_effect=RuntimeError("boom"),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/m", json={"image": _make_image_base64()})
            assert resp.status_code == 500
            assert "internal error" in resp.json()["detail"].lower()

    async def test_class_index_fallback_when_missing(self):
        info = _make_model_info(
            image_shape=[8, 8, 3],
            feature_names=[],
            index_to_class=None,
            predict_return=np.array([42]),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/m", json={"image": _make_image_base64()})
            assert resp.status_code == 200
            assert resp.json()["prediction"] == "42"

    async def test_cnn_algorithm_path(self):
        info = _make_model_info(
            algorithm="cnn",
            task_type="image_classification_cnn",
            image_shape=[8, 8, 3],
            feature_names=[],
            index_to_class={"0": "cat"},
            predict_return=np.array([0]),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/m", json={"image": _make_image_base64()})
            assert resp.status_code == 200
            assert resp.json()["prediction"] == "cat"

    async def test_normalization_stats_applied(self):
        info = _make_model_info(
            image_shape=[8, 8, 3],
            feature_names=[],
            normalization_stats={
                "mean": [0.5, 0.5, 0.5],
                "std": [0.25, 0.25, 0.25],
            },
            predict_return=np.array([0]),
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=_make_app({"m": info})),
            base_url="http://test",
        ) as client:
            resp = await client.post("/predict/m", json={"image": _make_image_base64()})
            assert resp.status_code == 200
            # Verify the model was called — normalization changes the input
            info.model.predict.assert_called_once()
            arr = info.model.predict.call_args[0][0]
            # With mean=0.5, std=0.25 normalization, values should be shifted
            # Original pixel values are ~0-1 (after /255), so normalized values
            # should be roughly in [-2, 2] range
            assert arr.min() < 0 or arr.max() > 1
