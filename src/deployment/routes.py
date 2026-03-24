# =============================================================================
# src/deployment/routes.py — API route definitions
# =============================================================================
from __future__ import annotations

import base64
import io
import logging

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np

from src.deployment.schemas import ModelListItem, PredictionResponse

router = APIRouter()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTML UI
# ---------------------------------------------------------------------------

_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MLOps Prediction API</title>
<style>
  *, *::before, *::after { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, sans-serif;
    max-width: 860px; margin: 48px auto; padding: 0 24px;
    color: #1a1a1a; background: #fafafa;
  }
  h1 { font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 36px; }
  h2 { font-size: 1rem; font-weight: 600; color: #444;
       text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 14px; }
  .cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 36px; }
  .card {
    border: 2px solid #e0e0e0; border-radius: 10px;
    padding: 18px 22px; cursor: pointer; min-width: 180px;
    background: #fff; transition: border-color 0.15s, box-shadow 0.15s;
  }
  .card:hover { border-color: #999; box-shadow: 0 2px 8px rgba(0,0,0,.07); }
  .card.active { border-color: #111; background: #f4f4f4; }
  .card-title { font-weight: 700; font-size: 1rem; margin-bottom: 6px; }
  .card-meta { color: #777; font-size: 0.78rem; line-height: 1.6; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 99px;
    font-size: 0.72rem; font-weight: 600; margin-top: 8px;
  }
  .badge-image { background: #e8f4fd; color: #1a6fa8; }
  .badge-tabular { background: #e8fde8; color: #1a7a2e; }
  hr { border: none; border-top: 1px solid #eee; margin: 0 0 28px; }
  .form-section { display: none; }
  .form-section.visible { display: block; }
  .form-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 20px; }
  .field-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 14px; margin-bottom: 24px;
  }
  label { display: block; font-size: 0.82rem; font-weight: 600;
          color: #555; margin-bottom: 5px; }
  input[type=number], input[type=file] {
    width: 100%; padding: 9px 12px; border: 1px solid #d0d0d0;
    border-radius: 6px; font-size: 0.9rem; background: #fff;
  }
  input[type=number]:focus { outline: 2px solid #111; outline-offset: 1px; }
  .upload-area {
    border: 2px dashed #ccc; border-radius: 8px; padding: 28px;
    text-align: center; margin-bottom: 24px; background: #fff;
  }
  .upload-area input[type=file] { display: none; }
  .upload-label {
    cursor: pointer; color: #444; font-size: 0.9rem;
  }
  .upload-label:hover { color: #111; }
  .upload-preview {
    max-width: 200px; max-height: 200px; margin: 14px auto 0;
    border-radius: 6px; display: none;
  }
  .btn {
    padding: 11px 30px; background: #111; color: #fff;
    border: none; border-radius: 6px; font-size: 1rem;
    cursor: pointer; transition: background 0.15s;
  }
  .btn:hover { background: #333; }
  .btn:disabled { background: #aaa; cursor: default; }
  .result {
    margin-top: 28px; padding: 20px 24px;
    background: #fff; border: 1px solid #e0e0e0;
    border-radius: 10px; display: none;
  }
  .result.visible { display: block; }
  .result-label { font-size: 0.78rem; color: #888; text-transform: uppercase;
                  letter-spacing: 0.06em; margin-bottom: 8px; }
  .result-value { font-size: 2rem; font-weight: 700; }
  .result-meta { font-size: 0.8rem; color: #999; margin-top: 10px; }
  .error-text { color: #c00; }
  .loading-text { color: #888; font-style: italic; }
</style>
</head>
<body>

<h1>MLOps Prediction API</h1>
<p class="subtitle">Select a Production model, fill in the input, and run a prediction.</p>

<h2>Available models</h2>
<div id="cards" class="cards">
  <span class="loading-text">Loading models&hellip;</span>
</div>

<div id="form-section" class="form-section">
  <hr>
  <div id="form-title" class="form-title"></div>
  <form id="predict-form" novalidate>
    <div id="form-fields"></div>
    <button type="submit" class="btn">Run prediction</button>
  </form>
</div>

<div id="result" class="result">
  <div class="result-label">Prediction</div>
  <div id="result-value" class="result-value"></div>
  <div id="result-meta" class="result-meta"></div>
</div>

<script>
let selectedModelName = null;
let allModels = {};

async function loadModels() {
  const cardsEl = document.getElementById('cards');
  try {
    const resp = await fetch('/models');
    const models = await resp.json();
    allModels = {};
    models.forEach(m => { allModels[m.name] = m; });

    if (!models.length) {
      cardsEl.innerHTML = '<span class="error-text">No Production models found in the registry.</span>';
      return;
    }
    cardsEl.innerHTML = models.map(m => {
      const isImage = m.input_type === 'image';
      const badge = isImage
        ? '<span class="badge badge-image">Image</span>'
        : '<span class="badge badge-tabular">Tabular</span>';
      return `<div class="card" data-name="${escHtml(m.name)}" onclick="selectModel(this.dataset.name)">
        <div class="card-title">${escHtml(m.dataset || m.name)}</div>
        <div class="card-meta">
          Algorithm: ${escHtml(m.algorithm)}<br>
          Version: ${escHtml(m.version)}<br>
          Type: ${escHtml(m.task_type)}
        </div>
        ${badge}
      </div>`;
    }).join('');
  } catch (err) {
    cardsEl.innerHTML = `<span class="error-text">Could not load models: ${escHtml(err.message)}</span>`;
  }
}

function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function selectModel(name) {
  selectedModelName = name;
  document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));
  document.querySelector(`.card[data-name="${CSS.escape(name)}"]`).classList.add('active');

  const m = allModels[name];
  document.getElementById('form-title').textContent =
    'Predict with: ' + (m.dataset || m.name) + ' (' + m.algorithm + ' v' + m.version + ')';
  buildForm(m);

  document.getElementById('form-section').classList.add('visible');
  document.getElementById('result').classList.remove('visible');
}

function buildForm(m) {
  const el = document.getElementById('form-fields');
  if (m.input_type === 'image') {
    el.innerHTML = `
      <div class="upload-area" id="upload-area">
        <input type="file" id="image-file" accept="image/*">
        <label class="upload-label" for="image-file" onclick="document.getElementById('image-file').click(); return false;">
          Click to select an image (PNG, JPEG)
        </label>
        <img id="upload-preview" class="upload-preview" alt="Preview">
      </div>`;
    document.getElementById('image-file').addEventListener('change', function() {
      const file = this.files[0];
      if (!file) return;
      document.querySelector('.upload-label').textContent = file.name;
      const preview = document.getElementById('upload-preview');
      preview.src = URL.createObjectURL(file);
      preview.style.display = 'block';
    });
  } else {
    el.innerHTML = `<div class="field-grid">${
      m.feature_names.map(f => `
        <div>
          <label>${escHtml(f)}</label>
          <input type="number" name="${escHtml(f)}" step="any" placeholder="0.0" required>
        </div>`).join('')
    }</div>`;
  }
}

document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedModelName) return;
  const m = allModels[selectedModelName];
  let body;

  if (m.input_type === 'image') {
    const file = document.getElementById('image-file').files[0];
    if (!file) { alert('Please select an image first.'); return; }
    const dataUrl = await readAsDataURL(file);
    body = { image: dataUrl.split(',')[1] };
  } else {
    body = {};
    let valid = true;
    m.feature_names.forEach(f => {
      const input = e.target.querySelector(`input[name="${CSS.escape(f)}"]`);
      const val = parseFloat(input.value);
      if (isNaN(val)) { input.focus(); valid = false; }
      else body[f] = val;
    });
    if (!valid) return;
  }

  const btn = e.target.querySelector('button[type=submit]');
  btn.textContent = 'Running\u2026';
  btn.disabled = true;

  try {
    const resp = await fetch('/predict/' + encodeURIComponent(selectedModelName), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await resp.json();
    const resultEl = document.getElementById('result');
    resultEl.classList.add('visible');
    if (!resp.ok) {
      document.getElementById('result-value').innerHTML =
        '<span class="error-text">' + escHtml(data.detail || 'Unknown error') + '</span>';
      document.getElementById('result-meta').textContent = '';
    } else {
      document.getElementById('result-value').textContent = data.prediction;
      document.getElementById('result-meta').textContent =
        'Algorithm: ' + data.algorithm +
        '  ·  Version: ' + data.model_version_id +
        '  ·  Type: ' + data.task_type;
    }
  } catch (err) {
    const resultEl = document.getElementById('result');
    resultEl.classList.add('visible');
    document.getElementById('result-value').innerHTML =
      '<span class="error-text">Network error: ' + escHtml(err.message) + '</span>';
    document.getElementById('result-meta').textContent = '';
  } finally {
    btn.textContent = 'Run prediction';
    btn.disabled = false;
  }
});

function readAsDataURL(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

loadModels();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui() -> HTMLResponse:
    """Serve the model selection and prediction UI."""
    return HTMLResponse(content=_UI_HTML)


@router.get("/health")
def health(request: Request) -> JSONResponse:
    """Health / readiness check.

    Returns 200 with model count when at least one model is loaded.
    Returns 503 when zero models are loaded (the API cannot serve predictions).
    """
    models: dict = request.app.state.model_state.get("models", {})
    model_count = len(models)
    if model_count == 0:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "models_loaded": 0},
        )
    return JSONResponse(
        content={"status": "ok", "models_loaded": model_count},
    )


@router.get("/models", response_model=list[ModelListItem])
def list_models(request: Request) -> list[ModelListItem]:
    """List all Production models currently loaded by the API."""
    models: dict = request.app.state.model_state.get("models", {})
    return [
        ModelListItem(
            name=info.model_name,
            dataset=info.dataset_name,
            algorithm=info.algorithm,
            task_type=info.task_type,
            version=info.model_version,
            trained_at=info.trained_at,
            input_type="image" if info.image_shape is not None else "tabular",
            feature_names=info.feature_names,
        )
        for info in models.values()
    ]


@router.post("/predict/{model_name}", response_model=PredictionResponse)
def predict(request: Request, model_name: str, body: dict) -> PredictionResponse:
    """Run inference on the named Production model.

    For tabular models: body is ``{feature_name: value, ...}``.
    For image models:   body is ``{"image": "<base64-encoded image string>"}``.
    """
    models: dict = request.app.state.model_state.get("models", {})

    if model_name not in models:
        available = list(models.keys())
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"Model '{model_name}' not found. Available models: {available}"
            },
        )

    model_info = models[model_name]

    if model_info.image_shape is not None:
        return _predict_image(model_info, body)
    return _predict_tabular(model_info, body)


# ---------------------------------------------------------------------------
# Tabular prediction
# ---------------------------------------------------------------------------

def _predict_tabular(model_info, body: dict):
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
        task_type=model_info.task_type,
    )


# ---------------------------------------------------------------------------
# Image prediction (image_classification sklearn and image_classification_cnn)
# ---------------------------------------------------------------------------

def _predict_image(model_info, body: dict):
    if "image" not in body:
        return JSONResponse(
            status_code=422,
            content={"detail": "Missing required field: 'image' (base64-encoded image string)."}
        )

    try:
        image_bytes = base64.b64decode(body["image"])
    except Exception:
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid base64 encoding in 'image' field."}
        )

    try:
        from PIL import Image as PILImage
        img = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
        h, w = model_info.image_shape[0], model_info.image_shape[1]
        img = img.resize((w, h))  # PIL uses (width, height)
        arr = np.array(img, dtype=np.float64) / 255.0  # (H, W, C)

        stats = model_info.normalization_stats or {}
        mean = stats.get("mean")
        std = stats.get("std")
        if mean is not None and std is not None:
            std_arr = np.array(std)
            std_arr = np.where(std_arr == 0, 1.0, std_arr)
            arr = (arr - np.array(mean)) / std_arr
    except Exception as e:
        logger.error("Image preprocessing failed: %s", e)
        return JSONResponse(
            status_code=422,
            content={"detail": f"Image preprocessing failed: {e}"}
        )

    try:
        if model_info.model_format == "pytorch":
            import torch
            # (H, W, C) → (1, C, H, W)
            tensor = torch.tensor(
                arr.transpose(2, 0, 1)[np.newaxis, ...], dtype=torch.float32
            )
            class_index = int(model_info.model.predict(tensor)[0])
        else:
            # sklearn image_classification — flatten to (1, H*W*C)
            flat = arr.reshape(1, -1).astype(np.float32)
            class_index = int(model_info.model.predict(flat)[0])
    except Exception as e:
        logger.error("Image inference failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"detail": "Inference failed due to an internal error."}
        )

    index_to_class = model_info.index_to_class or {}
    class_name = index_to_class.get(str(class_index), str(class_index))

    return PredictionResponse(
        prediction=class_name,
        model_version_id=model_info.dataset_version_id,
        algorithm=model_info.algorithm,
        task_type=model_info.task_type,
    )
