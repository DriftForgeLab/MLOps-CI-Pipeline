# MLOps-CI-Pipeline

An end-to-end MLOps pipeline demonstrating automated retraining, evaluation-based promotion, versioned deployment, and data drift monitoring, designed for a bachelor-level engineering project.

## Requirements

Python 3.12.x is required. Other versions are not officially supported.

## Setup

### 1. Create virtual environment

**macOS / Linux**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows**
```bash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate
```

### 2. Install project in editable mode
```bash
pip install --upgrade pip
pip install -e .
```

## CLI commands

All commands become available after `pip install -e .`.

### Run pipeline

Runs the full pipeline: versioning → validation → split → preprocessing → training → evaluation → promotion. All data preparation is handled automatically.

```bash
# Tabular classification (CSV datasets)
run-pipeline --config src/config/pipeline_tabular.yaml

# Image classification — CNN with PyTorch (JPG/PNG)
run-pipeline --config src/config/pipeline_image.yaml

# Raw DNG images through ISP pipeline → CNN
run-pipeline --config src/config/pipeline_raw_image.yaml
```

> **First run with a new dataset:** if `dataset.yaml` is missing, the pipeline will prompt
> you interactively for target column and task type. This only happens once.

### Roll back a model

Rolls back the Production model to a previous registry version. Omit `--version` to roll
back to the most recent Staging version. `--yes` skips the confirmation prompt.

```bash
rollback-model --config src/config/pipeline_tabular.yaml
rollback-model --config src/config/pipeline_tabular.yaml --version 2 --reason "accuracy regression" --yes
```

### Start the prediction API

Starts the FastAPI prediction service. Host and port are read from `src/config/deployment.yaml`.
Requires at least one model promoted to Production via `run-pipeline` first.

```bash
run-api
```

The API exposes:
- `GET /health` — liveness check
- `POST /predict` — single-sample prediction (JSON body)

### Monitor tabular data drift

Detects statistical drift between new production data (CSV) and the training reference.
Run this periodically after collecting new production rows.

```bash
# Minimal — auto-resolves dataset name and latest version from config:
monitor-drift \
    --batch-csv data/new_batch.csv \
    --model-name iris_rf \
    --config src/config/pipeline_tabular.yaml

# Explicit dataset version:
monitor-drift \
    --batch-csv data/new_batch.csv \
    --model-name iris_rf \
    --config src/config/pipeline_tabular.yaml \
    --dataset-version abc123hash
```

Results are saved as JSON in `outputs/drift_monitoring/<model-name>/`. If drift severity
meets the configured threshold, you will be prompted to choose a response action.

### Monitor image data drift

Measures pixel-distribution drift between training images and new production images.
Run this periodically after collecting new images from production.

**Step 1 — place new images in a folder** (flat or class subfolders, both work):
```
data/incoming/my_new_batch/
    image_001.jpg
    image_002.jpg
    ...
```

For raw DNG images:
```
data/incoming/my_new_batch/
    frame_101.dng
    frame_102.dng
    ...
```

**Step 2 — preprocess the new images into a batch file:**
```bash
# Standard JPG/PNG images
prepare-image-batch \
    --input-dir data/incoming/my_new_batch \
    --config src/config/pipeline_image.yaml

# Raw DNG images
prepare-image-batch \
    --input-dir data/incoming/my_new_batch \
    --config src/config/pipeline_raw_image.yaml
```

This automatically finds the correct training version and applies the same
preprocessing (resize, normalization) as during training. When done, it prints
the exact `monitor-drift-image` command to run next — just copy and paste it.

**Step 3 — run drift monitoring** (copy the command printed by step 2):
```bash
monitor-drift-image \
    --batch-npz data/batches/<timestamp>.npz \
    --config src/config/pipeline_image.yaml
```

For raw-image pipelines, add ISP scenario interpretation to get a plausible
physical explanation of the observed drift (e.g. "pattern resembles warm white balance shift").
This is interpretation of detected drift — the Wasserstein score is the actual measurement:
```bash
monitor-drift-image \
    --batch-npz data/batches/<timestamp>.npz \
    --config src/config/pipeline_raw_image.yaml \
    --drift-scenarios-dir data/drift_scenarios/ \
    --sensitivity-report data/drift_scenarios/sensitivity_report.json
```

Results are printed to the terminal and saved as JSON in
`outputs/drift_monitoring/images/`. If drift is severe enough, you will be
prompted to choose a response (retrain, collect more data, accept, escalate).

### Run tests

```bash
python -m pytest tests/ -v --tb=short
```

## Docker

The prediction service can be deployed as a Docker container. It loads the current
Production model from the MLflow registry at startup.

```bash
# Prerequisites: a Production model must exist (run pipeline and approve first)
run-pipeline --config src/config/pipeline_tabular.yaml

# Copy environment template
cp .env.example .env

# Build and start
docker compose -f docker/docker-compose.yml up --build

# Verify
curl http://localhost:8000/health

# Stop
docker compose -f docker/docker-compose.yml down

# Rebuild after config/code changes without losing volumes
docker compose -f docker/docker-compose.yml up --build --force-recreate
```

## MLflow tracking UI

Browse experiments, runs, metrics, and registered models:

```bash
mlflow ui --backend-store-uri mlruns
```

Open `http://localhost:5000` in a browser.

## Pipeline stages

| Stage            | Pipelines          | Description                                                                                       |
|------------------|--------------------|---------------------------------------------------------------------------------------------------|
| `preprocessing`  | all                | Selects features, normalises, writes `preprocessed/` (CSV or NPZ)                                |
| `training`       | all                | Trains the model defined in `training_*.yaml`; saves artifact to registry                         |
| `evaluation`     | all                | Computes metrics (accuracy, F1, etc.) against the held-out test split                             |
| `model_analysis` | image only         | **Raw images:** ISP sensitivity analysis — pre-computes scenario signatures for `monitor-drift-image`. **Standard images:** augmentation robustness — measures how much each transformation degrades accuracy before deployment. |
| `promotion`      | all                | Promotes model to Production if promotion rules pass; requests approval if configured             |

The `model_analysis` stage is **offline pre-deployment analysis**, not drift detection.
Drift detection — comparing the training reference against new production data over time — runs
separately via the monitoring CLIs (`monitor-drift`, `monitor-drift-image`) after real batches arrive.

## Data flow

**Tabular datasets:**
```
data/raw/<dataset>/data.csv
        ↓  versioning
data/processed/<dataset>/<version_id>/data.csv  +  train/  val/  test/
        ↓  preprocessing
data/processed/<dataset>/<version_id>/preprocessed/  train.csv  val.csv  test.csv
```

**Image datasets (JPG/PNG):**
```
data/raw/<dataset>/images/<class>/...
        ↓  versioning + stratified split
data/processed/<dataset>/<version_id>/{train,val,test}/images/<class>/...
        ↓  preprocessing (resize, normalise)
data/processed/<dataset>/<version_id>/preprocessed/  train.npz  val.npz  test.npz
```

**Raw DNG datasets (ISP pipeline):**
```
data/raw/<dataset>/images/<class>/*.DNG
        ↓  versioning + stratified split
data/processed/<dataset>/<version_id>/{train,val,test}/images/<class>/*.DNG
        ↓  ISP pipeline (black level → demosaicing → white balance →
           colour correction → denoising → sharpening → gamma)
        ↓  resize, normalise
data/processed/<dataset>/<version_id>/preprocessed/  train.npz  val.npz  test.npz
```

## Adding datasets

See `data/raw/README.md` for instructions on how to add new datasets.
