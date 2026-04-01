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
run-pipeline --config src/config/pipeline_raw_cnn.yaml
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

| Stage          | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `preprocessing`| Selects features, normalises, writes `preprocessed/` (CSV or NPZ)          |
| `training`     | Trains the model defined in `training_*.yaml`; saves artifact to registry   |
| `evaluation`   | Computes metrics (accuracy, F1, etc.) and drift tests against reference data |
| `promotion`    | Promotes model to Production if promotion rules pass; requests approval if configured |

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
        ↓  preprocessing (resize, normalise, flatten)
data/processed/<dataset>/<version_id>/preprocessed/  train.npz  val.npz  test.npz
```

**Raw DNG datasets (ISP pipeline):**
```
data/raw/<dataset>/images/<class>/*.DNG
        ↓  versioning + stratified split
data/processed/<dataset>/<version_id>/{train,val,test}/images/<class>/*.DNG
        ↓  ISP pipeline (black level → demosaicing → white balance →
           colour correction → denoising → sharpening → gamma)
        ↓  resize, normalise, flatten
data/processed/<dataset>/<version_id>/preprocessed/  train.npz  val.npz  test.npz
```

## Adding datasets

See `data/raw/README.md` for instructions on how to add new datasets.
