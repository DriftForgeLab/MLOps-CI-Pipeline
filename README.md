# MLOps-CI-Pipeline
An end-to-end MLOps pipeline demonstrating automated retraining, evaluation-based promotion, versioned deployment, and data drift monitoring, designed for a bachelor-level engineering project.

## Requirements

This project requires Python 3.12.x.
Other versions are not officially supported.

## Setup

### 1. Create virtual environment (Python 3.12 required)

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

### 2. Upgrade pip
```bash
python -m pip install --upgrade pip
```

### 3. Install project in editable mode
```bash
pip install -e .
```

## How to Run

**Run tests**
```bash
python -m pytest tests/ -v --tb=short
```

**Run pipeline**
```bash
run-pipeline --config src/config/pipeline.yaml
```

> **First run**: If a dataset is missing `dataset.yaml`, the pipeline will prompt
> you interactively to provide target column and task type. This only happens once —
> subsequent runs skip the prompt automatically.

## Pipeline Stages

The pipeline executes the following stages in order:

| Stage          | Status      | Description                            |
|---|---|---|
| `preprocessing`| Implemented | Selects feature and target columns from each split, writes to  `preprocessed/`  |
| `training`     | Placeholder | Model training — not yet implemented   |
| `evaluation`   | Placeholder | Model evaluation — not yet implemented |
| `deployment`   | Placeholder | Model deployment — not yet implemented |

## Data Flow

**Tabular datasets:**
```
data/raw/<dataset>/data.csv
        ↓  ingestion + versioning
data/processed/<dataset>/<version_id>/data.csv  +  train/  val/  test/
        ↓  preprocessing
data/processed/<dataset>/<version_id>/preprocessed/  train.csv  val.csv  test.csv
```

**Image datasets:**
```
data/raw/<dataset>/images/{class}/...
        ↓  versioning + stratified splitting
data/processed/<dataset>/<version_id>/train/images/{class}/...
        ↓  preprocessing (resize, normalize, flatten)
data/processed/<dataset>/<version_id>/preprocessed/  train.npz  val.npz  test.npz
```

Preprocessing reads column definitions (`target`, `features`) from the versioned `dataset.yaml` — no separate config file is needed.

## Deployment

The prediction service runs as a Docker container that loads the current
Production model from the MLflow registry at startup.

```bash
# 1. Make sure a Production model exists (run pipeline and approve)
run-pipeline --config src/config/pipeline.yaml

# 2. Copy environment template
cp .env.example .env

# 3. Build and start the container
docker compose -f docker/docker-compose.yml up --build

# 4. Verify
curl http://localhost:8000/health
```

See [docs/deployment.md](docs/deployment.md) for full configuration,
model governance details, and troubleshooting.

## Image Classification

## Image Classification

The pipeline supports image classification datasets using the ImageFolder
convention. Place class-labeled images under `data/raw/<name>/images/{class}/`
with a `dataset.yaml` specifying `task_type: image_classification`.
```bash
# sklearn-based (flattened pixel vectors)
run-pipeline --config src/config/pipeline_image_classification.yaml

# CNN-based with PyTorch (spatial feature learning, recommended for drift analysis)
run-pipeline --config src/config/pipeline_image_cnn.yaml
```

See [docs/image_datasets.md](docs/image_datasets.md) for folder structure,
preprocessing configuration, augmentation, and limitations.

## Adding Datasets
See `data/raw/README.md` for instructions on how to add new datasets.