# Deployment Guide

How to build, run, test, and deploy the containerized prediction service.

## Architecture

```
Host machine                          Docker container
+----------------------------------+  +------------------------------+
|                                  |  |  /app                        |
|  mlruns/  (MLflow tracking)  ----|->|  /app/mlruns  (read-only)    |
|  src/config/  (YAML configs) ----|->|  /app/src/config (read-only) |
|                                  |  |                              |
|  .env  (environment vars)    ----|->|  Environment variables       |
|                                  |  |                              |
|  localhost:8000  <----------------|--|  uvicorn :8000               |
+----------------------------------+  +------------------------------+
```

The model is **not** baked into the Docker image. It is loaded at runtime
from the volume-mounted `mlruns/` directory via the MLflow Model Registry.
This means the image stays the same across model versions — only a
container restart is needed after a new model is promoted.

## Prerequisites

- **Docker** (with Compose v2) installed and running.
- **Python 3.12** for running the pipeline locally.
- At least one model promoted to **Production** via the pipeline
  (`run-pipeline --config src/config/pipeline_tabular_classification.yaml` and approve when prompted).

## Quick Start

1. **Run the pipeline** and approve a model so a Production version exists:
   ```bash
   run-pipeline --config src/config/pipeline_tabular_classification.yaml
   ```

2. **Copy the environment file** and adjust if needed:
   ```bash
   cp .env.example .env
   ```

3. **Build and start** the container:
   ```bash
   docker compose -f docker/docker-compose.yml up --build
   ```

4. **Verify** the service is running:
   ```bash
   curl http://localhost:8000/health
   ```
   Expected: HTTP 200 with JSON body.

5. **Stop** the service:
   ```bash
   docker compose -f docker/docker-compose.yml down
   ```

## Configuration Reference

### Environment variables (`.env`)

| Variable              | Default                      | Description                                        |
|-----------------------|------------------------------|----------------------------------------------------|
| `MLFLOW_TRACKING_URI` | `/app/mlruns`                | Path to the MLflow tracking store inside container. |
| `API_PORT`            | `8000`                       | Host port mapped to the container's port 8000.     |
| `PIPELINE_CONFIG_PATH`| `/app/src/config/pipeline_tabular_classification.yaml` | Pipeline config path inside the container.       |
| `LOG_LEVEL`           | `INFO`                       | Logging level for the prediction service.          |
| `MODEL_STAGE`         | `Production`                 | MLflow stage to load. Only `Production` in practice.|
| `API_ADMIN_TOKEN`     | _(unset)_                    | Shared secret for `POST /admin/reload`. When set, the matching `X-Admin-Token` header is required; when unset, the endpoint is open. |

### Deployment config (`src/config/deployment.yaml`)

| Section       | Key                        | Default        | Description                                       |
|---------------|----------------------------|----------------|---------------------------------------------------|
| `server`      | `host`                     | `0.0.0.0`      | Bind address for uvicorn.                         |
| `server`      | `port`                     | `8000`         | Port inside the container.                        |
| `server`      | `log_level`                | `info`         | Uvicorn log level.                                |
| `model`       | `allowed_stage`            | `Production`   | Only this MLflow stage will be loaded.            |
| `model`       | `require_production_model` | `true`         | Fail startup if no Production model exists.       |
| `model`       | `startup_timeout_seconds`  | `120`          | Max seconds for model loading before timeout.     |
| `healthcheck` | `include_model_info`       | `true`         | Include model metadata in health endpoint response.|
| `reload`      | `enabled`                  | `false`        | Opt-in: POST the reload endpoint after a successful pipeline run. |
| `reload`      | `url`                      | `http://localhost:8000/admin/reload` | Reload endpoint of the running API. |
| `reload`      | `timeout_seconds`          | `5`            | Max seconds to wait for the reload HTTP call.     |

### Docker Compose overrides

The compose file at `docker/docker-compose.yml` exposes:
- **Port:** `${API_PORT:-8000}:8000` (configurable via `.env`).
- **Volumes:** `mlruns/` and `src/config/` mounted read-only.
- **Restart policy:** `unless-stopped` (auto-restarts on failure).

## Model Governance

### How Production enforcement works

The `src/deployment/startup_checks.py` module enforces that only the
Production-stage model is served:

1. At startup, it queries the MLflow Model Registry for versions with
   `stage="Production"`.
2. If no Production version exists, the service **refuses to start** with
   a clear error message.
3. The loaded model's lineage metadata (version, run_id, algorithm,
   dataset, promotion outcome) is preserved in a `ProductionModelInfo`
   object for use in API responses.

### What happens after a rollback

If you run `rollback-model` to promote a different version to Production,
the **running container still serves the previous model** until restarted.
This is by design — it prevents accidental mid-flight model switches.

To serve the rolled-back model, either restart the container:
```bash
docker compose -f docker/docker-compose.yml restart
```
or trigger a zero-downtime reload (see [Hot Model Reload](#hot-model-reload)).

### What is NOT served

- Models in `None`, `Staging`, or `Archived` stages are never loaded.
- Candidate models that failed promotion rules or were rejected are not
  accessible through the API.

## Hot Model Reload

The running API loads Production models **once at startup**. To pick up a
freshly promoted model without a process restart, the service exposes an
admin reload endpoint.

### `POST /admin/reload`

Re-queries the MLflow registry, rebuilds the loaded-model set, and **atomically
swaps it in** — in-flight `/predict` requests never observe a half-loaded
state. If the reload fails (registry unreachable, no models), the previously
loaded models stay live and the endpoint returns an error instead of crashing.

```bash
curl -X POST http://localhost:8000/admin/reload
```

Response (`200 OK`):
```json
{
  "status": "reloaded",
  "models_loaded": 1,
  "models": [{"name": "iris-classifier", "version": "2"}],
  "reloaded_at": "2026-05-15T12:00:00+00:00"
}
```

| Status | Meaning                                                          |
|--------|------------------------------------------------------------------|
| `200`  | Reload succeeded; body summarizes the newly loaded models.       |
| `403`  | `API_ADMIN_TOKEN` is set and the `X-Admin-Token` header is missing or wrong. |
| `503`  | Reload failed (e.g. registry unreachable); old models stay live. |

**Authentication.** When the `API_ADMIN_TOKEN` environment variable is set,
callers must send a matching `X-Admin-Token` header:
```bash
curl -X POST http://localhost:8000/admin/reload -H "X-Admin-Token: $API_ADMIN_TOKEN"
```
When `API_ADMIN_TOKEN` is unset the endpoint is open — set it (and bind the
server to a trusted network) for any non-local deployment.

### Opt-in pipeline trigger

The pipeline can ping the reload endpoint automatically after a successful
run. It is **disabled by default** to keep `run-pipeline` side-effect-free and
CI-safe. Enable it in `src/config/deployment.yaml`:

```yaml
reload:
  enabled: true
  url: "http://localhost:8000/admin/reload"
  timeout_seconds: 5
```

When enabled, after a successful run whose `deployment` stage completed, the
pipeline POSTs to `reload.url` (attaching `X-Admin-Token` from `API_ADMIN_TOKEN`
if set). The call is fully CI-safe: if no API is reachable, the failure is
logged and the pipeline run still succeeds.

## Building the Docker Image

### Manual build

```bash
docker build -f docker/Dockerfile -t mlops-prediction-api .
```

### Running without Compose

```bash
docker run --rm \
  -p 8000:8000 \
  -v "$(pwd)/mlruns:/app/mlruns:ro" \
  -v "$(pwd)/src/config:/app/src/config:ro" \
  -v "$(pwd)/artifacts:/app/artifacts:ro" \
  -v "$(pwd)/data/processed:/app/data/processed:ro" \
  -e MLFLOW_TRACKING_URI=/app/mlruns \
  -e PIPELINE_CONFIG_PATH=/app/src/config/pipeline_tabular_classification.yaml \
  mlops-prediction-api
```

## CI/CD

The GitHub Actions workflow (`.github/workflows/ci-pipeline.yml`) includes:

1. **Docker build** — builds the image to verify the Dockerfile is valid.
2. **Import check** — runs the container briefly to verify Python imports.

To trigger the CI workflow:
- Go to the repository on GitHub.
- Navigate to **Actions** > **CI Pipeline**.
- Click **Run workflow** > **Run workflow**.

## Troubleshooting

### "No Production model found in registry"

The pipeline has not been run with a successful promotion yet. Run:
```bash
run-pipeline --config src/config/pipeline_tabular_classification.yaml
```
and approve the model when prompted.

### "Cannot query MLflow Model Registry"

The `mlruns/` directory is not mounted or is empty. Verify:
```bash
ls mlruns/
```
should contain experiment directories. If running in Docker, check the
volume mount in `docker-compose.yml`.

### Port already in use

Change the `API_PORT` in `.env`:
```
API_PORT=8001
```

## Deployment Manifest

Every successful `run-pipeline` invocation emits
`outputs/deployment_manifest.json`. The manifest is a structured,
deterministic description of what was just promoted and how to serve it.
It is not a deployment trigger.

Key fields:

- `readiness.status` — one of `ready`, `no_production_model`,
  `promotion_not_approved`, `mlflow_unavailable`.
- `model.registry_name` / `model.registry_version` / `model.run_id` —
  registry coordinates of the current Production model.
- `service.app_import` / `service.entrypoint_cli` — how to start the
  FastAPI service (`run-api` or `uvicorn src.deployment.app:app`).
- `container.compose_service` / `container.image_name` — references to
  the existing Docker Compose service used for local serving.
- `env_template` — suggested `.env` entries; identical in spirit to
  `.env.example` but populated with the current run's resolved values.

Use the manifest as a deployment intent contract: read it from CI, attach
it to release artifacts, or feed it into your serving rollout tooling.
The manifest does not start Docker or push images.
