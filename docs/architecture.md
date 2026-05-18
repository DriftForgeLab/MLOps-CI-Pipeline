# Architecture

This document describes how the `MLOps-CI-Pipeline` code is organised and how its parts
interact. It is meant as a learning aid: read it alongside the source code, and treat the
code as the source of truth where the two ever disagree.

## Design goals

The architecture is shaped by four goals:

1. **Reproducibility** — the same inputs must always produce the same outputs.
2. **Auditability** — every model that reaches production must have a traceable history.
3. **Separation of concerns** — orchestration, stage logic, and configuration are kept
   independent so each can change without disturbing the others.
4. **Readability** — the project is meant for learning, so clarity is preferred over
   cleverness.

## Layered structure

The `src/` package is organised as layers. Data conceptually flows top to bottom, with a
thin orchestration layer sitting across all of them:

```
config   →  data  →  training  →  evaluation  →  drift / promotion  →  registry  →  deployment  →  monitoring
                              (orchestrated by src/pipeline/)
```

| Layer | Package | Responsibility |
|-------|---------|----------------|
| Configuration | `src/config/` | YAML contracts, frozen dataclasses, and validation. |
| Data | `src/data/` | Ingestion, content-addressed versioning, validation, splitting, preprocessing, the ISP pipeline, and drift-adaptation helpers. |
| Training | `src/training/` | Per-task trainers (classification, regression, image CNN) and a shared tabular helper. |
| Evaluation | `src/evaluation/` | Metric computation and pre-deployment robustness analysis. |
| Drift & promotion | `src/drift/`, `src/promotion/` | Drift computation/interpretation, the promotion rule engine, the comparator, and the approval gate. |
| Registry | `src/registry/` | MLflow model-registry integration, local artifact metadata, and rollback. |
| Deployment | `src/deployment/` | The FastAPI prediction service and its startup checks. |
| Monitoring | `src/monitoring/` | Drift-monitoring CLIs, the drift history index, and reporting. |
| Orchestration | `src/pipeline/` | The `run-pipeline` entry point, the stage registry, MLflow logging, and the run report. |

## Orchestration: the stage registry

`src/pipeline/run_pipeline.py` is the `run-pipeline` entry point. It parses arguments,
loads and validates the config, runs the four unconditional data-preparation steps
(detect → version → validate → split), configures MLflow, and then iterates over the
stages listed in the config's `pipeline_stages`.

Each stage is dispatched through `execute_stage()` in `src/pipeline/steps.py`, which
looks the stage up in the `_STAGE_REGISTRY` dictionary, times its execution, captures
exceptions, and returns a `StageResult`. A stage can finish as `completed`, `failed`, or
`blocked`; the first non-`completed` result stops the remaining stages.

This registry pattern means **adding a new stage requires only a new dictionary entry
and a validation constant** — the orchestrator itself does not change. It also keeps the
orchestrator small: it sequences stages but contains no stage logic.

## File-based stage I/O

Stages do not pass large objects to one another in memory. Instead, each stage writes
its outputs to disk, and the next stage reads them back. For example, preprocessing
writes `feature_map.json`, which training and evaluation then load and enforce as a
feature contract.

This has two consequences worth understanding:

- All file writes go through the atomic helpers in `src/common/io.py`, so an interrupted
  run never leaves a half-written file for a later run to consume.
- Because state lives on disk, stages are easy to inspect and re-run independently — a
  useful property for both debugging and teaching.

## Immutable configuration

Every config loader in `src/config/` returns a `@dataclass(frozen=True)` instance.
Stages receive the configuration by reference and cannot mutate it. This prevents a
class of bugs where one stage silently changes shared state that a later stage
depends on.

## MLflow as the system of record

MLflow stores run parameters, metrics, artifacts, lineage tags, and the model registry.
The filesystem (`artifacts/`, `outputs/`) keeps redundant, human-readable copies.

Two design choices are deliberate:

- **Training runs are immutable.** Runtime drift monitoring opens its own separate,
  non-nested MLflow runs so that ongoing observations never pollute the provenance of a
  training run.
- **Lineage tags are hard-validated.** The promotion flow refuses to finish if any
  required `lineage.*` tag is missing, so the audit trail can never be partial.

## Reproducibility anchors

Every run is identified by four values that are propagated consistently into MLflow tags:

1. `pipeline_execution_id` — a UUID generated per run.
2. `config_hash` — a SHA-256 hash of the pipeline YAML.
3. `dataset_version_id` — a content hash of the raw data, also used as a directory name.
4. `random_seed` — declared in the config and propagated to the scikit-learn and PyTorch
   random number generators.

Together, `(config_hash, dataset_version_id, random_seed)` is intended to determine the
training output on a given platform.

## The prediction service

The FastAPI application in `src/deployment/` is **read-only** with respect to training
state. At startup it queries the MLflow registry, loads every model at the allowed stage
(default `Production`) into memory, and serves predictions per model via
`POST /predict/{model_name}`. No training or promotion logic is reachable from the API.

## Where to read next

- [deployment.md](deployment.md) — running and configuring the prediction service.
- [drift-detection.md](drift-detection.md) — the drift-monitoring and retraining workflow.
- [evaluation_plan.md](evaluation_plan.md) — how models are evaluated and gated.
- [image_datasets.md](image_datasets.md) — the image and raw-DNG pipelines.
