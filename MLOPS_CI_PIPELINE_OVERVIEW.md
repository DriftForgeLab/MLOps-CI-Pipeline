# MLOps-CI-Pipeline — Technical Overview and Architectural Review

*A thesis-ready analysis of the repository `MLOPS-CI-PIPELINE`, synchronised with the code on the `main` branch.*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [End-to-End Pipeline Flow](#3-end-to-end-pipeline-flow)
4. [Execution Flow](#4-execution-flow)
5. [CI/CD Analysis](#5-cicd-analysis)
6. [Tools, Frameworks, and Methods](#6-tools-frameworks-and-methods)
7. [Architecture and Component Interaction](#7-architecture-and-component-interaction)
8. [MLOps Practices Present in the Project](#8-mlops-practices-present-in-the-project)
9. [Strengths, Weaknesses, and Gaps](#9-strengths-weaknesses-and-gaps)
10. [Thesis-Ready Summary](#10-thesis-ready-summary)
11. [Evidence Legend](#11-evidence-legend)

---

## 1. Project Overview

The repository implements a **lightweight, drift-aware MLOps pipeline** that automates the lifecycle of a supervised machine-learning model from raw data ingestion through controlled, auditable promotion to a containerised prediction service. The package is declared in [pyproject.toml](pyproject.toml) as `lightweight-mlops-pipeline` and described as a *"Lightweight drift-aware MLOps pipeline for controlled retraining and deployment"*. The [README.md](README.md) positions it as an end-to-end demonstrator of automated retraining, evaluation-based promotion, versioned deployment, and data drift monitoring, developed as a bachelor-level engineering project.

**Primary objective (Confirmed from code).** The system operationalises three concrete ML task types — enumerated in `VALID_TASK_TYPES` at [src/config/schema.py:15](src/config/schema.py#L15) — through a single CLI entry point `run-pipeline`:

1. **Tabular classification** (`task_type: classification`) on CSV datasets, using scikit-learn estimators (Random Forest, Logistic Regression).
2. **Tabular regression** (`task_type: regression`) on CSV datasets (Random Forest Regressor, Linear Regression).
3. **Image classification** (`task_type: image_classification_cnn`) on JPG/PNG inputs or on raw DNG inputs pre-processed through a deterministic seven-stage Image Signal Processing (ISP) pipeline. Both variants share the same PyTorch CNN trainer; the distinction is purely a preprocessing choice (`raw_input: true` plus an `isp:` block vs. a standard resize/normalise block).

The project's governance mechanisms (promotion rules, lineage tagging, approval gate, rollback, drift monitoring, containerised serving) are identical across all task types. The central engineering theme is therefore not the model itself but the **repeatable, auditable lifecycle surrounding the model**.

**Recent additions (Confirmed from code).** Two capabilities are present that are material to the architecture but were absent from earlier documentation:

- A **drift-adaptive fine-tuning workflow** that couples the `prepare-drift-training` CLI with a `--fine-tune` run of `run-pipeline`. Labeled drifted images are split into a training portion (copied into the raw dataset) and a held-out evaluation portion against which the current Production model is baselined. The subsequent fine-tune run is automatically re-evaluated on the same holdout and a before/after delta is surfaced to the approval gate.
- **Optional GPU acceleration.** A unified torch-device resolver ([src/common/device.py](src/common/device.py)) honours a `MODEL_DEVICE` environment variable (`auto`/`cpu`/`cuda`/`directml`) with a CUDA → DirectML → CPU fallback. An opt-in `[directml]` extra in [pyproject.toml:32-36](pyproject.toml#L32-L36) enables Windows AMD/Intel GPU acceleration through `torch-directml`. The ISP pipeline additionally exposes a `gpu_accelerated` flag routed through [src/data/isp_gpu.py](src/data/isp_gpu.py).

**Likely engineering context (Strong inference).** Commit history, references to *"sprints"*, self-identification as a bachelor-level project, and the coexistence of raw-DNG / ISP handling suggest an academic engineering thesis combining applied MLOps with image-processing research. The inclusion of sensor-level ISP scenario simulation for drift analysis indicates a focus on deployment robustness of image classifiers under physically motivated distribution shifts.

---

## 2. Repository Structure

The top-level layout cleanly separates source code, configuration, data, artifacts, infrastructure, tests, and documentation.

```
MLOPS-CI-Pipeline/
├── .github/workflows/ci-pipeline.yml      GitHub Actions CI workflow (manual dispatch only)
├── docker/
│   ├── Dockerfile                         Containerised prediction service image
│   └── docker-compose.yml                 Local deployment orchestration
├── docs/                                  Architecture, deployment, drift, evaluation,
│                                          threat-model, and decisions documents
├── src/                                   Python package `src` (installed via pip -e .)
│   ├── common/
│   │   ├── io.py                          Atomic write helpers
│   │   └── device.py                      Torch device resolver (CUDA / DirectML / CPU)
│   ├── config/                            YAML loaders, frozen dataclasses, validation
│   ├── data/                              Ingestion, versioning, validation, splitting,
│   │                                       tabular + image + raw-ISP preprocessing,
│   │                                       ISP pipeline (CPU + GPU), drift-adaptive
│   │                                       holdout prep, batch preparation CLI
│   ├── drift/                             Drift primitives: alignment, Evidently,
│   │                                       image compute, interpretation
│   ├── evaluation/                        Metric computation, ISP sensitivity,
│   │                                       augmentation robustness
│   ├── promotion/                         Rule engine, comparator, approval gate
│   ├── registry/                          MLflow registry integration, rollback CLI
│   ├── training/                          Per-task trainers + shared tabular core
│   ├── pipeline/                          run_pipeline entry, stage registry,
│   │                                       MLflow logger, run report
│   ├── deployment/                        FastAPI app, routes, schemas, startup checks
│   ├── monitoring/                        Online monitoring CLIs, history index,
│   │                                       MLflow sink for runtime drift, reports,
│   │                                       drift-training preparation CLI
│   └── lifecycle/                         states.py + transitions.py (scaffolding;
│                                           see §9 — not wired into the stage registry)
├── scripts/
│   └── download_fivek.py                  Helper that fetches MIT-Adobe FiveK raw images
├── data/
│   ├── raw/<dataset>/                     Immutable input datasets + dataset.yaml
│   ├── processed/<dataset>/<version_id>/  Content-addressed snapshots + splits
│   ├── drift_scenarios/                   Generated ISP drift scenario artifacts
│   ├── batches/images/                    Incoming drifted production batches (convention)
│   └── evaluation/drifted_holdout/        Drift-adaptive holdout images + baseline JSON
├── artifacts/runs/<version_id>/model/     Local model artifact cache
├── mlruns/                                MLflow file-based tracking store
├── outputs/                               Run reports, evaluation reports, promotion
│                                          decisions, drift monitoring, drift-adaptation evals
├── tests/                                 Pytest test suite (46 test modules — Confirmed)
├── pyproject.toml                         Package + dependency + CLI entry definitions
├── README.md                              User-facing operational guide
├── LICENSE                                MIT-style license
├── .env.example                           Environment-variable template for Docker
└── .dockerignore / .gitignore
```

Large, generated, or environment-specific content — `.venv/`, `__pycache__/`, `.env`, `artifacts/`, `data/processed/`, `data/drift_scenarios/`, `data/batches/`, `outputs/`, `mlruns/`, and all raw image binaries — is excluded via [.gitignore](.gitignore). Only `dataset.yaml` and per-dataset `README.md` files inside `data/raw/` are tracked.

**Top-level `ci/` directory (Corrected).** The top-level `ci/` directory referenced in earlier documentation is **not present on disk** — no empty YAML stubs exist. The active CI workflow lives under [.github/workflows/ci-pipeline.yml](.github/workflows/ci-pipeline.yml) and is the only CI surface in the repository.

**Purpose of the major directories (Confirmed from code):**

- `src/pipeline/` is the *orchestration* layer. [src/pipeline/run_pipeline.py](src/pipeline/run_pipeline.py) owns argument parsing, data preparation, MLflow setup, stage dispatch, run-report assembly, and overall exit status. [src/pipeline/steps.py](src/pipeline/steps.py) hosts the stage registry `_STAGE_REGISTRY` at [src/pipeline/steps.py:593-600](src/pipeline/steps.py#L593-L600) that maps stage names to implementation functions, wrapping each execution with timing, structured logging, and a `StageResult` dataclass.
- `src/config/` centralises every YAML contract. [src/config/schema.py](src/config/schema.py) contains frozen dataclasses and validation constants for pipeline, training, preprocessing, evaluation, promotion, deployment, and drift configuration. [src/config/loader.py](src/config/loader.py) re-exports `load_*` functions from dedicated loader modules so callers import a single facade. YAML files (`pipeline_tabular.yaml`, `pipeline_image.yaml`, `pipeline_raw_image.yaml`, `pipeline_fivek.yaml`, plus the `preprocessing_*`, `training_*`, `promotion.yaml`, `evaluation.yaml`, `deployment.yaml`, and `drift.yaml` contracts) live next to the loaders. The additional `pipeline_fivek.yaml` is a raw-image preset targeting the MIT-Adobe FiveK dataset and is paired with [scripts/download_fivek.py](scripts/download_fivek.py).
- `src/data/` performs content-addressed versioning ([src/data/versioning.py](src/data/versioning.py)), schema validation ([src/data/validate.py](src/data/validate.py)), deterministic splitting ([src/data/split.py](src/data/split.py), [src/data/image_split.py](src/data/image_split.py)), preprocessing ([src/data/preprocess.py](src/data/preprocess.py) for tabular; [src/data/image_preprocess.py](src/data/image_preprocess.py) for images), the seven-stage ISP pipeline ([src/data/isp_pipeline.py](src/data/isp_pipeline.py)) with an optional GPU implementation ([src/data/isp_gpu.py](src/data/isp_gpu.py)), ISP drift simulation ([src/data/image_drift_simulator.py](src/data/image_drift_simulator.py)), batch preparation for monitoring ([src/data/prepare_batch.py](src/data/prepare_batch.py), CLI wrapper [src/data/prepare_batch_cli.py](src/data/prepare_batch_cli.py)), and the drift-adaptation primitives ([src/data/drift_adaptation.py](src/data/drift_adaptation.py)) used by the fine-tuning workflow.
- `src/training/` contains `classification/`, `regression/`, and `image_classification_cnn/` sub-packages plus a shared helper ([src/training/_tabular.py](src/training/_tabular.py)) and an estimator factory ([src/training/model_factory.py](src/training/model_factory.py)). An **empty** `image_classification/` directory remains (only `__pycache__` is present — **Confirmed from disk**); it is a leftover from an earlier refactor and is not referenced by the stage registry. The CNN trainer ([src/training/image_classification_cnn/train.py](src/training/image_classification_cnn/train.py)) defines an sklearn-compatible `SimpleCNN` PyTorch module with a configurable architecture and supports both from-scratch training and fine-tuning of an existing Production model.
- `src/evaluation/` computes metrics ([src/evaluation/evaluate.py](src/evaluation/evaluate.py), [src/evaluation/metrics.py](src/evaluation/metrics.py)), ISP sensitivity ([src/evaluation/image_drift_analysis.py](src/evaluation/image_drift_analysis.py)), and augmentation robustness ([src/evaluation/standard_image_robustness.py](src/evaluation/standard_image_robustness.py)). A legacy `promotion_rules.py` module co-exists here; the authoritative rule engine is in `src/promotion/`.
- `src/promotion/` holds the rule engine ([src/promotion/rules.py](src/promotion/rules.py)), the interactive approval gate ([src/promotion/approval.py](src/promotion/approval.py)), and the metric comparator ([src/promotion/comparator.py](src/promotion/comparator.py)).
- `src/drift/` contains dataset alignment ([src/drift/alignment.py](src/drift/alignment.py)), Evidently-based tabular compute ([src/drift/compute.py](src/drift/compute.py)), image drift compute ([src/drift/image_compute.py](src/drift/image_compute.py)), and result interpretation / severity grading ([src/drift/interpret.py](src/drift/interpret.py)).
- `src/registry/` integrates with the MLflow Model Registry ([src/registry/model_registry.py](src/registry/model_registry.py)), persists local artifacts ([src/registry/metadata.py](src/registry/metadata.py)), records the promotion decision ([src/registry/promotion_record.py](src/registry/promotion_record.py)), and exposes the `rollback-model` CLI ([src/registry/rollback.py](src/registry/rollback.py)).
- `src/deployment/` contains a FastAPI application ([src/deployment/app.py](src/deployment/app.py)), route handlers including a static HTML UI ([src/deployment/routes.py](src/deployment/routes.py)), Pydantic schemas ([src/deployment/schemas.py](src/deployment/schemas.py)), and the startup governance checks ([src/deployment/startup_checks.py](src/deployment/startup_checks.py)).
- `src/monitoring/` exposes the operational CLIs `monitor-drift`, `monitor-drift-image`, `monitor-summary`, `prepare-image-batch`, and `prepare-drift-training`. It also contains the drift history index ([src/monitoring/history.py](src/monitoring/history.py)), a standalone MLflow sink for runtime drift ([src/monitoring/mlflow_sink.py](src/monitoring/mlflow_sink.py)), the decision gate ([src/monitoring/drift_decision.py](src/monitoring/drift_decision.py)), and reporting helpers ([src/monitoring/reports.py](src/monitoring/reports.py)).
- `src/lifecycle/` contains only `states.py` and `transitions.py` (**Unclear / needs verification** — the package exposes state/transition primitives but is not imported by the orchestrator, stage registry, or registry promotion flow; at present it reads as scaffolding rather than live infrastructure).

The project structure therefore reflects a typical **layered MLOps architecture**: configuration → data → training → evaluation → drift/promotion → registry → deployment → monitoring, with a thin orchestration layer (`src/pipeline/`) sitting across these layers.

---

## 3. End-to-End Pipeline Flow

The pipeline exposes six stages through its stage registry in [src/pipeline/steps.py:593-600](src/pipeline/steps.py#L593-L600): `preprocessing`, `training`, `evaluation`, `model_analysis`, `promotion`, and `deployment`. The `deployment` stage is still a no-op placeholder — **Confirmed from code** at [src/pipeline/steps.py:70-71](src/pipeline/steps.py#L70-L71) (`_placeholder_stage` logs a single "no-op" line) and [src/pipeline/steps.py:599](src/pipeline/steps.py#L599). Before any stage executes, `run_pipeline` performs four hard-coded preparation steps: dataset-yaml detection, dataset versioning, validation, and stratified splitting ([src/pipeline/run_pipeline.py:94-102](src/pipeline/run_pipeline.py#L94-L102)).

### 3.1 Data Ingestion and Versioning

Raw data lives in `data/raw/<dataset>/`. For tabular datasets this is a single `data.csv` plus a `dataset.yaml` describing name, task type, features, target, schema, and constraints. For image datasets, it is an `images/<class>/...` directory tree with the same `dataset.yaml` structure (see [docs/image_datasets.md](docs/image_datasets.md)).

[src/data/create_dataset_yaml.py](src/data/create_dataset_yaml.py) can interactively generate `dataset.yaml` on first run for unknown datasets. It is called with a `non_interactive` flag driven by `sys.stdin.isatty()` so it degrades gracefully under headless CI ([src/pipeline/run_pipeline.py:95](src/pipeline/run_pipeline.py#L95)).

[src/data/versioning.py](src/data/versioning.py) creates a content-addressed snapshot under `data/processed/<dataset>/<version_id>/`. The `version_id` is the first 12 hex characters of the SHA-256 of the raw CSV for tabular data, or a folder-hash (see [src/data/image_utils.py](src/data/image_utils.py)) for image data. This yields deterministic, reproducible dataset identity — **identical data always produces the same `version_id`**.

### 3.2 Validation and Splitting

[src/data/validate.py](src/data/validate.py) enforces both *structural* validation (files present, YAML keys correct) and *content* validation (dtypes, null fraction, label set, minimum row count) against the declared `dataset.yaml`. [src/data/split.py](src/data/split.py) and [src/data/image_split.py](src/data/image_split.py) then perform deterministic, seed-driven stratified train/val/test splits. Classification tasks are stratified by label to guarantee class presence in each split.

### 3.3 Preprocessing Stage

`_preprocessing_stage` ([src/pipeline/steps.py:74-97](src/pipeline/steps.py#L74-L97)) dispatches to the tabular or image preprocessing module depending on `task_type`. Key properties:

- **Idempotent**: a SHA-256 fingerprint of the preprocessing configuration and inputs is stored in `preprocessed/metadata.json`; matching hashes short-circuit the work.
- **Contract-first**: writes a `feature_map.json` that becomes the authoritative list of model input columns and the target column. All downstream stages (training, evaluation, API, drift monitoring) load and enforce this contract.
- **Policy-driven tabular transforms**: [src/config/preprocessing_tabular.yaml](src/config/preprocessing_tabular.yaml) controls optional one-hot / ordinal encoding, standard / min-max scaling, imputation strategies, and fail-on-null enforcement via a scikit-learn `ColumnTransformer`.
- **Image pipeline**: resizes, normalises, and (for raw inputs) pushes each image through the seven-stage ISP pipeline. The ISP pipeline can be executed on the CPU or through the GPU implementation in [src/data/isp_gpu.py](src/data/isp_gpu.py), gated by the `gpu_accelerated: true` flag under the preprocessing `isp:` block (`ISPConfig.gpu_accelerated` at [src/config/schema.py:273](src/config/schema.py#L273)). For raw-image runs the stage additionally calls `log_isp_versioning_to_mlflow` so the ISP pipeline version and key config parameters are recorded as MLflow tags ([src/pipeline/steps.py:87-97](src/pipeline/steps.py#L87-L97)).

Outputs are materialised to `data/processed/<dataset>/<version_id>/preprocessed/` as `train/val/test.csv` (tabular) or `train/val/test.npz` (image), together with `feature_map.json`, `metadata.json`, and (tabular only) the serialised `pipeline.pkl` (scikit-learn `Pipeline`).

### 3.4 Training Stage

`_training_stage` ([src/pipeline/steps.py:100-120](src/pipeline/steps.py#L100-L120)) dispatches to one of three trainers based on `config.task_type`:

- `classification` → [src/training/classification/train.py](src/training/classification/train.py), which delegates to `run_tabular_training` in [src/training/_tabular.py](src/training/_tabular.py). The trainer loads `feature_map.json`, hard-fails on column mismatch, reads `train.csv`, instantiates the estimator via [src/training/model_factory.py](src/training/model_factory.py), and fits it. It returns a `TrainingResult` with algorithm, hyperparameters, dataset_version_id, random seed, trained-at timestamp, and training row count.
- `regression` → [src/training/regression/train.py](src/training/regression/train.py) (thin wrapper over the same shared helper).
- `image_classification_cnn` → [src/training/image_classification_cnn/train.py](src/training/image_classification_cnn/train.py), which constructs a `SimpleCNN` PyTorch module with configurable conv-layer channels, kernel sizes, FC units, and dropout (driven by [src/config/training_image_cnn.yaml](src/config/training_image_cnn.yaml)), trains with Adam + `CrossEntropyLoss`, and exposes an sklearn-compatible `predict` method for evaluation reuse. The trainer resolves its torch device via `src/common/device.py`, enabling CUDA/DirectML acceleration when available.

A `--fine-tune` flag on the `run-pipeline` CLI switches CNN training from *from-scratch* to *continue from Production weights* using the `fine_tune` hyperparameter block (fewer epochs, lower learning rate, optional backbone freeze) defined by `CNN_FINE_TUNE_KEYS` at [src/config/schema.py:30](src/config/schema.py#L30). For non-CNN tasks the flag is accepted but has no effect (the random-forest-style estimators have no weights to fine-tune) and a log line in [src/pipeline/steps.py:102-107](src/pipeline/steps.py#L102-L107) makes this explicit.

The trained artefact is saved locally via [src/registry/metadata.py](src/registry/metadata.py) to `artifacts/runs/<version_id>/model/model.joblib` (scikit-learn) or `model.pt` (PyTorch; the tensor is forced to CPU before serialisation so artefacts are device-portable). The training result is then logged to MLflow via `log_training_to_mlflow` and `log_training_artifacts_to_mlflow` in [src/pipeline/mlflow_logger.py](src/pipeline/mlflow_logger.py).

### 3.5 Evaluation Stage

`_evaluation_stage` delegates to `evaluate()` in [src/evaluation/evaluate.py](src/evaluation/evaluate.py). The evaluator:

1. Loads the model artefact from `artifacts/runs/<version_id>/model/` (either `model.pt` via `torch.load(..., weights_only=False, map_location="cpu")` — then moved to the resolved device — or `model.joblib` via `joblib.load`).
2. Re-loads `feature_map.json` to enforce the feature contract on evaluation data.
3. Loads the **validation** split (not the held-out test split — **Confirmed from code** at [src/evaluation/evaluate.py:91-112](src/evaluation/evaluate.py#L91-L112)) as either CSV (tabular) or NPZ (image). CNN tensors are transposed from `NHWC` to `NCHW` when needed.
4. Runs `model.predict(X)` and computes metrics via `_compute_metrics`: `accuracy`, `precision`, `recall`, `f1_score` for classification (with configurable averaging from `evaluation.yaml`); `mae`, `mse`, `rmse`, `r2` for regression. Confusion matrices are additionally computed for CNN image classification.
5. Queries `get_production_model_metrics()` in [src/registry/model_registry.py](src/registry/model_registry.py) to fetch metrics of the currently Production-staged model and runs `compare_metrics()` from [src/promotion/comparator.py](src/promotion/comparator.py) to produce a per-metric delta and an `overall_verdict`. If no Production model exists yet, `no_baseline_comparison()` is returned instead (bootstrap scenario).
6. Writes `outputs/evaluation_report.json`, logs metrics and the comparison to MLflow, and returns the report to the orchestrator.

### 3.6 Model Analysis Stage (image pipelines only)

[src/pipeline/steps.py:140-221](src/pipeline/steps.py#L140-L221) defines `_model_analysis_stage`, explicitly documented as *"offline pre-deployment analysis, not drift detection"*. Two gates precede the analysis: (1) it is skipped when `_MIN_ANALYSIS_ACCURACY = 0.6` is not exceeded — running a robustness analysis on an untrained-looking model would be uninformative — and (2) in interactive terminals the user is prompted whether to run it at all (raw ISP runs can be lengthy). The stage then branches on whether the preprocessing config has `raw_input=True` and an `isp:` block:

- **Raw DNG / ISP path** → `_run_isp_simulation_and_sensitivity` ([src/pipeline/steps.py:224-305](src/pipeline/steps.py#L224-L305)). Scenarios declared under `image.scenarios` in [src/config/drift.yaml](src/config/drift.yaml) (warm/cool white-balance shifts, noise injection, dark/bright gamma, heavy sharpening) are used by [src/data/image_drift_simulator.py](src/data/image_drift_simulator.py) to regenerate the training images with perturbed ISP parameters. `run_sensitivity_analysis` in [src/evaluation/image_drift_analysis.py](src/evaluation/image_drift_analysis.py) then measures how much each scenario moves classification accuracy / F1 relative to the baseline. Results are persisted as `sensitivity_report.json` and `sensitivity_report.html`, and per-scenario metrics plus analysis tags are mirrored into MLflow.
- **Standard JPG/PNG path** → `_run_augmentation_robustness` ([src/pipeline/steps.py:308-362](src/pipeline/steps.py#L308-L362)). Eight fixed augmentations (horizontal flip, rotation, brightness / contrast shifts, Gaussian noise, Gaussian blur) are applied to the validation split and the accuracy delta is measured per augmentation.

Both branches function as a *pre-deployment stress test*: they quantify the model's sensitivity to plausible transformations **before** it reaches Production. For raw pipelines the resulting scenario signatures are later consumed by `monitor-drift-image` to attach physical interpretations to observed runtime drift.

### 3.7 Promotion Stage

`_promotion_stage` at [src/pipeline/steps.py:381-481](src/pipeline/steps.py#L381-L481) is the governance heart of the pipeline:

1. Loads `outputs/evaluation_report.json`.
2. Executes the rule engine in [src/promotion/rules.py](src/promotion/rules.py) using thresholds from [src/config/promotion.yaml](src/config/promotion.yaml). Any violation raises `PromotionBlockedError` (defined at [src/pipeline/steps.py:135-137](src/pipeline/steps.py#L135-L137)) with a structured error that enumerates every failed rule. This is not a system failure — the orchestrator treats it as its own `blocked` outcome (see §4.4).
3. Logs the comparison verdict against the current Production model.
4. Resolves the registry model name and tries to load the most recent drift result for that model via `load_latest_drift` ([src/monitoring/history.py](src/monitoring/history.py)). The summary is presented to the approver alongside the evaluation report.
5. If the run is a `--fine-tune` run and `outputs/drift_adaptation_eval.json` exists, loads it and passes the before/after comparison into the approval gate (`request_approval(..., drift_eval=drift_eval)` at [src/pipeline/steps.py:444](src/pipeline/steps.py#L444)).
6. Calls `request_approval()` in [src/promotion/approval.py](src/promotion/approval.py), which prints a human-readable summary (metrics, comparison, drift snapshot, optional drift-adaptation delta) and requires an interactive approve / reject / cancel decision. Rejection demands a mandatory reason; empty reasons and `EOFError` / `KeyboardInterrupt` are treated as cancellations that raise a `ValueError` and abort the pipeline.
7. Writes `outputs/promotion_decision.json` and logs the decision to MLflow (including a `_drift_provenance` block so the decision record is auditable even if the drift JSON is later rotated).
8. On approval, calls `register_approved_model()` and `promote_to_production()` from [src/registry/model_registry.py](src/registry/model_registry.py). These register the MLflow run's model under a deterministic registry name and transition the new version to stage `Production`, archiving the previous version.
9. Calls `attach_lineage_tags()`, which writes a comprehensive set of `lineage.*` tags (run_id, dataset version, algorithm, trained_at, promotion outcome, comparison verdict, metrics). The function **hard-fails** if any required lineage tag is missing, preventing an incomplete audit trail.

### 3.8 Drift-Adaptive Fine-Tuning Pre-Promotion Hook (fine-tune runs only)

This hook is **not** a stage in `_STAGE_REGISTRY`; it is wired directly into the orchestration loop in [src/pipeline/run_pipeline.py:131-144](src/pipeline/run_pipeline.py#L131-L144). When `--fine-tune` is set and the next stage is `promotion`, `run_drift_adaptation_eval` ([src/pipeline/steps.py:484-590](src/pipeline/steps.py#L484-L590)) runs *immediately before* the promotion stage. It:

1. Looks for a holdout at `data/evaluation/drifted_holdout/<dataset>/` and a sibling baseline-metrics JSON written by `prepare-drift-training`. Returns `None` (silent skip) if either is missing, so the hook is a no-op on non-adaptive fine-tune runs.
2. Loads the newly fine-tuned model from `artifacts/runs/<version_id>/model/model.pt`.
3. Loads the preprocessing config and either the normalization stats saved alongside the holdout (preferred, preserved across dataset-version churn) or the training stats of the current dataset version.
4. Runs `evaluate_on_holdout_dir` to produce post-fine-tune metrics on the held-out drifted images, computes a per-metric delta against the baseline, and returns `{baseline, after_finetuning, delta, improved}`.
5. Writes `outputs/drift_adaptation_eval.json` and logs `drift_adapt.delta_*` metrics plus `drift_adapt.after_accuracy` to the active MLflow run, and attaches the JSON as an MLflow artifact.

The subsequent `_promotion_stage` reads this same JSON and hands it to the approval gate so the human approver sees a concrete before/after comparison on held-out drifted data.

### 3.9 Deployment Stage

`deployment` is registered as a no-op placeholder in the stage registry ([src/pipeline/steps.py:599](src/pipeline/steps.py#L599) → `_placeholder_stage`). The actual deployment is *out-of-band*: a separate FastAPI service is started via the `run-api` command or the Docker container, and at startup it queries the MLflow Model Registry and loads every model version at the allowed stage (default `Production`) into memory ([src/deployment/app.py:26-35](src/deployment/app.py#L26-L35), `load_all_production_models` in [src/deployment/startup_checks.py](src/deployment/startup_checks.py)). There is no rolling deployment, canary, or blue-green switch triggered from the pipeline itself — the operator restarts the container to pick up a newly promoted version.

### 3.10 Monitoring and Retraining

Online drift detection is implemented in [src/monitoring/drift.py](src/monitoring/drift.py) (tabular) and [src/monitoring/image_drift_monitor.py](src/monitoring/image_drift_monitor.py) (image). These are invoked manually after real production data is collected via the CLIs `monitor-drift`, `monitor-drift-image`, and `monitor-summary`. Tabular drift uses Evidently's `DataDriftPreset` plus per-feature extraction in [src/drift/compute.py](src/drift/compute.py); image drift uses per-channel Wasserstein distance and MMD in CNN feature space in [src/drift/image_compute.py](src/drift/image_compute.py).

Each run produces a JSON snapshot under `outputs/drift_monitoring/<model-name>/` and an entry in a `history.jsonl` index ([src/monitoring/history.py](src/monitoring/history.py)) so later commands (`monitor-summary`, the promotion approval summary) can read back the most recent result. Runtime drift is also forwarded to a **dedicated standalone** MLflow run via [src/monitoring/mlflow_sink.py](src/monitoring/mlflow_sink.py) — it is intentionally *not* nested inside any training run, so the immutable training provenance stays separate from ongoing runtime observations.

If severity exceeds the configured `alert_severity`, the user is prompted through [src/monitoring/drift_decision.py](src/monitoring/drift_decision.py) to choose a response (retrain, collect more data, accept, escalate). Retraining itself is not automated — it requires re-invoking `run-pipeline`. The recommended response to detected image drift is the *drift-adaptive* workflow: organise the drifted batch into an ImageFolder layout, run `prepare-drift-training` to split it into a training portion (added to the raw dataset) and a held-out evaluation set with a Production-model baseline, then run `run-pipeline --fine-tune` so the pre-promotion hook (§3.8) can produce an automatic before/after comparison for the approver.

---

## 4. Execution Flow

### 4.1 Entry Points

The installable package exposes **eight** CLI commands through `[project.scripts]` in [pyproject.toml:38-46](pyproject.toml#L38-L46):

| Command | Target | Purpose |
|---|---|---|
| `run-pipeline` | `src.pipeline.run_pipeline:main` | End-to-end training + promotion (supports `--fine-tune`) |
| `rollback-model` | `src.registry.rollback:main` | Roll a previous version to Production |
| `run-api` | `src.deployment.app:main` | Launch the FastAPI prediction service |
| `monitor-drift` | `src.monitoring.monitor_cli:main` | Tabular batch drift detection |
| `monitor-drift-image` | `src.monitoring.image_monitor_cli:main` | Image batch drift detection |
| `monitor-summary` | `src.monitoring.monitor_summary_cli:main` | Aggregated drift report |
| `prepare-image-batch` | `src.data.prepare_batch_cli:main` | Pre-process new images for drift monitoring |
| `prepare-drift-training` | `src.monitoring.prepare_drift_cli:main` | Prepare labeled drifted images for drift-adaptive fine-tuning (baseline + holdout + copy into dataset) |

### 4.2 Orchestration Logic in `run-pipeline`

`main()` in [src/pipeline/run_pipeline.py](src/pipeline/run_pipeline.py) performs the following ordered steps:

1. Parse `--config <path>` (required) and the optional boolean `--fine-tune` flag.
2. Load and validate the config through `load_config()` — this traverses the frozen dataclass graph in `schema.py` and surfaces user-friendly errors for `FileNotFoundError`, `yaml.YAMLError`, and `ValueError`; each raises `SystemExit(1)`.
3. Run `detect_and_generate()` → `create_dataset_version()` → `validate_dataset()` → `split_dataset()`. These four data-preparation calls are *unconditional*: they always run before any stage executes.
4. Compute a SHA-256 hash of the config file (`compute_config_hash`) and generate a random `pipeline_execution_id` (UUID).
5. Call `configure_mlflow()` which sets the tracking URI, starts a run, and attaches pipeline-level tags (`pipeline.execution_id`, `pipeline.config_hash`, `pipeline.task_type`, `pipeline.dataset_version_id`, `pipeline.fine_tune`, etc.). MLflow setup failures are logged but do not abort the run.
6. Iterate over `config.pipeline_stages`. If `--fine-tune` is set and the next stage is `promotion`, call `run_drift_adaptation_eval` first (§3.8) and log its delta metrics. Then call `execute_stage(stage_name, config, version_id, fine_tune=fine_tune)`. Each call returns a `StageResult` dataclass with status (`completed`, `failed`, or `blocked`), timestamps, duration, and optional error string. On the first `failed` or `blocked` status, the remaining stages are skipped.
7. In the `finally` block the orchestrator builds the run report (`build_run_report()` / `write_run_report()` in [src/pipeline/report.py](src/pipeline/report.py)), attaches the config file and — if `model_analysis` completed — the ISP scenario artifacts to MLflow, sets `pipeline.overall_status`, and ends the run.
8. Exit with code 0 (success), 1 (failure), or 2 (blocked by promotion rules — distinct, user-visible message printed to stdout).

### 4.3 Stage Dispatch Pattern

`execute_stage` ([src/pipeline/steps.py:603-667](src/pipeline/steps.py#L603-L667)) looks up the stage function in `_STAGE_REGISTRY`, records `START` / `END` markers with wall-clock timing, catches `PromotionBlockedError` distinctly (status = `blocked`, logged at WARNING) and any other exception generically (status = `failed`, logged at ERROR with traceback), and returns a `StageResult`. Adding a new stage therefore only requires a new entry in the registry and a validation constant in `VALID_PIPELINE_STAGES` ([src/config/schema.py:20](src/config/schema.py#L20)) — the orchestrator remains untouched.

### 4.4 Manual vs Automated Execution

- **Manual / local**: `run-pipeline --config src/config/pipeline_tabular.yaml` from the command line. The approval gate in `_promotion_stage` requires `stdin.isatty()` for interactive input; in headless environments `input()` receives an `EOFError` and is treated as a cancellation, so the pipeline exits before registering a model.
- **Automated / CI**: Triggered through GitHub Actions (see §5). The CI run uses the tabular config, and because the promotion gate is interactive it will cancel gracefully without promoting a model in CI — the CI objective is to verify end-to-end pipeline wiring and Docker build, not to produce Production artefacts from CI.
- **Containerised serving**: `docker compose -f docker/docker-compose.yml up --build` starts the FastAPI container with read-only mounts for `mlruns/`, `src/config/`, `artifacts/`, and `data/processed/`, and starts uvicorn on port 8000 (overridable via `API_PORT` in `.env`). The container image is intentionally CPU-only — the `[directml]` extra is not installed ([pyproject.toml:34-36](pyproject.toml#L34-L36)).

### 4.5 Rollback Execution Flow

The `rollback-model` CLI in [src/registry/rollback.py](src/registry/rollback.py) exhibits particularly careful failure handling:

1. Fetches all versions of the registered model.
2. Prints them as a formatted table with version, stage, trained-at timestamp, and algorithm.
3. Interactively (or via `--version` / `--yes`) selects a target and demands a rollback reason.
4. Writes a **pre-promotion intent JSON** (`status=initiated`) *before* calling `promote_to_production`, guaranteeing an audit trail even if the subsequent step fails.
5. Calls `promote_to_production(..., promoted_by="rollback-cli")`, which archives the current Production version and transitions the target.
6. Calls `_complete_rollback_audit()`, which writes governance tags (`governance.last_rollback_at`, `governance.rollback_reason`, `governance.rollback_executed_by`), overwrites the intent JSON with `status=completed`, and creates a dedicated MLflow *governance* experiment run. If this mandatory post-step fails, the process exits with a clear message explaining that the registry state has changed but the audit is incomplete.

The CLI also detects and warns about *incomplete prior rollbacks* (partial governance state) rather than silently proceeding.

---

## 5. CI/CD Analysis

Continuous integration is implemented in **GitHub Actions** via [.github/workflows/ci-pipeline.yml](.github/workflows/ci-pipeline.yml). The top-level `ci/` directory referenced in earlier documentation **does not exist** in the current repository tree (**Confirmed from disk**) — no empty placeholder YAMLs are tracked.

### 5.1 Trigger and Execution Model

- **Trigger**: `workflow_dispatch` only. The workflow is **manually invoked** from the GitHub Actions UI; there is no `push`, `pull_request`, or `schedule` trigger. Automatic CI-on-commit is not configured.
- **Runner**: `ubuntu-latest`.
- **Jobs**: a single `run-pipeline` job with nine sequential steps.

### 5.2 Pipeline Steps (Confirmed from code)

1. **Checkout repository** (`actions/checkout@v4`).
2. **Set up Python 3.12** (`actions/setup-python@v5`).
3. **Install dependencies**: `pip install -e . pytest`. Editable installation builds the package and brings in the pinned dependencies from `pyproject.toml`. The optional `[directml]` extra is not installed in CI.
4. **Run tests**: `pytest tests/ -v --tb=short`. The full pytest suite (46 test files across `common/`, `config/`, `data/`, `deployment/`, `drift/`, `evaluation/`, `monitoring/`, `pipeline/`, `promotion/`, `registry/`, `training/`) is executed as a gate before continuing.
5. **Run pipeline**: `run-pipeline --config src/config/pipeline_tabular.yaml`. This executes preprocessing → training → evaluation → promotion on the tabular reference dataset. Because stdin is not a TTY in CI, the approval gate cancels and the promotion stage will not actually write a Production model to the registry (**Strong inference**).
6. **Print run report**: `cat outputs/run_report.json` for logging visibility.
7. **Upload run report**: `actions/upload-artifact@v4` attaches `run_report.json` as a downloadable artefact named `run-report`.
8. **Build Docker image**: `docker build -f docker/Dockerfile -t mlops-prediction-api:ci-${{ github.sha }} .` verifies the image builds cleanly on every CI run.
9. **Verify container imports**: a throwaway container is started to import `src.deployment.startup_checks`, confirming the container's Python path and module wiring are intact.

### 5.3 Scope and Gaps

- **Present**: dependency install, unit / integration tests, full tabular pipeline run, run-report artefact upload, Docker image build, and container smoke test.
- **Missing / Unclear**: no linting (`ruff`, `flake8`, `mypy`) is configured; no branch protection rules or required-status-check configuration can be derived from the repo; no release / deployment workflow pushes the image to a registry (`ghcr.io`, Docker Hub, ECR); no integration test against a running FastAPI service; no environment promotion (staging → production); no scheduled drift-monitoring job; no CI job that exercises the image pipeline (the CI only touches the tabular config).

---

## 6. Tools, Frameworks, and Methods

Every dependency is pinned in [pyproject.toml:11-29](pyproject.toml#L11-L29). The following table maps each dependency to its role.

| Dependency | Pinned version | Role in the system |
|---|---|---|
| `setuptools` | `>=68,<81` | Build backend |
| `numpy` | `1.26.4` | Numerical arrays, image tensors, metric computation |
| `pandas` | `2.2.1` | Tabular I/O and split manipulation |
| `scikit-learn` | `1.4.2` | Random Forest / Logistic Regression / Linear Regression, `train_test_split`, `ColumnTransformer`, `Pipeline`, scalers, encoders, metrics |
| `mlflow` | `2.11.3` | Experiment tracking, Model Registry, stage transitions, lineage tags, governance runs |
| `protobuf` | `<5` | MLflow serialisation dependency (pinned for compat) |
| `fastapi` | `0.110.0` | Prediction service HTTP layer |
| `uvicorn` | `0.29.0` | ASGI server for FastAPI |
| `evidently` | `0.6.7` | Tabular drift detection (`DataDriftPreset`, KS / chi-square tests) |
| `pyyaml` | `6.0.1` | Parsing every YAML config |
| `Pillow` | `>=10.0,<12` | Image I/O for JPG/PNG pipelines and API request decoding |
| `torch` | `2.4.1` | CNN definition, training, inference |
| `torchvision` | `0.19.1` | Image transforms (used in robustness analysis) |
| `rawpy` | `>=0.18,<1.0` | DNG raw file I/O and camera metadata extraction for the ISP stage |
| `colour-demosaicing` | `>=0.2,<1.0` | Bayer → RGB demosaicing (bilinear, Malvar 2004, Menon 2007) |
| `scikit-image` | `>=0.22,<1.0` | YUV conversion, unsharp mask, resize for the ISP stage |
| `scipy` | `>=1.12,<2.0` | Gaussian / median filter and convolution for ISP steps |
| `python-dateutil` | `>=2.9,<3.0` | Robust timestamp parsing (registry metadata, governance events) |
| *(optional)* `torch-directml` | `>=0.2,<1.0` | Windows AMD/Intel GPU acceleration — installed only via the `[directml]` extra |

**Test framework**: `pytest` (installed alongside the editable package in CI; used in `tests/conftest.py` and every test module). **Containerisation**: Docker with a `python:3.12-slim` base image and `libgomp1` installed for scikit-learn; non-root `appuser`; HTTP healthcheck on `/health`. **Infrastructure as Code**: only Docker Compose is used; there are no Terraform / Pulumi / Helm manifests. **Experiment tracking backend**: the default is a **local file-based** MLflow store at `mlruns/`, but `config.mlflow.tracking_uri` and the `MLFLOW_TRACKING_URI` environment variable can redirect to a remote MLflow server.

**Methods implemented in code (Confirmed):**

- **Content-addressed dataset versioning** via SHA-256 of raw bytes (tabular) or a directory-level hash (images).
- **Deterministic seeded stratified splitting** with scikit-learn, for both tabular CSVs and image folders.
- **Data-contract preprocessing**: `feature_map.json` acts as the single source of truth for feature order and target between training, evaluation, serving, and drift monitoring.
- **Promotion rule engine**: YAML-declared rules with operators `>=`, `<=`, `>`, `<`, `==` (with a `1e-6` tolerance), returning structured violations.
- **Interactive human-in-the-loop approval gate** that reads drift state (and, on fine-tune runs, before/after holdout performance) into the summary and enforces mandatory rejection reasons.
- **Drift-adaptive fine-tuning** — a two-step workflow: `prepare-drift-training` splits a labeled drifted batch into training and holdout sets, evaluates the current Production model on the holdout as a baseline, then copies training images into the raw dataset; `run-pipeline --fine-tune` continues training from Production weights and automatically re-evaluates on the same holdout before the approval gate.
- **Lineage tagging** at the model-version level with hard-fail validation.
- **Drift detection**: Evidently-based KS / chi-square for tabular; per-channel Wasserstein distance and MMD in CNN feature space for images ([src/drift/image_compute.py](src/drift/image_compute.py)).
- **Drift history index** (`history.jsonl`) alongside full JSON snapshots, consumed by the approval summary and the `monitor-summary` CLI.
- **Standalone MLflow sink for runtime drift** so training runs stay immutable and separate from runtime observations.
- **Reproducible ISP pipeline** with black-level correction, demosaicing, white balance, colour correction, denoising, sharpening, and gamma — with an **optional GPU implementation** ([src/data/isp_gpu.py](src/data/isp_gpu.py)) gated by `gpu_accelerated: true`.
- **Unified torch device resolution** ([src/common/device.py](src/common/device.py)) with `MODEL_DEVICE=auto|cpu|cuda|directml` and CUDA → DirectML → CPU fallback.
- **CNN fine-tuning path**: loads Production weights, optionally freezes the backbone, and continues training with a reduced-epoch / lower-LR configuration.
- **Atomic file writes** throughout ([src/common/io.py](src/common/io.py)) to prevent partial-write corruption.
- **Governance MLflow runs** for rollbacks, enabling cross-model auditing via MLflow search.

---

## 7. Architecture and Component Interaction

### 7.1 High-Level Data and Control Flow

```
        data/raw/<dataset>/                 ← user-managed input
              │
              ▼
┌─── run-pipeline [--fine-tune] ─────────────────────────────────────┐
│                                                                    │
│  detect_and_generate ─▶ versioning ─▶ validation ─▶ split          │
│                                                   │                │
│                                                   ▼                │
│   ┌── _STAGE_REGISTRY (preprocessing, training, evaluation,        │
│   │   model_analysis, promotion, deployment) ──┐                   │
│   │                                              ▼                 │
│   │                  execute_stage(name, config, version_id,       │
│   │                                 fine_tune=bool)                │
│   │                                                                │
│   │   (on --fine-tune, before promotion:                           │
│   │     run_drift_adaptation_eval → outputs/drift_adaptation_eval) │
│   └─────────────────────────┬───────────────────┬──────────────────┘
│                             │                   │                  │
│                             ▼                   ▼                  │
│       artifacts/runs/<version_id>/model   mlruns/ (MLflow)         │
│                             │                   │                  │
│                             ▼                   ▼                  │
│                   outputs/*.json        Model Registry             │
└────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
                                       ┌────────────────┐
                                       │  run-api       │
                                       │  (FastAPI)     │
                                       │  :8000         │
                                       │  loads ALL     │
                                       │  Production    │
                                       │  models        │
                                       └────────────────┘
                                                 ▲
                                                 │
                           ┌────────────────────┴────────────────────┐
                           │  monitor-drift / -image / summary       │
                           │  prepare-image-batch                    │
                           │  prepare-drift-training                 │
                           │  rollback-model                         │
                           └─────────────────────────────────────────┘
                                                 │
                                                 ▼
                                       outputs/drift_monitoring/
                                       history.jsonl + *.json
                                       data/evaluation/drifted_holdout/
                                       + standalone MLflow runs
```

### 7.2 Module Interactions

- **Configuration is immutable**: every config loader returns a `@dataclass(frozen=True)` instance. Pipeline stages receive the config by reference; they cannot mutate it. Validation logic lives in [src/config/validation.py](src/config/validation.py). This rules out a class of "stage modified shared state" bugs at the type level.
- **Inputs and outputs are file-based**: each stage writes its output to disk and the next stage reads it. The orchestrator never holds large in-memory state. This enables stage-by-stage re-runs once an external orchestration layer is added, but for now stages must run in-process.
- **MLflow is the system of record** for metrics, parameters, lineage tags, artifacts, model registry, and rollbacks. The filesystem (`artifacts/`, `outputs/`) holds redundant, human-readable copies. Runtime drift runs are deliberately opened as **separate, non-nested** MLflow runs so immutable training provenance is not polluted.
- **The prediction API is strictly read-only** with respect to training state. At startup it calls `load_all_production_models` ([src/deployment/app.py:27](src/deployment/app.py#L27)), which queries the MLflow Registry for every model version at `deploy_config.model.allowed_stage` (default `Production`) and loads each into memory with its lineage metadata, feature map, and — for image models — normalisation stats. Per-model routing is served via `POST /predict/{model_name}`; `GET /models` enumerates loaded models with task-type badges; `GET /` serves a static HTML UI with a predict form; `GET /health` returns 200 when at least one model is loaded and 503 when none are. No training or promotion logic is reachable from the API.

### 7.3 Reproducibility Anchors

Reproducibility is anchored in four independent identifiers that are consistently propagated through the pipeline:

1. **`pipeline_execution_id`** — UUID per run, tagged on the MLflow run.
2. **`config_hash`** — SHA-256 of the pipeline YAML, tagged and used in lineage.
3. **`dataset_version_id`** — content-hash of raw data, used as directory name and MLflow tag.
4. **`random_seed`** — declared in config, propagated to splits, sklearn models, torch RNG (CPU + CUDA), and `DataLoader` shuffle generator.

The combination of (config_hash, dataset_version_id, random_seed) should yield bitwise-identical training outputs on the same platform (**Strong inference** — this is the design; true bit-identity across OS/CPU/GPU/DirectML is not empirically verified in tests, and the device resolver can legitimately dispatch the same run to different hardware on different hosts).

---

## 8. MLOps Practices Present in the Project

This section maps the system onto widely recognised MLOps capability axes.

| Practice | Status | Evidence |
|---|---|---|
| **Version control (code)** | Present | Git repository on GitHub with PR history visible in recent commits. |
| **Version control (data)** | Present | Content-addressed dataset versioning via SHA-256 in [src/data/versioning.py](src/data/versioning.py). |
| **Version control (config)** | Present | Frozen YAML + `config_hash` tag on every MLflow run. |
| **Reproducible environments** | Present | Pinned dependencies in `pyproject.toml`; explicit Python 3.12 constraint; `.venv` workflow in README; Docker image on `python:3.12-slim`. |
| **Deterministic seeding** | Present | `random_seed` drives sklearn and PyTorch RNGs including CUDA and DataLoader. |
| **Modular pipelines** | Present | Stage registry decouples orchestration from implementation. |
| **Automation** | Partial | Pipeline and tests run automatically in CI (on manual dispatch); promotion requires human approval (by design); retraining is manual. |
| **Data validation** | Present | Two-layer validation (structural + content) before any training runs. |
| **Feature contract enforcement** | Present | `feature_map.json` strictly validated by both the training and evaluation stages. |
| **Experiment tracking** | Present | MLflow with tags, params, metrics, and artefacts for every stage. |
| **Model registry** | Present | MLflow Model Registry with `Production` / `Archived` transitions and lineage tags. |
| **Model validation / promotion rules** | Present | Declarative rule engine with distinct `blocked` pipeline outcome and exit code. |
| **Human-in-the-loop governance** | Present | Interactive approval gate with mandatory rejection reason; drift context and (on fine-tune runs) before/after holdout performance surfaced. |
| **Lineage tracking** | Present | Hard-validated `lineage.*` tags on every registered version. |
| **Deployment automation** | Partial | `run-api` / Docker Compose are one command, but deployment is not triggered from the pipeline itself or from CI. |
| **Drift monitoring (tabular)** | Present | Evidently `DataDriftPreset`, severity classification, alert thresholds, interactive decision gate, JSONL history index, standalone MLflow sink. |
| **Drift monitoring (image)** | Present | Wasserstein + MMD-based scoring; optional ISP scenario interpretation. |
| **Drift-adaptive retraining** | Present | `prepare-drift-training` + `run-pipeline --fine-tune` + automatic holdout before/after comparison; still human-initiated, not automated. |
| **Pre-deployment robustness** | Present | ISP sensitivity analysis and augmentation-robustness analysis. |
| **Rollback** | Present | Full `rollback-model` CLI with pre-intent, post-audit, and governance MLflow runs. |
| **Model fine-tuning** | Present | `--fine-tune` flag continues CNN training from the Production model. |
| **GPU acceleration (opt-in)** | Partial | Unified torch-device resolver; ISP GPU path; `[directml]` extra for Windows. Docker image deliberately CPU-only. |
| **Containerisation** | Present | Multi-volume Docker Compose, non-root user, healthchecks, runtime model load. |
| **CI integration** | Partial | Manual-trigger GitHub Actions pipeline with tests, tabular pipeline run, report upload, and Docker build; no lint/typecheck, no push-to-registry, no image-pipeline CI. |
| **Monitoring dashboards** | Absent | No dashboards or metrics endpoints for production observability beyond `/health`. |
| **Automated retraining** | Absent | Triggering retraining remains manual. |

---

## 9. Strengths, Weaknesses, and Gaps

### 9.1 Strengths

1. **Strong configuration discipline.** The use of frozen dataclasses, a dedicated `validation.py`, strict required-key sets, and a unified `load_config()` facade is unusually thorough for a project of this size. It makes the system tolerant to refactoring and resistant to silent config regressions.
2. **Serious attention to reproducibility.** The four-anchor approach (execution id, config hash, dataset version id, random seed) is explicit and consistently propagated into MLflow tags. PyTorch is configured for deterministic CuDNN behaviour.
3. **Audit-first governance.** The rollback CLI's *pre-intent → state-change → post-audit* ordering, the hard-fail on incomplete lineage tags, and the use of a dedicated `{model_name}-governance` MLflow experiment are production-grade patterns that many larger codebases lack.
4. **Clean stage-registry orchestration.** Adding a stage only requires a single dictionary entry, enabling future extension (e.g. a real `deployment` stage) without touching the core orchestrator.
5. **Explicit separation of offline analysis vs online drift detection.** The code and documentation both stress this distinction, preventing a common conceptual error.
6. **Physically motivated image drift scenarios.** The raw-DNG / ISP pipeline enables controlled, repeatable drift experiments that mirror realistic sensor and processing variations, which is a substantive research contribution over purely synthetic augmentation.
7. **Principled separation of training and runtime MLflow runs.** Runtime drift opens its own MLflow run rather than nesting under a training run, so training provenance stays immutable while runtime observations accumulate as time series.
8. **Distinct pipeline outcomes.** `completed`, `failed`, and `blocked` are separate states with separate exit codes (0/1/2) and separate user-facing messaging — a blocked promotion is never confused with a system error.
9. **Atomic I/O helpers** across all write paths prevent partial-write corruption under interruption.
10. **Drift-adaptive fine-tuning loop is tightly integrated.** The orchestrator runs the holdout evaluation automatically on `--fine-tune` runs and surfaces the before/after delta to the approval gate, so the operator does not need a separate manual comparison.
11. **Broad test coverage**: 46 test modules across `common/`, `config/`, `data/`, `deployment/`, `drift/`, `evaluation/`, `monitoring/`, `pipeline/`, `promotion/`, `registry/`, and `training/`, including integration tests for CNN and image pipelines, MLflow round-trip tests, and fine-tuning.

### 9.2 Weaknesses

1. **Evaluation uses the validation split, not the held-out test split.** [src/evaluation/evaluate.py:91-112](src/evaluation/evaluate.py#L91-L112) loads `val.csv` / `val.npz`, which means promotion decisions are made on the same split used during model selection. This can inflate promotion decisions — a meaningful methodological limitation for an academic thesis. Any documentation describing this stage as running "on the held-out test split" does not match the implementation.
2. **CI is manually triggered only.** With `on: workflow_dispatch`, pull requests and pushes do not automatically run tests or build the Docker image, weakening the "CI" in "MLOps-CI-Pipeline". There is also no lint / type-check step and no CI exercise of the image pipelines.
3. **Promotion gate is inherently incompatible with headless automation.** The interactive `input()` call in [src/promotion/approval.py](src/promotion/approval.py) means CI cannot produce promoted models without code changes, and any programmatic driver must mock stdin.
4. **Deployment stage is a no-op placeholder.** The pipeline does not actually deploy — the operator must manually restart the container after promotion. Production model swaps therefore incur downtime and are not themselves audited.
5. **Image drift thresholds are dataset-dependent magic numbers.** The default image-severity thresholds in [src/drift/image_compute.py](src/drift/image_compute.py) are explicitly documented as *"not universal"* but no per-dataset calibration workflow is automated.
6. **Residual `src/training/image_classification/` directory** is completely empty (only `__pycache__`). It is an effective stub that should either be deleted or re-purposed; its presence is confusing.
7. **`src/lifecycle/` is not wired into the pipeline.** The package contains only `states.py` and `transitions.py` and is not imported by the orchestrator, the stage registry, or the registry promotion flow (**Unclear / needs verification** — the intended integration is not evident from the code).
8. **A legacy `src/evaluation/promotion_rules.py` coexists with the authoritative `src/promotion/rules.py`.** This duplication invites confusion; one of the two is dead code.
9. **Some placeholder dataclasses and TODO comments** (e.g. the empty `RegressionEvalConfig` at [src/config/schema.py:331-333](src/config/schema.py#L331-L333)) remain, suggesting unfinished regression-side polish.
10. **No monitoring stack beyond one-shot CLIs.** There is no Prometheus/Grafana, no scheduled drift job, no latency metrics exported by the API.

### 9.3 Risks for Real Production Use

- **Single-host scope.** MLflow tracking URI defaults to a local file path; the Docker Compose setup mounts the local `mlruns/` directory. Multi-node deployment would require a remote MLflow backend (S3 / RDS) not currently configured.
- **Security scope is minimal.** `/predict/{model_name}` accepts arbitrary JSON and base64 images with no authentication, rate limiting, or CORS policy. [docs/threat_model.md](docs/threat_model.md) exists but its completeness is **Unclear / needs verification**.
- **No canary / blue-green deploy.** Promotion archives the current Production version in a single atomic stage transition; the live container continues to serve the previous model until manually restarted.
- **Dependency pinning without a lockfile.** `pyproject.toml` pins direct dependencies but there is no `requirements.lock` / `poetry.lock`, so transitive dependencies may drift across installs.
- **DirectML path is not covered by CI.** The `[directml]` extra is Windows-only and is never exercised on the Linux GitHub runner, so regressions on that path would surface only at runtime on a developer's machine.

---

## 10. Thesis-Ready Summary

The `MLOps-CI-Pipeline` repository implements a cohesive, lightweight but rigorously engineered MLOps system that operationalises three ML task types — tabular classification, tabular regression, and image classification (standard JPG/PNG or raw DNG via a deterministic ISP pipeline) — through a single configurable command-line entry point. Its central engineering contribution is not any single model but rather the *lifecycle* surrounding the model: content-addressed dataset versioning, deterministic splitting and preprocessing, reproducible training (including a Production-aware fine-tuning path), declarative promotion rules with a distinct `blocked` outcome, a human-in-the-loop approval gate that surfaces live drift context and — on fine-tune runs — before/after holdout performance, hard-validated lineage tagging in the MLflow Model Registry, and an auditable rollback procedure with pre-intent and post-audit governance runs.

Architecturally, the system follows a clean layered design (`config → data → training → evaluation → drift/promotion → registry → deployment → monitoring`) with a thin stage-registry orchestrator in `src/pipeline/`. Immutability is enforced at the configuration layer through frozen dataclasses, and reproducibility is anchored in four propagated identifiers (pipeline execution id, config hash, dataset version id, random seed). The prediction service is delivered as a FastAPI / Uvicorn application inside a non-root, health-checked Docker container that enumerates and loads **all** models at the allowed stage (default `Production`) from the MLflow Registry at startup and routes inference per-model via `POST /predict/{model_name}`, ensuring that only governance-approved artefacts are ever served.

A distinctive research dimension is the treatment of *image drift*. The project integrates a deterministic seven-stage Image Signal Processing pipeline (black-level correction → demosaicing → white balance → colour correction → denoising → sharpening → gamma), optionally GPU-accelerated, which is used both for pre-deployment sensitivity analysis and for physically motivated drift scenarios (warm/cool white balance, gamma shifts, noise changes, heavy sharpening). Tabular drift is handled through Evidently's `DataDriftPreset` with statistical tests and configurable severity levels; image drift is measured by per-channel Wasserstein distance and MMD in CNN feature space. Runtime drift results are persisted to a JSONL history index and logged to **separate** MLflow runs so training provenance remains immutable. A dedicated *drift-adaptive fine-tuning* workflow — `prepare-drift-training` followed by `run-pipeline --fine-tune` — closes the loop from drift detection back to model improvement, with an automatic held-out before/after evaluation presented to the approver.

CI/CD is implemented as a manually triggered GitHub Actions workflow that installs the package, runs the pytest suite, executes the tabular pipeline end-to-end, uploads the run report as an artefact, builds the Docker image, and verifies container-level imports. Its scope is deliberately modest: it validates plumbing, not production deployments. Major gaps — interactive-only promotion, missing automatic triggers, no linting or type checking, deployment stage as a placeholder, image pipelines not exercised in CI — leave clearly identified avenues for future work without undermining the core contribution.

Overall, the project demonstrates a well-founded understanding of MLOps principles and implements the vast majority of them in a pragmatic, traceable way. Its principal academic value is as an **integrated, fully reproducible reference implementation** showing how data versioning, experiment tracking, model registry governance, drift monitoring, drift-adaptive retraining, and containerised serving interact in a single coherent codebase, with an original contribution in the form of physically grounded, ISP-driven image drift scenarios.

---

## 11. Evidence Legend

Throughout the document, claims fall into one of three categories:

- **Confirmed from code** — directly visible in a file that was read during the analysis. File paths and line numbers are cited for traceability.
- **Strong inference** — not written explicitly in a single place, but reliably derivable from multiple consistent pieces of evidence (dependency pins, comments, naming conventions, YAML structure, CI behaviour).
- **Unclear / needs verification** — called out where an aspect cannot be determined from the repository alone (e.g. whether `src/lifecycle/` is intended to be wired into a future stage; the completeness of `docs/threat_model.md`; the empirical noise floor for image drift thresholds; bit-identity of reproducible runs across CUDA/DirectML/CPU).

All file and directory references are rendered as relative markdown links so they can be followed directly from this document within the repository.
