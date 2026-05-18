# Drift Detection

How tabular and image drift are detected, reported, and (today) acted on in this
project.

## Architectural Position

**Drift detection is intentionally external to the training pipeline.** A
training run produces a model + reference artifacts (the train split, feature
map, image statistics). Drift is computed afterwards, by separate CLIs run on
new batches, against those reference artifacts.

```
training pipeline ──▶ model + reference artifacts ──┐
                                                    ▼
                                       monitor CLI on new batch ──▶ drift report
```

The pipeline does **not** run a drift stage.

## Current Capability

What works today:

- `monitor-drift` (tabular) and `monitor-drift-image` (image) compute drift
  against per-model reference data and write a timestamped JSON report.
- Per-feature and overall severity classification, config-driven for both
  tabular (`drift.yaml`) and image (`drift.image.severity`) tasks.
- Optional interactive decision prompt when run in a TTY.
- In a non-interactive (CI/cron) run the CLI exits with code 2 when observed
  severity reaches `monitoring.fail_on_severity` — a usable CI gate.
- Every run logs the result to a separate, non-nested MLflow run and appends
  an entry to a `history.jsonl` drift-history index.
- The promotion stage reads the latest drift result (`load_latest_drift`) and
  passes it to the approval summary. The monitoring drift block is rendered as
  a fallback so the reviewer always sees a drift signal; it is suppressed only
  when a drift-adaptation eval block is shown (post-fine-tuning the raw drift
  label would mislead). When no drift history exists an explicit
  "no drift data" banner is rendered instead.

## Source Layout

```
src/
├── config/
│   ├── schema.py            # DriftConfig + sub-dataclasses
│   ├── drift.yaml           # Default thresholds + monitoring knobs
│   └── drift_loader.py      # load_drift_config()
├── drift/
│   ├── alignment.py         # Reference loading + schema validation
│   ├── compute.py           # Evidently wrapper for tabular drift
│   ├── image_compute.py     # Image statistical / embedding drift
│   └── interpret.py         # Severity classification + result assembly
├── monitoring/
│   ├── drift.py             # monitor_batch() — tabular entry point
│   ├── monitor_cli.py       # `monitor-drift` CLI
│   ├── image_drift_monitor.py  # monitor_image_batch()
│   ├── image_monitor_cli.py # `monitor-drift-image` CLI
│   ├── drift_decision.py    # Interactive decision prompt + CI exit gate
│   ├── history.py           # JSONL drift-history index
│   ├── mlflow_sink.py       # Runtime-drift MLflow sink
│   ├── monitor_summary_cli.py  # `monitor-summary` CLI
│   └── reports.py           # JSON / HTML writers
├── pipeline/
│   └── mlflow_logger.py     # log_drift_metrics_to_mlflow() — usable from
│                            # within a pipeline run only
└── promotion/
    └── approval.py          # Renders the drift block in the approval summary
```

## Tabular Drift Configuration

All tabular settings live in `src/config/drift.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Master switch (currently only consulted by monitor CLIs) |
| `reference_source` | `"train"` | Which split is the reference baseline |
| `stattest.numerical` | `"ks"` | Statistical test for numerical features |
| `stattest.categorical` | `"chisquare"` | Statistical test for categorical features |
| `stattest_threshold.numerical` | `0.05` | P-value threshold for numerical features |
| `stattest_threshold.categorical` | `0.05` | P-value threshold for categorical features |
| `drift_share` | `0.5` | Fraction of features that must drift for dataset-level drift |
| `severity.low_max` | `0.25` | drift_share at or below this is "low" severity |
| `severity.medium_max` | `0.50` | drift_share at or below this is "medium" severity |
| `feature_severity.high_below` | `0.001` | P-value below this is "high" per-feature severity |
| `feature_severity.medium_below` | `0.01` | P-value below this is "medium" per-feature severity |
| `monitoring.enabled` | `true` | Enable batch monitoring |
| `monitoring.min_batch_size` | `10` | Skip analysis if batch has fewer rows |
| `monitoring.alert_severity` | `"medium"` | Log a warning when overall severity reaches this level |
| `monitoring.fail_on_severity` | `"high"` | In non-interactive runs, exit with code 2 when severity reaches this level (`never` disables the gate) |

Available statistical tests: `ks`, `chisquare`, `psi`, `wasserstein`, `jensenshannon`.

## Severity Classification

### Overall (dataset-level)

Based on `drift_share` = drifted features / total features:

| Condition | Severity |
|-----------|----------|
| drift_share <= `severity.low_max` (0.25) | **low** |
| drift_share <= `severity.medium_max` (0.50) | **medium** |
| drift_share > `severity.medium_max` | **high** |

### Per-feature (tabular)

Based on the p-value from the statistical test:

| Condition | Severity |
|-----------|----------|
| Not drifted (p-value >= stattest_threshold) | **low** |
| p-value < `feature_severity.high_below` (0.001) | **high** |
| p-value < `feature_severity.medium_below` (0.01) | **medium** |
| Otherwise | **low** |

### Image severity

Image overall severity thresholds are configured under `drift.image.severity`
in `drift.yaml`:

```yaml
image:
  severity:
    medium: 0.10
    high: 0.25
```

Per-channel Wasserstein distances are aggregated; an overall score at or above
`high` is "high", at or above `medium` is "medium", else "low". When a caller
supplies no thresholds, `DEFAULT_IMAGE_SEVERITY_THRESHOLDS` in
`src/drift/image_compute.py` provides the same values as a fallback default.

## Tabular Batch Monitoring

The `monitor-drift` CLI runs drift analysis on a new data batch against a
model's reference data.

### Usage

```bash
monitor-drift \
  --batch-csv data/new_batch.csv \
  --model-name iris_rf \
  --dataset-name iris \
  --dataset-version <version-hash> \
  --config src/config/pipeline_tabular_classification.yaml
```

### Options

| Flag | Required | Description |
|------|----------|-------------|
| `--batch-csv` | yes | Path to the new data batch CSV |
| `--model-name` | yes | Model name for logging context |
| `--dataset-name` | yes | Dataset name (e.g., `iris`) |
| `--dataset-version` | yes | Dataset version ID (content-addressed hash) |
| `--config` | yes | Path to pipeline config file |
| `--drift-config` | no | Override drift config (defaults to pipeline config's drift path) |
| `--output-dir` | no | Output directory (defaults to `outputs/drift_monitoring/<model-name>/`) |

### Behavior

1. Loads pipeline and drift configs.
2. Loads and validates the batch CSV.
3. Loads the reference data for the specified model/dataset.
4. Runs `monitor_batch()` — Evidently `DataDriftPreset` with configured stat tests.
5. If the batch is below `min_batch_size`, exits cleanly with no output.
6. Prints a drift summary to stdout.
7. Saves a timestamped JSON file: `<output-dir>/<YYYYMMDDTHHMMSSZ>.json`.
8. Logs the result to an MLflow runtime-drift run and appends a `history.jsonl` entry.
9. Logs a warning if overall severity reaches `alert_severity` or above.
10. In an interactive shell only, prompts for a drift decision.

In a non-interactive (non-TTY) run the CLI exits with code 2 when overall
severity reaches `monitoring.fail_on_severity`; otherwise it exits 0.

## Image Batch Monitoring

The `monitor-drift-image` CLI computes image drift against a per-model
reference (channel statistics for `statistical` mode, CNN embeddings for
`embedding` mode), and optionally matches the observed perturbation against
ISP scenarios generated at training time.

```bash
monitor-drift-image \
  --batch-dir data/new_batch_images/ \
  --model-name iris_image_cnn \
  --config src/config/pipeline_image_cnn.yaml \
  --method statistical \
  --drift-scenarios-dir data/drift_scenarios/
```

Same exit-code behaviour as the tabular CLI: in a non-interactive run it exits
with code 2 when severity reaches `monitoring.fail_on_severity`, else 0.

## Drift Result Schema (v1.0.0)

Both tabular and image monitors produce a JSON result with this shape:

```json
{
  "schema_version": "1.0.0",
  "drift_type": "tabular",
  "generated_at": "2026-04-06T12:00:00Z",
  "pipeline_execution_id": "<version_id>",
  "dataset_version_id": "<version_id>",
  "task_type": "classification",

  "reference_dataset": {
    "source": "train",
    "row_count": 120,
    "feature_count": 4
  },
  "current_dataset": {
    "source": "val",
    "row_count": 30,
    "feature_count": 4
  },

  "overall": {
    "dataset_drift_detected": false,
    "drift_share": 0.25,
    "drifted_feature_count": 1,
    "total_feature_count": 4,
    "severity": "low"
  },

  "features": {
    "sepal_length": {
      "column_type": "numerical",
      "drift_detected": true,
      "drift_score": 0.003,
      "stattest_name": "ks",
      "stattest_threshold": 0.05,
      "severity": "medium"
    }
  },

  "artifacts": {},

  "config_snapshot": {
    "stattest_numerical": "ks",
    "stattest_categorical": "chisquare",
    "stattest_threshold_numerical": 0.05,
    "stattest_threshold_categorical": 0.05,
    "drift_share_threshold": 0.5,
    "severity_low_max": 0.25,
    "severity_medium_max": 0.50
  }
}
```

`task_type` is taken from the pipeline config's `task_type` and recorded on
every result. It falls back to `"unknown"` only when `monitor_batch()` is
called directly without the argument being passed.

## ISP Sensitivity vs. Drift (Disambiguation)

The training pipeline runs an offline **ISP sensitivity / augmentation
robustness** analysis, not drift detection. These reports live under
`data/drift_scenarios/` (legacy directory name) and are logged to MLflow under
the `analysis.*` tag namespace via `_run_isp_simulation_and_sensitivity` in
`src/pipeline/steps.py`. The `log_isp_scenario_artifacts()` helper in
`src/pipeline/mlflow_logger.py` records the presence of these reports on the
training run.

These analyses tell you how *sensitive* a model is to controlled perturbations,
not whether real incoming data has drifted. Real drift detection happens only
in the monitor CLIs above.

## Disabling Drift Monitoring

Set `enabled: false` (or `monitoring.enabled: false`) in `drift.yaml` and the
monitor CLIs will short-circuit. The training pipeline is unaffected — it
never runs drift in the first place.

## Responding to Drift: Fine-Tuning the CNN Model

When `monitor-drift-image` detects drift and you choose **retrain** from the
decision menu, the recommended follow-up for CNN image pipelines is to
fine-tune the existing Production model rather than training from scratch.

### Why fine-tune instead of retrain from scratch?

Full retraining discards all previously learned features. Fine-tuning starts
from the Production model weights and continues training with fewer epochs and
a lower learning rate, adapting the model to the new data distribution while
preserving what it already knows. This is particularly effective when the drift
is moderate (e.g. lighting shift, slight colour change) rather than a
completely new data source.

Fine-tuning has no effect on tabular (random forest) pipelines — random forests
do not have reusable weights. Only CNN pipelines benefit.

### How to fine-tune after detecting drift

```bash
# 1. Add or update images in data/raw/<dataset>/images/ to include new data

# 2. Run the pipeline with --fine-tune
run-pipeline --config src/config/pipeline_image_cnn.yaml --fine-tune

# 3. The pipeline will:
#    - Preprocess the updated dataset
#    - Load the current Production model weights from MLflow Registry
#    - Continue training with fine-tune hyperparameters (fewer epochs, lower lr)
#    - Evaluate, compare against Production, and prompt for promotion
```

If no Production model exists (first run), `--fine-tune` falls back to
training from scratch with a warning.

### Fine-tune hyperparameters

Configured in `src/config/training_image_cnn.yaml` under the `fine_tune:` block:

| Parameter         | Default  | Description                                         |
|-------------------|----------|-----------------------------------------------------|
| `epochs`          | `20`     | Additional training epochs (fewer than full training)|
| `learning_rate`   | `0.0001` | Lower LR to avoid overwriting learned features      |
| `freeze_backbone` | `false`  | If `true`, only the final classifier head is trained|

### MLflow traceability

Fine-tune runs are tagged `pipeline.fine_tune = true` in MLflow and the
`TrainingResult` hyperparameters include `fine_tuned`, `fine_tune_epochs`, and
`fine_tune_lr` for full audit traceability.
