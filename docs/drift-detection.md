# Drift Detection

How tabular data drift is detected, reported, and acted on in the pipeline.

## Overview

The drift system compares a **current** dataset (validation split during training,
or a new batch during monitoring) against a **reference** dataset (the training
split) to detect statistical distribution shifts. It uses
[Evidently](https://docs.evidentlyai.com/) under the hood and produces a
standard result schema that feeds into reporting, MLflow logging, and the
promotion gate.

```
reference (train.csv)  ──┐
                         ├──▶  Evidently  ──▶  severity classification  ──▶  action
current   (val / batch) ─┘
```

## Architecture

```
src/
├── config/
│   ├── schema.py            # DriftConfig + sub-dataclasses
│   ├── drift.yaml           # Default drift thresholds
│   └── drift_loader.py      # load_drift_config()
├── drift/
│   ├── alignment.py         # Reference loading + schema validation
│   ├── compute.py           # Evidently wrapper + result extraction
│   └── interpret.py         # Severity classification + result assembly
├── evaluation/
│   └── drift_tests.py       # Pipeline orchestration (train vs val)
├── monitoring/
│   ├── drift.py             # Batch monitoring entry point
│   ├── monitor_cli.py       # `monitor-drift` CLI command
│   └── reports.py           # Print, save JSON/HTML reports
├── pipeline/
│   ├── steps.py             # _drift_stage() + drift blocking in _promotion_stage()
│   └── mlflow_logger.py     # log_drift_metrics_to_mlflow()
└── promotion/
    └── approval.py          # Drift info in approval summary
```

## Configuration

All drift settings live in `src/config/drift.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Master switch for drift analysis |
| `reference_source` | `"train"` | Which split to use as reference baseline |
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
| `monitoring.min_batch_size` | `30` | Skip analysis if batch has fewer rows |
| `monitoring.alert_severity` | `"medium"` | Log a warning when overall severity reaches this level |

Available statistical tests: `ks`, `chisquare`, `psi`, `wasserstein`, `jensenshannon`.

Promotion blocking is configured separately in `src/config/promotion.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `drift_block_severity` | `"high"` | Block promotion when overall severity is at or above this |

## Severity Classification

### Overall (dataset-level)

Based on `drift_share` = drifted features / total features:

| Condition | Severity |
|-----------|----------|
| drift_share <= `severity.low_max` (0.25) | **low** |
| drift_share <= `severity.medium_max` (0.50) | **medium** |
| drift_share > `severity.medium_max` | **high** |

### Per-feature

Based on the p-value from the statistical test:

| Condition | Severity |
|-----------|----------|
| Not drifted (p-value >= stattest_threshold) | **low** |
| p-value < `feature_severity.high_below` (0.001) | **high** |
| p-value < `feature_severity.medium_below` (0.01) | **medium** |
| Otherwise | **low** |

## Pipeline Integration

The drift stage runs after evaluation and before promotion:

```
preprocessing → training → evaluation → drift → promotion → deployment
```

### What the drift stage does

1. Loads the reference dataset (train split) and feature map
2. Loads the current dataset (validation split)
3. Validates feature alignment (schema compatibility)
4. Runs Evidently `DataDriftPreset` with configured stat tests
5. Classifies overall and per-feature severity
6. Saves `drift_report.json` and `drift_report.html` to `data/drift_scenarios/`
7. Logs drift metrics and tags to MLflow
8. Writes `drift_result.json` to `outputs/` for the promotion stage

### Promotion gate

The promotion stage reads `outputs/drift_result.json` and checks the overall
severity against `drift_block_severity`. If severity is at or above the
threshold, a `DRIFT_SEVERITY` violation is added, which blocks promotion
unless the user explicitly overrides it in the approval gate.

The approval summary displays drift status including whether dataset drift was
detected, the overall severity, and which features drifted.

### MLflow tracking

The drift stage logs the following to the active MLflow run:

**Metrics** (one per feature):
- `drift.<feature_name>.score` — the p-value from the statistical test

**Tags:**
- `drift.overall_severity` — low / medium / high
- `drift.dataset_drift_detected` — true / false
- `drift.drifted_feature_count` — integer count
- `drift.drift_share` — float (0.0 - 1.0)

**Artifacts** (best-effort):
- `drift_report.html` — Evidently interactive dashboard
- `drift_report.json` — standard drift result

## Offline Batch Monitoring

The `monitor-drift` CLI command runs drift analysis on a new data batch against
a model's reference data, outside of the training pipeline.

### Usage

```bash
monitor-drift \
  --batch-csv data/new_batch.csv \
  --model-name iris_rf \
  --dataset-name iris \
  --dataset-version <version-hash> \
  --config src/config/pipeline_tabular.yaml
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

1. Loads pipeline and drift configs
2. Loads and validates the batch CSV
3. Loads the reference data for the specified model/dataset
4. Runs `monitor_batch()` — same drift engine as the pipeline stage
5. If the batch is below `min_batch_size`, exits cleanly with no output
6. Prints a drift summary to stdout
7. Saves results as a timestamped JSON file: `<output-dir>/<YYYYMMDDTHHMMSSZ>.json`
8. Logs a warning if overall severity reaches `alert_severity` or above

## Drift Result Schema (v1.0.0)

Every drift analysis (pipeline or monitoring) produces a JSON result with this
structure:

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

## Disabling Drift

Set `enabled: false` in `drift.yaml` to skip the drift stage entirely. The
pipeline will log a message and continue to the promotion stage without drift
checks.
