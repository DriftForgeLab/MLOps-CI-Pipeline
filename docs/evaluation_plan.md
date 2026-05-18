# Evaluation Plan

This document explains how a trained model is evaluated and how that evaluation feeds
the promotion decision. It describes the behaviour implemented in `src/evaluation/` and
`src/promotion/`; the code is the source of truth where the two disagree.

## Purpose of the evaluation stage

The `evaluation` stage answers one question: **is the newly trained model good enough,
and is it better than the model currently in production?** Its output is consumed by the
`promotion` stage, which turns that answer into a governed decision.

The stage is implemented by `evaluate()` in `src/evaluation/evaluate.py`. It writes
`outputs/evaluation_report.json` and logs the same metrics to MLflow.

## Which data is evaluated

Evaluation runs on the **validation split** (`val.csv` for tabular tasks, `val.npz` for
image tasks). It does **not** use the held-out test split.

This is a deliberate simplification for a teaching project. It has one consequence
worth being clear about: because the validation split is also used during model
selection, metrics measured on it can look better than they would on truly unseen
data. A stricter setup would report the final promotion metrics on the untouched test
split. This limitation is also listed in the project README under "Limitations and
known gaps".

## Metrics

Metrics are computed by `_compute_metrics` and depend on the task type:

| Task type | Metrics |
|-----------|---------|
| `classification` | `accuracy`, `precision`, `recall`, `f1_score` (averaging configurable in `evaluation.yaml`) |
| `regression` | `mae`, `mse`, `rmse`, `r2` |
| `image_classification_cnn` | The classification metrics above, plus a confusion matrix. |

The feature contract (`feature_map.json`) is re-loaded and enforced before any metric is
computed, so evaluation fails fast if the evaluation data does not match the columns the
model was trained on.

## Comparison against the production model

After computing the candidate's metrics, the stage fetches the metrics of the current
`Production` model from the MLflow registry and runs `compare_metrics()` from
`src/promotion/comparator.py`. This produces a per-metric delta and an
`overall_verdict` (for example, "better", "worse", or "mixed").

If no Production model exists yet — the bootstrap case — `no_baseline_comparison()` is
returned instead, and the candidate is evaluated on its absolute metrics alone.

## Promotion gating

The `promotion` stage applies the rule engine in `src/promotion/rules.py` using the
thresholds declared in `src/config/promotion.yaml`. The default gates are:

| Task type | Gate |
|-----------|------|
| Classification | `accuracy >= 0.80` and weighted `f1_score >= 0.75` |
| Regression | `r2 >= 0.80` and `mae <= 0.40` |

The `promotion_evaluation_split` key in `promotion.yaml` selects which split the gates
are checked against (`val`, `test`, or `both`); the default is `val`.

If any rule fails, the run ends with a distinct `blocked` outcome and exit code 2 — this
is treated as a normal governance result, not a system error. If the rules pass, the
interactive approval gate (`src/promotion/approval.py`) presents the metrics, the
comparison verdict, and live drift context for a final human approve/reject decision.

## Pre-deployment robustness analysis (image pipelines)

For image pipelines, the optional `model_analysis` stage performs an offline robustness
analysis *before* deployment. It is **not** part of metric-based promotion gating and
**not** drift detection:

- **Raw DNG pipelines** run an ISP sensitivity analysis: training images are regenerated
  with perturbed ISP parameters (white-balance shifts, noise, gamma changes, sharpening)
  and the resulting accuracy/F1 movement is measured per scenario.
- **Standard JPG/PNG pipelines** run an augmentation-robustness analysis: a fixed set of
  augmentations is applied to the validation split and the accuracy delta is measured.

Both branches quantify how sensitive the model is to plausible transformations before it
reaches production.

## Drift-adaptive holdout evaluation

When a run follows a `prepare-drift-training` (image) or `prepare-drift-training-tabular`
(tabular) preparation step, an extra evaluation runs immediately before the promotion
stage. It evaluates the new model on a held-out set of drifted data and compares it to a
baseline captured for the previous Production model, producing a before/after delta that
is surfaced to the human approver. See [drift-detection.md](drift-detection.md) for the
full workflow.

## Summary

```
train model
   ↓
evaluation stage  →  metrics on the validation split  →  outputs/evaluation_report.json
   ↓
compare against the current Production model
   ↓
promotion stage   →  rule engine (promotion.yaml)  →  pass / blocked
   ↓ (if pass)
interactive approval gate  →  approve / reject
```
