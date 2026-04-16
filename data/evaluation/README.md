# Evaluation Data

This directory stores evaluation artifacts produced by the pipeline.

## Drift Adaptation Holdout (`drifted_holdout/`)

Created by `prepare-drift-training`. Contains:

```
data/evaluation/drifted_holdout/<dataset_name>/
    <class_a>/              ← held-out drifted images (ImageFolder layout)
    <class_b>/
    baseline_metrics.json   ← Production model accuracy on holdout before fine-tuning
    normalization_stats.json ← Training normalization stats for consistent evaluation
```

The holdout is evaluated automatically at the end of every `run-pipeline --fine-tune`
run. Results are printed as a before/after comparison table and saved to
`outputs/drift_adaptation_eval.json`.

These files are not tracked in git (see `.gitignore`).
