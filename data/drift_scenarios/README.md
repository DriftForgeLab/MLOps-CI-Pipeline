# Drift Scenarios

This directory holds the output of the `model_analysis` stage, which runs only
for image pipelines. The directory name is a bit misleading — it is kept for
historical reasons. What lands here is offline robustness analysis, not drift
detection on real production data.

Outputs are written per dataset and version:

```
data/drift_scenarios/<dataset>/<version_id>/
```

## Generated files

| File | Pipeline | Description |
|------|----------|-------------|
| `sensitivity_report.json` / `.html` | Raw DNG (ISP) | ISP sensitivity analysis — how metrics move when ISP parameters are perturbed |
| `robustness_report.json` / `.html`  | Standard JPG/PNG | Augmentation robustness — how metrics move under image augmentations |

For raw DNG pipelines, the ISP simulation step also writes the perturbed
scenario images into this directory before the sensitivity analysis reads them.

Files are overwritten when the same dataset version is re-analysed.

## What this is not

This is not online drift detection. Comparing the training data against new
production batches is done separately by the `monitor-drift` and
`monitor-drift-image` CLIs, which write timestamped JSON to
`outputs/drift_monitoring/<model-name>/`.

The reports here are also logged as MLflow artifacts on the pipeline run. See
[docs/drift-detection.md](../../docs/drift-detection.md) for the full picture.
