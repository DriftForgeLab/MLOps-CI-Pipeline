# Image Dataset Support

The pipeline supports CNN-based image classification alongside tabular datasets.
Image datasets flow through the same pipeline entry point (`run-pipeline`) with
`task_type: image_classification_cnn` — no separate CLI or workflow is needed.

## Folder Structure

Image datasets use the **ImageFolder** convention:

```
data/raw/<dataset_name>/
  dataset.yaml
  images/
    class_a/
      img001.jpg
      img002.png
    class_b/
      img003.jpg
```

Each subdirectory under `images/` is a class label. All images in that
directory are assigned that label.

## dataset.yaml

```yaml
name: my_images
task_type: image_classification_cnn
description: "My image classification dataset"
source: "custom"
created_at: "2026-03-18"
features: []          # no tabular features
target: label
schema: {}            # no column-level schema
image_properties:
  expected_formats: [".jpg", ".png"]
  min_images_per_class: 5
constraints:
  min_rows: 10
  max_null_fraction: 0.0
  label_classes: [class_a, class_b]
```

Key differences from tabular datasets:
- `features: []` and `schema: {}` — images have no tabular columns.
- `image_properties` — declares expected image formats and minimum class sizes.
- `task_type: image_classification_cnn` — drives dispatch throughout the pipeline.

If `dataset.yaml` is missing, the pipeline will auto-detect the `images/` folder
and prompt you interactively to generate it.

## Preprocessing Configuration

Two preprocessing configs are available depending on image format:

**Standard JPG/PNG** — `src/config/preprocessing_image.yaml`:
```yaml
image:
  target_size: [64, 64]
  color_mode: "rgb"
  normalize: true
  flatten: false        # preserve spatial structure for CNN
  augmentation:
    enabled: false
    horizontal_flip: false
    rotation_degrees: 0
    augmentation_factor: 1
```

**Raw DNG (ISP pipeline)** — `src/config/preprocessing_raw_image.yaml`:
```yaml
image:
  target_size: [64, 64]
  color_mode: "rgb"
  normalize: true
  flatten: false
  raw_input: true       # triggers ISP pipeline: DNG → demosaic → WB → CCM → gamma → CNN
  isp:
    demosaicing:
      algorithm: "bilinear"   # bilinear | malvar2004 | menon2007
    gamma_correction:
      gamma: 2.2
    # ... (see full config for all ISP options)
```

Normalization statistics (mean, std) are computed **only from training images**
and applied to all splits (leak-proof).

## Pipeline Configuration

```yaml
# src/config/pipeline_image_cnn.yaml
task_type: "image_classification_cnn"
dataset: sample_images    # change to your dataset name under data/raw/
configs:
  preprocessing: "src/config/preprocessing_image.yaml"
  training: "src/config/training_image_cnn.yaml"
  # ... other configs unchanged
```

For raw DNG datasets, use `pipeline_image_raw.yaml` instead, which points to
`preprocessing_raw_image.yaml`.

## Architecture Configuration

```yaml
# src/config/training_image_cnn.yaml
model:
  algorithm: "cnn"
  architecture:
    conv_layers:
      - out_channels: 32
        kernel_size: 3
      - out_channels: 64
        kernel_size: 3
    fc_units: 128
    dropout: 0.3
  hyperparameters:
    epochs: 10
    batch_size: 32
    learning_rate: 0.001
```

## Running

```bash
# Standard JPG/PNG dataset
run-pipeline --config src/config/pipeline_image_cnn.yaml

# Raw DNG dataset (ISP pipeline)
run-pipeline --config src/config/pipeline_image_raw.yaml
```

The pipeline executes: versioning → splitting → preprocessing → training →
evaluation → promotion, identical to tabular datasets.

## Data Flow

```
data/raw/<dataset>/images/{class}/...
        ↓  versioning (manifest hash)
data/processed/<dataset>/<version_id>/images/{class}/...
        ↓  stratified splitting
data/processed/<dataset>/<version_id>/train/images/{class}/...
                                      val/images/{class}/...
                                      test/images/{class}/...
        ↓  preprocessing (resize, normalize)
data/processed/<dataset>/<version_id>/preprocessed/
    train.npz  val.npz  test.npz
    feature_map.json  pipeline.pkl  metadata.json
```

### Output Artifacts

- **`{split}.npz`** — Compressed numpy archives with `X` (features, shape `N×H×W×C`) and `y` (integer labels).
- **`feature_map.json`** — Class mapping, normalization stats, image shape, feature count.
- **`pipeline.pkl`** — Serialized normalization parameters for inference.

## Augmentation

Offline augmentation expands the training set before model fitting:

```yaml
augmentation:
  enabled: true
  horizontal_flip: true
  rotation_degrees: 90        # 0/90/180/270 degree rotations
  augmentation_factor: 3      # 3x training set size
```

Augmentation is applied **only to training data**, never to validation or test.
The augmented dataset is deterministic given the same seed and config.

## Limitations

- All images are loaded into memory as numpy arrays. Datasets larger than
  ~10,000 images at 64×64 may require significant RAM.
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`, `.dng`.
  Unreadable images are skipped with a warning.
- Raw DNG processing requires `rawpy` and `colour-demosaicing` to be installed.
