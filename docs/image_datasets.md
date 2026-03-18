# Image Dataset Support

The pipeline supports image classification datasets alongside tabular datasets.
Image datasets flow through the same pipeline entry point (`run-pipeline`) with
`task_type: image_classification` — no separate CLI or workflow is needed.

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
task_type: image_classification
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
- `task_type: image_classification` — drives dispatch throughout the pipeline.

If `dataset.yaml` is missing, the pipeline will auto-detect the `images/` folder
and prompt you interactively to generate it.

## Preprocessing Configuration

Create a preprocessing config with an `image` section:

```yaml
# src/config/preprocessing_image.yaml
image:
  target_size: [64, 64]       # (height, width) resize
  color_mode: "rgb"           # "rgb" or "grayscale"
  normalize: true             # [0,1] scaling + channel-wise normalization
  flatten: true               # flatten to 1D for sklearn
  augmentation:
    enabled: false
    horizontal_flip: false
    rotation_degrees: 0
    augmentation_factor: 1    # 1 = no augmentation, 2 = double the training set
```

Normalization statistics (mean, std) are computed **only from training images**
and applied to all splits (leak-proof).

## Pipeline Configuration

```yaml
# src/config/pipeline_image_classification.yaml
task_type: "image_classification"
dataset: my_images
configs:
  preprocessing: "src/config/preprocessing_image.yaml"
  training: "src/config/training_classification.yaml"
  # ... other configs unchanged
```

The same training configs work for image classification — sklearn models
(RandomForest, LogisticRegression) receive flattened pixel vectors.

## Running

```bash
run-pipeline --config src/config/pipeline_image_classification.yaml
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
        ↓  preprocessing (resize, normalize, flatten)
data/processed/<dataset>/<version_id>/preprocessed/
    train.npz  val.npz  test.npz
    feature_map.json  pipeline.pkl  metadata.json
```

### Output Artifacts

- **`{split}.npz`** — Compressed numpy archives with `X` (features) and `y` (integer labels).
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

- Images are flattened to 1D vectors for sklearn classifiers. This works for
  small images but does not capture spatial features. For larger images or
  better accuracy, a CNN-based approach (future task_type) is recommended.
- All images are loaded into memory as numpy arrays. Datasets larger than
  ~10,000 images at 64x64 may require significant RAM.
- Only `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, and `.gif` formats are supported.
  Unreadable images are skipped with a warning.
