# Raw Data

This directory contains raw datasets used as input to the pipeline.
Each dataset lives in its own subdirectory. `dataset.yaml` files are tracked in git;
image files are excluded (see below).

## Datasets

### california_housing (`data/raw/california_housing/`)
Tabular regression dataset. `data.csv` is tracked in git.

- **Task:** Regression — predict median house value (`MedHouseVal`) in $100k units
- **Source:** `sklearn.datasets.fetch_california_housing()` (Pace & Barry, 1997)
- **Rows:** 20,640 | **Features:** 8 numeric (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude)
- **No missing values, no categorical features** — preprocessing is column-selection + optional scaling only
- **Pipeline config:** `src/config/pipeline_tabular_regression.yaml`
- **Expected RF performance (val split):** R² ≈ 0.81, MAE ≈ 0.38, RMSE ≈ 0.52
- **Promotion gates:** R² ≥ 0.75 and MAE ≤ 0.55 (configured in `src/config/promotion.yaml`)

To regenerate `data.csv` from scratch:
```python
from sklearn.datasets import fetch_california_housing
import pandas as pd
fetch_california_housing(as_frame=True).frame.to_csv("data/raw/california_housing/data.csv", index=False)
```

### iris (`data/raw/iris/`)
Tabular classification dataset. `data.csv` is tracked in git.

### sample_images (`data/raw/sample_images/`)
Small JPG image dataset for testing the image classification pipeline.
Images are tracked in git (small files).

### cifar10 (`data/raw/cifar10/`)
Imported CIFAR-10 dataset converted into the pipeline's ImageFolder PNG layout.

- Source: https://www.cs.toronto.edu/~kriz/cifar.html
- Import command: `python scripts/import_cifar10.py`
- Pipeline config: `src/config/pipeline_cifar10.yaml`

The generated PNG files are not tracked in git. The importer writes
`data/raw/cifar10/images/<class>/*.png` plus `data/raw/cifar10/dataset.yaml`.

### drone_raw (`data/raw/drone_raw/`)
Raw DNG aerial drone images used to demonstrate the ISP preprocessing pipeline.

- Code reference: [raw2logit](https://github.com/aiaudit-org/raw2logit)
- Image source: [Zenodo record 5235536](https://zenodo.org/records/5235536)

**Images are NOT tracked in git** because raw DNG files are large.

This dataset is included as a **small reproducible subset** for verifying that the raw image pipeline works end-to-end:
raw DNG → ISP preprocessing → train/val/test split → CNN pipeline.

The 12-image subset is **not intended for meaningful model training, robust evaluation, or realistic drift analysis**. It is only intended as a lightweight technical demonstration. For actual use, provide a larger raw dataset using the same folder structure described below.

**Folder structure for any raw DNG dataset:**
```
data/raw/<dataset_name>/images/
    <class_a>/    ← all DNG images for class A
    <class_b>/    ← all DNG images for class B
```
Subdirectory names become the class labels. Minimum 5 images per class is required.

**To reproduce the demo subset (12 images from Zenodo):**
1. Download from [Zenodo record 5235536](https://zenodo.org/records/5235536):
   - scene_a (6 files): `DJI_0001.DNG`, `DJI_0003.DNG`, `DJI_0004.DNG`, `DJI_0005.DNG`, `DJI_0009.DNG`, `DJI_0010.DNG`
   - scene_b (6 files): `DJI_0011.DNG`, `DJI_0013.DNG`, `DJI_0020.DNG`, `DJI_0025.DNG`, `DJI_0027.DNG`, `DJI_0029.DNG`
2. Place each group in the corresponding subfolder:
```
data/raw/drone_raw/images/
    scene_a/    ← 6 files from step 1 (scene_a)
    scene_b/    ← 6 files from step 1 (scene_b)
```
3. Run: `run-pipeline --config src/config/pipeline_image_raw.yaml`

Note: Larger raw datasets are supported, but they require more storage, preprocessing time, and compute. In practice, large raw-image runs are better suited to local execution than routine execution in GitHub Actions.

## Directory Structure

**Tabular datasets:**
```
data/raw/<dataset_name>/
    dataset.yaml   ← tracked in git
    data.csv       ← tracked in git (small files only)
```

**Image datasets:**
```
data/raw/<dataset_name>/
    dataset.yaml          ← tracked in git
    images/
        <class_name>/
            image1.jpg    ← Not tracket
```

## Adding a New Tabular Dataset

1. Create `data/raw/<name>/data.csv`
2. Run the pipeline — it detects missing metadata and prompts for target column and task type:
```bash
run-pipeline --config src/config/pipeline_tabular_classification.yaml
```

## Adding a New Image Dataset

1. Create `data/raw/<name>/images/<class>/` and place images inside
2. Create `data/raw/<name>/dataset.yaml` — see existing files as templates
3. Create `src/config/pipeline_<name>.yaml` pointing to the correct preprocessing config
4. Run the pipeline:
```bash
run-pipeline --config src/config/pipeline_<name>.yaml
```

For raw DNG images that require ISP preprocessing, use `preprocessing_raw_image.yaml` as the
preprocessing config. For standard JPG/PNG images, use `preprocessing_image.yaml`.
