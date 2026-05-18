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
- **Promotion gates:** R² ≥ 0.80 and MAE ≤ 0.40 (configured in `src/config/promotion.yaml`)

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
- Pipeline config: `src/config/pipeline_cifar10.yaml`

CIFAR-10 is consumed in the standard ImageFolder layout
(`data/raw/cifar10/images/<class>/*.png` plus `data/raw/cifar10/dataset.yaml`).
The conversion from the original CIFAR-10 archive into this layout was performed
with a separate import script that is **not included in this repository**. To
use this dataset, recreate that layout yourself — for example by downloading
CIFAR-10 via `torchvision` and writing one PNG per image into the per-class
folders. The generated PNG files are not tracked in git.

### fivek (`data/raw/fivek/`)
Raw DNG photographs from the MIT-Adobe FiveK dataset, used to demonstrate the
ISP preprocessing pipeline on raw images.

- Image source: [MIT-Adobe FiveK Dataset](https://data.csail.mit.edu/graphics/fivek/)
- Pipeline config: `src/config/pipeline_fivek.yaml`

**Images are NOT tracked in git** because raw DNG files are large.

The pipeline consumes FiveK as a raw-image classification dataset, end-to-end:
raw DNG → ISP preprocessing → train/val/test split → CNN pipeline. FiveK is
published as an image-enhancement dataset rather than a labelled classification
dataset, so the class labels come from how you sort the photos into subfolders.

**Folder structure for any raw DNG dataset:**
```
data/raw/<dataset_name>/images/
    <class_a>/    ← all DNG images for class A
    <class_b>/    ← all DNG images for class B
```
Subdirectory names become the class labels. Minimum 5 images per class is required.

**To reproduce:**
1. Download the raw DNG files from the
   [MIT-Adobe FiveK Dataset](https://data.csail.mit.edu/graphics/fivek/).
2. Sort a subset of the photos into per-class subfolders under
   `data/raw/fivek/images/`:
```
data/raw/fivek/images/
    <class_a>/    ← DNG images for class A
    <class_b>/    ← DNG images for class B
```
3. Run: `run-pipeline --config src/config/pipeline_fivek.yaml`

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
            image1.jpg    ← not tracked in git
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
