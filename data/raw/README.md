# Raw Data

This directory contains raw datasets used as input to the pipeline.

## Directory Structure

Each dataset is stored in its own subdirectory:
```
data/raw/<dataset_name>/
    dataset.yaml
    data.csv
```

## dataset.yaml Schema

| Field         | Type   | Required | Description                                              |
|---------------|--------|----------|----------------------------------------------------------|
| name          | string | yes      | Unique dataset identifier                                |
| task_type     | string | yes      | `classification` or `regression`                         |
| description   | string | yes      | Human-readable description                               |
| source        | string | yes      | Origin of the dataset                                    |
| created_at    | string | yes      | Creation date (YYYY-MM-DD)                               |
| features      | list   | yes      | List of input feature column names                       |
| target        | string | yes      | Target column name                                       |
| schema        | map    | yes      | Column names with expected dtype (`float`, `int`, `string`, `bool`) |
| constraints   | map    | yes      | Data contract rules enforced on every pipeline run       |

### `constraints` sub-fields

| Field              | Type   | Default | Description                                                  |
|--------------------|--------|---------|--------------------------------------------------------------|
| min_rows           | int    | 10      | Minimum rows required in each split                          |
| max_null_fraction  | float  | 0.0     | Maximum allowed null fraction per column (0.0 = no nulls)   |
| label_classes      | list   | —       | Valid label values (classification only); auto-detected from data |

The `constraints` block is **auto-generated** when `dataset.yaml` is created interactively.
It does not need to be written by hand.


## Adding a New Dataset

1. Create a new directory under `data/raw/<dataset_name>/`
2. Place your dataset file as `data.csv` inside that directory
3. Run the pipeline — it will automatically detect the missing metadata and prompt you to fill in the required information:

```
   run-pipeline --config src/config/pipeline.yaml
```