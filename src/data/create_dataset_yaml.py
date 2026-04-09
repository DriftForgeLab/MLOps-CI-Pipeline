# =============================================================================
# src/data/create_dataset_yaml.py — Dataset metadata generator
# =============================================================================
# Automatically detects datasets in data/raw/ that are missing dataset.yaml
# and prompts the user interactively to generate the missing metadata.
# =============================================================================

from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from src.common.io import atomic_write_text


def _ask(prompt: str, valid: dict) -> str:
    hint = ", ".join(f"{k} = {v}" for k, v in valid.items())
    while True:
        raw = input(f"{prompt} [{hint}]: ").strip().upper()
        if raw in valid:
            return valid[raw]
        print(f"  Invalid input. Please enter one of: {hint}")


def _ask_text(prompt: str, default: str = "") -> str:
    hint = f" (default: {default})" if default else ""
    raw = input(f"{prompt}{hint}: ").strip()
    return raw if raw else default


def _generate_image_dataset_yaml(dataset_dir: Path) -> None:
    """Generate dataset.yaml for an image dataset with ImageFolder structure."""
    images_dir = dataset_dir / "images"
    class_dirs = sorted(d for d in images_dir.iterdir() if d.is_dir())
    class_names = [d.name for d in class_dirs]

    print(f"\nImage dataset '{dataset_dir.name}' is missing dataset.yaml.")
    print(f"  Detected classes: {', '.join(class_names)}")
    print(f"  Task type: image_classification_cnn (auto-detected from images/ folder)\n")

    source = _ask_text("  Source/origin of dataset", default="unknown")
    description = _ask_text("  Short description", default="")

    # Detect image formats present
    image_extensions = set()
    for class_dir in class_dirs:
        for f in class_dir.iterdir():
            if f.is_file():
                image_extensions.add(f.suffix.lower())

    metadata = {
        "name": dataset_dir.name,
        "task_type": "image_classification_cnn",
        "description": description,
        "source": source,
        "created_at": str(date.today()),
        "features": [],
        "target": "label",
        "schema": {},
        "image_properties": {
            "expected_formats": sorted(image_extensions),
            "min_images_per_class": 5,
        },
        "constraints": {
            "min_rows": 10,
            "max_null_fraction": 0.0,
            "label_classes": class_names,
        },
    }

    output_path = dataset_dir / "dataset.yaml"
    atomic_write_text(output_path, yaml.dump(metadata, default_flow_style=False, sort_keys=False))

    print(f"  dataset.yaml created for '{dataset_dir.name}'.")


def generate_for_dataset(dataset_dir: Path) -> None:
    # Check for image dataset (has images/ subdirectory with class folders)
    images_dir = dataset_dir / "images"
    if images_dir.is_dir() and any(d.is_dir() for d in images_dir.iterdir()):
        _generate_image_dataset_yaml(dataset_dir)
        return

    csv_path = dataset_dir / "data.csv"
    if not csv_path.exists():
        print(f"  Skipping {dataset_dir.name} — no data.csv or images/ found.")
        return

    print(f"\nDataset '{dataset_dir.name}' is missing dataset.yaml.")
    print("Please answer the following questions:\n")

    df = pd.read_csv(csv_path)
    print(f"  Columns found: {', '.join(df.columns)}\n")


    target = None
    while target not in df.columns:
        target = input(f"  Target column name: ").strip()
        if target not in df.columns:
            print(f"  Column '{target}' not found. Try again.")

    task_type = _ask(
        "Task type",
        {"C": "classification", "R": "regression"}
    )

    source = _ask_text("  Source/origin of dataset", default="unknown")
    description = _ask_text("  Short description", default="")

    features = [col for col in df.columns if col != target]
    type_map = {
        "float64": "float", "float32": "float", "Float64": "float",
        "int64": "int", "int32": "int", "Int64": "int",
        "object": "string", "string": "string", "category": "string",
        "bool": "bool", "boolean": "bool",
    }
    schema = {col: type_map.get(str(dtype), "string") for col, dtype in df.dtypes.items()}

    # Build constraints block — auto-populated from data, no additional prompts needed.
    # label_classes is only included for classification so validate_split_data() can
    # check label integrity on every future pipeline run with this dataset.
    constraints: dict = {
        "min_rows": 10,
        "max_null_fraction": 0.0,
    }
    if task_type == "classification":
        constraints["label_classes"] = sorted(df[target].dropna().unique().tolist())

    metadata = {
        "name": dataset_dir.name,
        "task_type": task_type,
        "description": description,
        "source": source,
        "created_at": str(date.today()),
        "features": features,
        "target": target,
        "schema": schema,
        "constraints": constraints,
    }

    output_path = dataset_dir / "dataset.yaml"
    atomic_write_text(output_path, yaml.dump(metadata, default_flow_style=False, sort_keys=False))

    print(f"  dataset.yaml created for '{dataset_dir.name}'.")


def detect_and_generate(
    raw_dir: Path = Path("data/raw"),
    non_interactive: bool = False,
) -> list[str]:
    
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: '{raw_dir.resolve()}'. "
            "Expected structure: data/raw/<dataset_name>/data.csv (+ dataset.yaml). "
            "Fix: create the directory, or run from repo root, or set the correct raw_dir path."
        )
    if not raw_dir.is_dir():
        raise NotADirectoryError(
            f"Raw data path exists but is not a directory: '{raw_dir.resolve()}'. "
            "Fix: point raw_dir to a directory."
        )
    
    missing = [
        d for d in raw_dir.iterdir()
        if d.is_dir() and not (d / "dataset.yaml").exists()
    ]

    if not missing:
        return []

    if non_interactive:
        names = ", ".join(d.name for d in missing)
        raise RuntimeError(
            f"non-interactive mode: {len(missing)} dataset(s) are missing dataset.yaml: "
            f"{names}. Run the pipeline interactively once to generate the missing files."
        )

    print("\n=== Dataset Setup ===")
    print(f"Found {len(missing)} dataset(s) without metadata:\n")

    for dataset_dir in missing:
        generate_for_dataset(dataset_dir)

    return [d.name for d in missing]


if __name__ == "__main__":
    detect_and_generate()