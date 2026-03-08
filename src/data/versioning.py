# =============================================================================
# src/data/versioning.py — Dataset versioning
# =============================================================================
# Creates a deterministic, immutable version of a raw dataset in data/processed/.
# The version ID is a SHA-256 hash of the dataset content, ensuring that
# identical data always produces the same version ID.
# =============================================================================

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def _compute_version_id(csv_path: Path) -> str:
    raw_bytes = csv_path.read_bytes()
    return hashlib.sha256(raw_bytes).hexdigest()[:12]


def create_dataset_version(dataset_name: str, raw_dir: Path = Path("data/raw"), processed_dir: Path = Path("data/processed")) -> Path:
    raw_dataset_dir = raw_dir / dataset_name
    csv_path = raw_dataset_dir / "data.csv"
    yaml_path = raw_dataset_dir / "dataset.yaml"

    if not csv_path.exists():
        raise FileNotFoundError(f"No data.csv found in {raw_dataset_dir}")
    if not yaml_path.exists():
        raise FileNotFoundError(f"No dataset.yaml found in {raw_dataset_dir}")

    version_id = _compute_version_id(csv_path)

    version_dir = processed_dir / dataset_name / version_id
    if version_dir.exists():
        print(f"  Version {version_id} already exists for '{dataset_name}' — skipping.")
        return version_dir

    version_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(csv_path, version_dir / "data.csv")

    with open(yaml_path, "r") as f:
        metadata = yaml.safe_load(f)

    metadata["version_id"] = version_id
    metadata["versioned_at"] = datetime.now(timezone.utc).isoformat()

    with open(version_dir / "dataset.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

    print(f"  Dataset version created: {version_dir}")
    return version_dir