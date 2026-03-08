# =============================================================================
# src/data/split.py — Deterministic train/val/test split
# =============================================================================
# Splits a versioned dataset into train, val, and test sets using the
# random seed from config. Same seed always produces the same split.
#
# Stratification: for classification tasks, stratify= is passed to sklearn
# so that each class appears in each split at the same proportion as in the
# full dataset. This prevents a split from being missing a class entirely,
# which would silently degrade both training and evaluation.
# =============================================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.common.io import atomic_write_csv, atomic_write_json, atomic_write_text

logger = logging.getLogger(__name__)

### !!! IMPORTANT: May need to change where it writes the split metadata back to -
# currently it updates dataset.yaml in place, but it might be better to write a separate split.yaml

def _validate_split_ratios(train_ratio: float, val_ratio: float) -> float:
    """
    Validate split ratios and return computed test_ratio.
    
    Enforces:
+       - 0 < train_ratio < 1
        - 0 <= val_ratio < 1
        - train_ratio + val_ratio < 1  (so test_ratio > 0)

    Args:
        train_ratio (float): Proportion of data to use for training (must be in (0, 1)). 
        val_ratio (float): Proportion of data to use for validation (must be in [0, 1) and train_ratio + val_ratio < 1).

    Returns:
        float: Computed test_ratio = 1 - train_ratio - val_ratio (guaranteed to be > 0 if validation passes).
    """
    for name, r in [("train_ratio", train_ratio), ("val_ratio", val_ratio)]:
        if not isinstance(r, (int, float)):
            raise ValueError(f"{name} must be a number, got {type(r).__name__}")
    
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    elif not (0.0 <= val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    
    total = train_ratio + val_ratio
    if not (total < 1.0):
        raise ValueError(
            f"Invalid split ratios: train_ratio + val_ratio must be < 1. "
            f"Got {train_ratio} + {val_ratio} = {total}."
        )
        
    test_ratio = 1.0 - total
    if test_ratio <= 0.0:
        raise ValueError(
            f"Computed test_ratio must be > 0. Got test_ratio={test_ratio} "
            f"from train_ratio={train_ratio}, val_ratio={val_ratio}."
        )
    return test_ratio

def _write_split_metadata(split_dir: Path, payload: dict) -> None:
    """Write per-split metadata.json (small, explicit, CI-friendly)."""
    split_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(split_dir / "metadata.json", payload)

def _split_outputs_exist(version_dir: Path) -> bool:
    return (
        all((version_dir / s / "data.csv").exists() for s in ("train", "val", "test"))
        and all((version_dir / s / "metadata.json").exists() for s in ("train", "val", "test"))
        and (version_dir / "dataset.yaml").exists()
    )

def split_dataset(
    dataset_name: str,
    version_id: str,
    random_seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    processed_dir: Path = Path("data/processed"),
) -> None:
    version_dir = processed_dir / dataset_name / version_id
    csv_path = version_dir / "data.csv"

    train_dir = version_dir / "train"
    val_dir = version_dir / "val"
    test_dir = version_dir / "test"


    if _split_outputs_exist(version_dir):
        logger.info("  Split outputs already exist for version '%s' — skipping.", version_id)
        return
        
    test_ratio = _validate_split_ratios(train_ratio, val_ratio)

    yaml_path = version_dir / "dataset.yaml"
    with open(yaml_path) as f:
        metadata = yaml.safe_load(f)

    task_type: str = metadata.get("task_type", "regression")
    target: str = metadata.get("target", "")

    df = pd.read_csv(csv_path)

    # Use stratify for classification so class proportions are preserved per split.
    stratify_col = df[target] if task_type == "classification" and target in df.columns else None

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        stratify=stratify_col,
    )
    temp_stratify = temp_df[target] if stratify_col is not None else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(test_ratio / (val_ratio + test_ratio)),
        random_state=random_seed,
        stratify=temp_stratify,
    )

    for split_dir, split_df in [(train_dir, train_df), (val_dir, val_df), (test_dir, test_df)]:
        split_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_csv(split_df, split_dir / "data.csv")
        
    now = datetime.now(timezone.utc).isoformat()
    base_meta = {
        "dataset_name": dataset_name,
        "dataset_version_id": version_id,
        "created_at": now,
        "random_seed": random_seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "task_type": task_type,
        "target": target,
    }
    
    def _label_dist(d: pd.DataFrame) -> dict | None:
        if task_type == "classification" and target in d.columns:
            vc = d[target].value_counts(dropna=False)
            return {str(k): int(v) for k, v in vc.items()}
        return None
    
    for name, d, split_dir in [("train", train_df, train_dir), ("val", val_df, val_dir), ("test", test_df, test_dir)]:
        payload = dict(base_meta)
        payload.update({"split": name, "rows": int(len(d))})
        ld = _label_dist(d)
        if ld is not None:
            payload["label_distribution"] = ld
        _write_split_metadata(split_dir, payload)       
        
    split_block: dict = {
        "train": len(train_df),
        "val": len(val_df),
        "test": len(test_df),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": round(test_ratio, 2),
        "random_seed": random_seed,
    }

    # Write per-split class counts for classification — used by drift analysis
    # and validation to confirm boundaries are preserved.
    if task_type == "classification" and target in df.columns:
        split_block["label_distribution"] = {
            "train": train_df[target].value_counts().to_dict(),
            "val":   val_df[target].value_counts().to_dict(),
            "test":  test_df[target].value_counts().to_dict(),
        }

    metadata["split"] = split_block

    yaml_text = yaml.dump(metadata, default_flow_style=False, sort_keys=False)
    atomic_write_text(yaml_path, yaml_text)

    print(f"  Dataset split: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
    logger.info(
        "Split complete for %s/%s (seed=%s): train=%d val=%d test=%d",
        dataset_name, version_id, random_seed, len(train_df), len(val_df), len(test_df)
    )
