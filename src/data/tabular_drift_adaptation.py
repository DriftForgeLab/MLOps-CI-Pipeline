# =============================================================================
# src/data/tabular_drift_adaptation.py — Tabular drift-adaptive retraining
# =============================================================================
# Pure data operations for the tabular drift-adaptive retraining workflow:
#
#   1. Load feature contract (input features, target, fitted pipeline).
#   2. Validate and split a labeled drifted CSV into train / holdout.
#   3. Evaluate a trained model on the holdout CSV → baseline metrics.
#   4. Save holdout CSV and baseline metrics alongside each other.
#   5. Append training rows to the raw dataset CSV → triggers new version_id.
#
# This module is intentionally stateless — functions take paths in, produce
# files out, and return results. All orchestration lives in the CLI.
# =============================================================================

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.io import atomic_write_json
from src.data.preprocess import PREPROCESSED_SUBDIR

logger = logging.getLogger(__name__)

_BASELINE_FILENAME = "baseline_metrics.json"
_HOLDOUT_FILENAME = "holdout.csv"


# =============================================================================
# Feature contract
# =============================================================================

def load_feature_contract(
    processed_dir: Path,
    dataset_name: str,
    version_id: str,
) -> dict:
    """Load input features, target, output features, and fitted sklearn pipeline.

    Args:
        processed_dir: Base processed data directory.
        dataset_name:  Dataset identifier.
        version_id:    Content-addressed version hash.

    Returns:
        Dict with keys: input_features, output_features, target, pipeline.

    Raises:
        FileNotFoundError: If feature_map.json or pipeline.pkl are missing.
        ValueError: If feature_map.json is malformed.
    """
    preprocessed_dir = Path(processed_dir) / dataset_name / version_id / PREPROCESSED_SUBDIR

    fm_path = preprocessed_dir / "feature_map.json"
    if not fm_path.exists():
        raise FileNotFoundError(
            f"feature_map.json not found at '{fm_path}'. "
            "Run the preprocessing stage first."
        )
    with open(fm_path) as f:
        fm = json.load(f)

    numeric_input: list[str] = fm.get("numeric_input") or []
    categorical_input: list[str] = fm.get("categorical_input") or []
    input_features = numeric_input + categorical_input

    output_features: list[str] = fm.get("output_features", [])
    target: str = fm.get("target", "")

    if not input_features:
        raise ValueError(
            f"feature_map.json at '{fm_path}' has no numeric_input or categorical_input. "
            "Re-run preprocessing to regenerate feature_map.json."
        )
    if not target:
        raise ValueError(f"feature_map.json at '{fm_path}' has no 'target' key.")

    pipeline_path = preprocessed_dir / "pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(
            f"pipeline.pkl not found at '{pipeline_path}'. "
            "Run the preprocessing stage first."
        )
    pipeline = joblib.load(pipeline_path)

    return {
        "input_features": input_features,
        "output_features": output_features,
        "target": target,
        "pipeline": pipeline,
    }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_on_holdout_tabular(
    model,
    holdout_df: pd.DataFrame,
    feature_contract: dict,
    task_type: str,
) -> dict:
    """Evaluate a tabular model on the held-out drifted rows.

    Applies the same preprocessing pipeline that was used during training, then
    runs model.predict(). Computes classification or regression metrics depending
    on task_type.

    Args:
        model:            Fitted sklearn model (joblib-loaded).
        holdout_df:       DataFrame with raw feature columns and target column.
        feature_contract: Output of load_feature_contract().
        task_type:        "classification" or "regression".

    Returns:
        Dict of metric name → float value, plus n_samples (int).

    Raises:
        ValueError: If holdout_df is missing required columns.
    """
    input_features: list[str] = feature_contract["input_features"]
    target: str = feature_contract["target"]
    pipeline = feature_contract["pipeline"]

    required = set(input_features) | {target}
    missing = required - set(holdout_df.columns)
    if missing:
        raise ValueError(
            f"Holdout CSV is missing required columns: {sorted(missing)}. "
            f"Expected columns: {sorted(required)}"
        )

    X_raw = holdout_df[input_features]
    y_true = holdout_df[target].values

    X = pipeline.transform(X_raw)

    y_pred = model.predict(X)

    if task_type == "classification":
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
        )
        return {
            "accuracy":  round(float(accuracy_score(y_true, y_pred)), 4),
            "f1_score":  round(float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "recall":    round(float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
            "n_samples": int(len(y_true)),
        }
    else:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        return {
            "r2":        round(float(r2_score(y_true, y_pred)), 4),
            "mae":       round(float(mean_absolute_error(y_true, y_pred)), 4),
            "rmse":      round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
            "n_samples": int(len(y_true)),
        }


# =============================================================================
# Holdout split
# =============================================================================

def split_holdout_tabular(
    df: pd.DataFrame,
    target: str,
    task_type: str,
    holdout_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a labeled drifted DataFrame into training and holdout portions.

    Uses stratified splitting for classification (preserves class proportions),
    and random splitting for regression.

    Args:
        df:            Labeled drifted DataFrame (raw features + target).
        target:        Target column name.
        task_type:     "classification" or "regression".
        holdout_ratio: Fraction to hold out (0 < ratio < 1).
        seed:          Random seed for reproducibility.

    Returns:
        Tuple (train_df, holdout_df) both reset-indexed.

    Raises:
        ValueError: If df has fewer than 2 rows.
    """
    from sklearn.model_selection import train_test_split

    if len(df) < 2:
        raise ValueError(
            f"Drifted CSV has only {len(df)} row(s) — need at least 2 to split."
        )

    stratify = df[target] if task_type == "classification" else None

    # Clamp ratio so both splits always have at least 1 row
    ratio = max(1 / len(df), min(holdout_ratio, (len(df) - 1) / len(df)))

    try:
        train_df, holdout_df = train_test_split(
            df,
            test_size=ratio,
            random_state=seed,
            stratify=stratify,
        )
    except ValueError:
        # Stratification can fail if any class has only 1 sample
        train_df, holdout_df = train_test_split(
            df,
            test_size=ratio,
            random_state=seed,
        )
        logger.warning(
            "Stratified split failed (some classes may have too few samples). "
            "Falling back to random split."
        )

    return train_df.reset_index(drop=True), holdout_df.reset_index(drop=True)


# =============================================================================
# Dataset mutation (triggers new version_id)
# =============================================================================

def append_training_rows_to_dataset(
    train_df: pd.DataFrame,
    raw_csv_path: Path,
) -> int:
    """Append drifted training rows to the raw dataset CSV.

    After calling this function the CSV at raw_csv_path contains the original
    rows plus the new drifted rows. The next pipeline run will compute a new
    SHA-256 hash of the file, generating a fresh version_id and triggering
    re-preprocessing on the merged dataset.

    Args:
        train_df:      Drifted training rows (same column schema as the dataset).
        raw_csv_path:  Path to data/raw/<dataset>/data.csv.

    Returns:
        Number of rows appended.

    Raises:
        FileNotFoundError: If raw_csv_path does not exist.
        ValueError: If train_df columns do not match the existing CSV.
    """
    raw_csv_path = Path(raw_csv_path)
    if not raw_csv_path.exists():
        raise FileNotFoundError(
            f"Raw dataset CSV not found at '{raw_csv_path}'. "
            "Ensure the dataset has been set up correctly."
        )

    existing_df = pd.read_csv(raw_csv_path)

    if list(train_df.columns) != list(existing_df.columns):
        raise ValueError(
            f"Drifted CSV columns {list(train_df.columns)} do not match "
            f"dataset CSV columns {list(existing_df.columns)}. "
            "Ensure the drifted CSV has the same schema as the original dataset."
        )

    merged_df = pd.concat([existing_df, train_df], ignore_index=True)
    merged_df.to_csv(raw_csv_path, index=False)

    logger.info(
        "Appended %d drifted row(s) to '%s' (total now %d rows).",
        len(train_df), raw_csv_path, len(merged_df),
    )
    return len(train_df)


# =============================================================================
# Persistence
# =============================================================================

def save_holdout_csv(holdout_df: pd.DataFrame, holdout_dir: Path) -> Path:
    """Save the holdout DataFrame to holdout_dir/holdout.csv.

    Args:
        holdout_df:  Held-out drifted rows (raw features + target).
        holdout_dir: Root holdout directory (created if missing).

    Returns:
        Path to the written CSV.
    """
    holdout_dir = Path(holdout_dir)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    path = holdout_dir / _HOLDOUT_FILENAME
    holdout_df.to_csv(path, index=False)
    logger.info("Holdout CSV saved to '%s' (%d rows).", path, len(holdout_df))
    return path


def save_baseline_metrics(metrics: dict, holdout_dir: Path) -> Path:
    """Persist baseline (pre-retrain) evaluation metrics to the holdout dir.

    Args:
        metrics:     Output of evaluate_on_holdout_tabular().
        holdout_dir: Root holdout directory.

    Returns:
        Path to the written baseline_metrics.json.
    """
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Baseline evaluation of the Production model on the held-out "
            "drifted rows, measured before retraining on the drifted batch."
        ),
        "metrics": metrics,
    }
    path = Path(holdout_dir) / _BASELINE_FILENAME
    atomic_write_json(path, payload)
    logger.info("Baseline metrics saved to '%s'.", path)
    return path


def load_baseline_metrics(holdout_dir: Path) -> dict | None:
    """Load previously saved baseline metrics, or None if not found.

    Args:
        holdout_dir: Root holdout directory.

    Returns:
        Baseline payload dict (with "metrics" key), or None.
    """
    path = Path(holdout_dir) / _BASELINE_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_holdout_csv(holdout_dir: Path) -> pd.DataFrame | None:
    """Load the saved holdout CSV, or None if not found.

    Args:
        holdout_dir: Root holdout directory.

    Returns:
        Holdout DataFrame, or None if holdout.csv does not exist.
    """
    path = Path(holdout_dir) / _HOLDOUT_FILENAME
    if not path.exists():
        return None
    return pd.read_csv(path)
