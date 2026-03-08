# =============================================================================
# src/training/regression/train.py — Regression training
# =============================================================================
# Responsibility: Train a regression model on the preprocessed training
# split and return the fitted estimator with training metadata.
# =============================================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.config.loader import PipelineConfig, load_training_config
from src.data.preprocess import PREPROCESSED_SUBDIR
from src.training.model_factory import create_model
from src.training import TrainingResult

logger = logging.getLogger(__name__)


def run_training(config: PipelineConfig, version_id: str) -> TrainingResult:
    """
    Train a regression model on the preprocessed training split.

    Feature and target columns are loaded from feature_map.json (written by
    the preprocessing stage). This removes all positional column inference
    and raises loudly if the CSV columns do not match the contract.

    Args:
        config:     Validated PipelineConfig from load_config()
        version_id: Dataset version ID from versioning step

    Returns:
        TrainingResult with fitted model and training metadata.

    Raises:
        FileNotFoundError: If preprocessed data or feature_map.json is missing.
        ValueError:        If CSV columns do not match feature_map.json contract.
    """
    training_config = load_training_config(Path(config.configs.training))

    version_dir = Path(config.data.processed) / config.dataset / version_id
    preprocessed_dir = version_dir / PREPROCESSED_SUBDIR

    # --- Load feature contract from feature_map.json ---
    feature_map_path = preprocessed_dir / "feature_map.json"
    if not feature_map_path.exists():
        raise FileNotFoundError(
            f"feature_map.json not found at '{feature_map_path}'. "
            "Run the preprocessing stage before training."
        )
    with open(feature_map_path) as f:
        feature_map = json.load(f)
    expected_features: list[str] = feature_map["output_features"]
    target: str = feature_map["target"]

    # --- Load dataset.yaml to confirm target (belt-and-suspenders) ---
    yaml_path = version_dir / "dataset.yaml"
    if yaml_path.exists():
        with open(yaml_path) as f:
            meta = yaml.safe_load(f)
        yaml_target = meta.get("target")
        if yaml_target and yaml_target != target:
            raise ValueError(
                f"Target mismatch: feature_map.json says '{target}' "
                f"but dataset.yaml says '{yaml_target}'. "
                "Re-run preprocessing to regenerate feature_map.json."
            )

    preprocessed_path = preprocessed_dir / "train.csv"
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed training data not found: {preprocessed_path}"
        )

    df = pd.read_csv(preprocessed_path)

    # --- Loud mismatch error: CSV columns must match feature_map contract ---
    expected_cols = expected_features + [target]
    actual_cols = list(df.columns)
    if actual_cols != expected_cols:
        raise ValueError(
            f"Preprocessed train.csv columns do not match feature_map.json contract.\n"
            f"  Expected: {expected_cols}\n"
            f"  Actual:   {actual_cols}\n"
            "Re-run preprocessing to regenerate train.csv and feature_map.json."
        )

    X = df[expected_features].values
    y = df[target].values

    model = create_model(training_config, task_type="regression", random_seed=config.random_seed)
    model.fit(X, y)

    hp = training_config.model.hyperparameters
    hyperparameters = {k: v for k, v in vars(hp).items()}

    logger.info(
        "  Training complete: algorithm=%s, rows=%d",
        training_config.model.algorithm,
        len(df),
    )

    return TrainingResult(
        model=model,
        algorithm=training_config.model.algorithm,
        hyperparameters=hyperparameters,
        dataset_version_id=version_id,
        random_seed=config.random_seed,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_rows=len(df),
    )
