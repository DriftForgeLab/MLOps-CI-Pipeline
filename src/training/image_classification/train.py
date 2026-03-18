# =============================================================================
# src/training/image_classification/train.py — Image classification training
# =============================================================================
# Responsibility: Train a classification model on preprocessed image data
# (flattened numpy vectors from NPZ archives) and return the fitted estimator
# with training metadata.
#
# Follows the same contract as classification/train.py — returns TrainingResult.
# =============================================================================

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.config.loader import PipelineConfig, load_training_config
from src.data.preprocess import PREPROCESSED_SUBDIR
from src.training.model_factory import create_model
from src.training import TrainingResult

logger = logging.getLogger(__name__)


def run_training(config: PipelineConfig, version_id: str) -> TrainingResult:
    """
    Train a classification model on preprocessed image data (NPZ).

    Loads flattened image vectors from train.npz and class metadata from
    feature_map.json. Uses the same model factory as tabular classification.

    Args:
        config:     Validated PipelineConfig from load_config()
        version_id: Dataset version ID from versioning step

    Returns:
        TrainingResult with fitted model and training metadata.

    Raises:
        FileNotFoundError: If preprocessed data or feature_map.json is missing.
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

    # --- Load preprocessed training data from NPZ ---
    train_npz_path = preprocessed_dir / "train.npz"
    if not train_npz_path.exists():
        raise FileNotFoundError(
            f"Preprocessed training data not found: {train_npz_path}"
        )

    data = np.load(train_npz_path)
    X, y = data["X"], data["y"]

    model = create_model(
        training_config, task_type="image_classification", random_seed=config.random_seed
    )
    model.fit(X, y)

    hp = training_config.model.hyperparameters
    hyperparameters = {k: v for k, v in vars(hp).items()}

    logger.info(
        "  Training complete: algorithm=%s, samples=%d, features=%d",
        training_config.model.algorithm,
        X.shape[0],
        X.shape[1],
    )

    return TrainingResult(
        model=model,
        algorithm=training_config.model.algorithm,
        hyperparameters=hyperparameters,
        dataset_version_id=version_id,
        random_seed=config.random_seed,
        trained_at=datetime.now(timezone.utc).isoformat(),
        train_rows=X.shape[0],
    )
