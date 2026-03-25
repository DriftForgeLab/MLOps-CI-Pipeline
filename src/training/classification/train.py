# =============================================================================
# src/training/classification/train.py — Classification training
# =============================================================================
# Responsibility: Train a classification model on the preprocessed training
# split and return the fitted estimator with training metadata.
#
# Thin wrapper around the shared tabular training logic in _tabular.py.
# =============================================================================

from src.config.schema import PipelineConfig
from src.training import TrainingResult
from src.training._tabular import run_tabular_training


def run_training(config: PipelineConfig, version_id: str) -> TrainingResult:
    """
    Train a classification model on the preprocessed training split.

    Delegates to the shared tabular training flow with task_type="classification".

    Args:
        config:     Validated PipelineConfig from load_config()
        version_id: Dataset version ID from versioning step

    Returns:
        TrainingResult with fitted model and training metadata.

    Raises:
        FileNotFoundError: If preprocessed data or feature_map.json is missing.
        ValueError:        If CSV columns do not match feature_map.json contract.
    """
    return run_tabular_training(config, version_id, task_type="classification")
