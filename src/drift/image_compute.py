# =============================================================================
# src/drift/image_compute.py — Core image drift computation
# =============================================================================
# Provides the statistical primitives for image drift detection.
# Mirrors the role of compute.py (Evidently wrapper) for tabular drift.
#
# Two methods:
#   statistical — per-channel Wasserstein distance between pixel distributions
#   embedding   — Maximum Mean Discrepancy (MMD) in CNN feature space
#
# These functions are pure computation: they take arrays in, return scores out.
# Orchestration (batch size check, result assembly, scenario matching) lives in
# src/monitoring/image_drift_monitor.py.
# =============================================================================

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Wasserstein distance thresholds for severity classification.
#
# These defaults assume z-scored images (mean≈0, std≈1 per channel),
# which is what the preprocessing pipeline produces when normalize=True.
#
# IMPORTANT: These values are NOT universal. Validate empirically on your
# dataset. Rule of thumb: set medium ≈ 2× noise floor, high ≈ 5× noise floor,
# where the noise floor is the Wasserstein distance between two independent
# val splits from the same dataset.
DEFAULT_IMAGE_SEVERITY_THRESHOLDS: dict[str, float] = {
    "medium": 0.10,   # Wasserstein ≥ 0.10 → medium  (z-scored images)
    "high":   0.25,   # Wasserstein ≥ 0.25 → high    (z-scored images)
}


def compute_statistical_drift(
    batch: np.ndarray,
    reference: np.ndarray,
) -> tuple[dict[str, float], float]:
    """Compute per-channel Wasserstein distance between batch and reference.

    Flattens ALL pixels across ALL images in each group into a single 1-D
    array per channel, then computes the first Wasserstein distance (earth
    mover's distance) between the two flattened pixel-value distributions.

    IMPORTANT — what this measures:
      This is a PIXEL DISTRIBUTION statistic, not an image-level statistic.
      A batch of N images with H×W pixels each contributes N×H×W values to
      the comparison. The distance reflects aggregate shifts in brightness,
      colour balance, noise level, or contrast — not whether individual
      images look different in a semantic sense.

    Works on both RGB (N, H, W, 3) and grayscale (N, H, W) inputs:
      RGB:       returns scores for channels "r", "g", "b"; overall = mean.
      Grayscale: returns score for channel "gray"; overall = gray score.

    Args:
        batch:     Incoming image batch, float64, any normalization.
        reference: Reference images (training split), same preprocessing.

    Returns:
        Tuple of (channel_scores dict, overall_score float).

    Raises:
        ImportError: If scipy is not installed.
        ValueError:  If batch shape is unexpected.
    """
    try:
        from scipy.stats import wasserstein_distance
    except ImportError:
        raise ImportError(
            "scipy is required for statistical drift detection. "
            "Install with: pip install scipy"
        )

    channel_scores: dict[str, float] = {}

    if batch.ndim == 4 and batch.shape[-1] >= 3:
        # RGB — compare each colour channel independently
        for i, name in enumerate(["r", "g", "b"]):
            ref_flat = reference[..., i].ravel()
            bat_flat = batch[..., i].ravel()
            channel_scores[name] = float(wasserstein_distance(ref_flat, bat_flat))
        overall_score = float(np.mean(list(channel_scores.values())))

    elif batch.ndim == 4 and batch.shape[-1] == 1:
        # Single-channel stored as (N, H, W, 1)
        ref_flat = reference[..., 0].ravel()
        bat_flat = batch[..., 0].ravel()
        channel_scores["gray"] = float(wasserstein_distance(ref_flat, bat_flat))
        overall_score = channel_scores["gray"]

    elif batch.ndim == 3:
        # Grayscale (N, H, W)
        ref_flat = reference.ravel()
        bat_flat = batch.ravel()
        channel_scores["gray"] = float(wasserstein_distance(ref_flat, bat_flat))
        overall_score = channel_scores["gray"]

    else:
        raise ValueError(
            f"Unexpected batch shape {batch.shape}. Expected (N, H, W, C) or (N, H, W)."
        )

    return channel_scores, overall_score


def compute_embedding_drift(
    batch: np.ndarray,
    reference: np.ndarray,
    model,
    task_type: str = "image_classification_cnn",
) -> float:
    """Compute MMD between CNN feature embeddings of batch and reference.

    Extracts penultimate-layer features from a SimpleCNN model, then computes
    the unbiased Maximum Mean Discrepancy with an RBF kernel.

    Args:
        batch:     Incoming images, shape (N, H, W, C), float64.
        reference: Reference images (training split), same shape.
        model:     Trained SimpleCNN (PyTorch). Must have a .net Sequential.
        task_type: Accepted for API compatibility; NHWC→NCHW transpose always applied.

    Returns:
        MMD score (float, ≥ 0). Higher = more distribution shift.

    Raises:
        ImportError: If torch is not installed.
        ValueError:  If the model does not have a .net attribute.
    """
    if not hasattr(model, "net"):
        raise ValueError(
            "Embedding drift requires a SimpleCNN model with a .net Sequential attribute. "
            "Use method='statistical' for other model types."
        )

    ref_features = _extract_cnn_features(model, reference)
    bat_features = _extract_cnn_features(model, batch)
    return _compute_mmd(ref_features, bat_features)


def classify_image_severity(
    drift_score: float,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Classify a Wasserstein drift score into a severity label.

    Args:
        drift_score: Wasserstein distance (≥ 0).
        thresholds:  Dict with "medium" and "high" float thresholds.
                     Defaults to DEFAULT_IMAGE_SEVERITY_THRESHOLDS.

    Returns:
        "low", "medium", or "high".
    """
    t = thresholds or DEFAULT_IMAGE_SEVERITY_THRESHOLDS
    if drift_score >= t.get("high", DEFAULT_IMAGE_SEVERITY_THRESHOLDS["high"]):
        return "high"
    if drift_score >= t.get("medium", DEFAULT_IMAGE_SEVERITY_THRESHOLDS["medium"]):
        return "medium"
    return "low"


# =============================================================================
# Private helpers
# =============================================================================

def _extract_cnn_features(model, X: np.ndarray) -> np.ndarray:
    """Extract penultimate-layer features from a SimpleCNN model.

    IMPORTANT — model contract:
      Assumes the model is a SimpleCNN from src/training/image_classification_cnn/train.py.
      Requires model.net to be an nn.Sequential whose last element is the
      classification head. NHWC→NCHW transpose is always applied.

    Args:
        model: SimpleCNN with a .net Sequential attribute.
        X:     Images, shape (N, H, W, C), float64.

    Returns:
        Feature array (N, feature_dim), float32.
    """
    import torch

    X_t = X.transpose(0, 3, 1, 2) if X.ndim == 4 else X
    tensor = torch.tensor(X_t, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        feature_extractor = model.net[:-1]
        features = feature_extractor(tensor)

    return features.numpy()


def _compute_mmd(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute unbiased Maximum Mean Discrepancy with an RBF kernel.

    Uses the median-heuristic bandwidth: sigma^2 = median of pairwise squared
    distances in the combined set.

    Args:
        X: Reference feature matrix (n, d).
        Y: Batch feature matrix (m, d).

    Returns:
        MMD estimate (float, ≥ 0 in expectation).

    Raises:
        ImportError: If scikit-learn is not installed.
    """
    try:
        from sklearn.metrics.pairwise import rbf_kernel
    except ImportError:
        raise ImportError(
            "scikit-learn is required for MMD computation. "
            "Install with: pip install scikit-learn"
        )

    n, m = len(X), len(Y)
    if n < 2 or m < 2:
        logger.warning(
            "MMD requires at least 2 samples per group (got %d ref, %d batch). "
            "Returning 0.0.", n, m,
        )
        return 0.0

    combined = np.concatenate([X, Y], axis=0)
    sq_dists = np.sum((combined[:, None] - combined[None, :]) ** 2, axis=-1)
    median_sq = float(np.median(sq_dists[sq_dists > 0])) if (sq_dists > 0).any() else 1.0
    gamma = 1.0 / (median_sq + 1e-8)

    KXX = rbf_kernel(X, X, gamma=gamma)
    KYY = rbf_kernel(Y, Y, gamma=gamma)
    KXY = rbf_kernel(X, Y, gamma=gamma)

    mmd = (
        (np.sum(KXX) - np.trace(KXX)) / (n * (n - 1))
        + (np.sum(KYY) - np.trace(KYY)) / (m * (m - 1))
        - 2.0 * np.mean(KXY)
    )
    return float(max(0.0, mmd))
