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


def compute_multiscale_drift(
    batch: np.ndarray,
    reference: np.ndarray,
    num_levels: int = 3,
) -> dict:
    """Characterise drift by comparing image distributions at multiple spatial scales.

    Builds a Gaussian pyramid from both batch and reference, then computes the
    per-channel Wasserstein distance at each pyramid level independently.  The
    resulting profile answers *what kind* of content is drifting, not just
    *whether* drift exists:

      - Score decays steeply toward coarser levels (high scale_decay_ratio)
        → drift is concentrated in fine detail: sensor noise, sharpening
          parameter changes, or JPEG / compression artefacts.

      - Score persists at coarser levels (ratio near 1.0)
        → drift is global / low-frequency: brightness shift, colour cast,
          or overall illumination / exposure change.

      - Intermediate decay
        → mid-scale source: texture change or scene content shift.

    This is an *interpretation layer*, not an additional detection step.
    The authoritative drift decision comes from compute_statistical_drift or
    compute_embedding_drift.  Multi-scale analysis is meaningful only when
    those functions already report a non-trivial drift score.

    Args:
        batch:      Incoming image batch, (N, H, W, C) or (N, H, W), float64.
        reference:  Reference images from the training split, same shape.
        num_levels: Number of pyramid levels to compute, including level 0
                    (full resolution).  Levels are silently capped if images
                    become too small (minimum spatial dimension: 4 px).

    Returns:
        Dict with keys:
          "levels"            — list of per-level result dicts, each with:
                                  level, resolution_factor, spatial_shape,
                                  channel_scores, overall_score.
          "scale_decay_ratio" — level-0 overall score divided by the coarsest
                                level overall score.  High values indicate
                                fine-detail drift; values near 1.0 indicate
                                global / coarse drift.
          "interpretation"    — plain-language string describing the likely
                                drift source.

    Raises:
        ImportError: If scipy is not installed.
        ValueError:  If batch shape is incompatible with pyramid construction.
    """
    batch_pyramid = _build_gaussian_pyramid(batch, num_levels)
    ref_pyramid = _build_gaussian_pyramid(reference, num_levels)

    # Use the smaller of the two pyramid depths (should always be equal).
    depth = min(len(batch_pyramid), len(ref_pyramid))

    level_results: list[dict] = []
    for lvl in range(depth):
        bat_lvl = batch_pyramid[lvl]
        ref_lvl = ref_pyramid[lvl]

        channel_scores, overall_score = compute_statistical_drift(bat_lvl, ref_lvl)

        h = bat_lvl.shape[1]
        w = bat_lvl.shape[2]

        level_results.append({
            "level": lvl,
            "resolution_factor": f"1/{2 ** lvl}",
            "spatial_shape": [h, w],
            "channel_scores": {k: round(v, 6) for k, v in channel_scores.items()},
            "overall_score": round(overall_score, 6),
        })

    level_0_score = level_results[0]["overall_score"]
    coarsest_score = level_results[-1]["overall_score"]
    scale_decay_ratio = round(level_0_score / max(coarsest_score, 1e-8), 3)

    interpretation = _interpret_scale_profile(scale_decay_ratio)

    return {
        "levels": level_results,
        "scale_decay_ratio": scale_decay_ratio,
        "interpretation": interpretation,
    }


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

def _build_gaussian_pyramid(
    images: np.ndarray,
    num_levels: int,
) -> list[np.ndarray]:
    """Build a Gaussian pyramid by iteratively blurring and halving spatial resolution.

    Each level is produced by applying a mild Gaussian blur (sigma=0.5 in
    each spatial dimension) and then downsampling by 2 via strided slicing.
    The blur prevents aliasing and ensures that coarser levels represent
    genuinely lower-frequency content rather than subsampled artefacts.

    Levels stop being appended if the downsampled spatial dimensions would
    fall below 4 px in either axis.

    Args:
        images:     Image array (N, H, W, C) or (N, H, W), float64.
        num_levels: Maximum number of levels (including level 0 = original).

    Returns:
        List of arrays.  Element 0 is the original; each subsequent element
        is half the spatial resolution of the previous.
    """
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError:
        raise ImportError(
            "scipy is required for multi-scale drift analysis. "
            "Install with: pip install scipy"
        )

    pyramid = [images]
    current = images

    for _ in range(num_levels - 1):
        if current.ndim == 4:  # (N, H, W, C)
            blurred = gaussian_filter(
                current.astype(np.float64), sigma=[0, 0.5, 0.5, 0]
            )
            downsampled = blurred[:, ::2, ::2, :]
        elif current.ndim == 3:  # (N, H, W)
            blurred = gaussian_filter(
                current.astype(np.float64), sigma=[0, 0.5, 0.5]
            )
            downsampled = blurred[:, ::2, ::2]
        else:
            break

        # Both spatial dimensions must be >= 4 px to remain statistically useful.
        if min(downsampled.shape[1], downsampled.shape[2]) < 4:
            break

        pyramid.append(downsampled)
        current = downsampled

    return pyramid


def _interpret_scale_profile(scale_decay_ratio: float) -> str:
    """Map a scale-decay ratio to a plain-language drift source description.

    Thresholds are intentionally coarse — the interpretation is a directional
    hint, not a definitive causal claim.

    Args:
        scale_decay_ratio: level-0 Wasserstein / coarsest-level Wasserstein.
                           Values near 1.0 mean drift persists at all scales.
                           High values mean drift vanishes at coarser scales.

    Returns:
        Human-readable interpretation string.
    """
    if scale_decay_ratio > 3.0:
        return (
            "fine-detail drift — the distribution shift is concentrated in "
            "high-frequency content (likely: sensor noise floor change, "
            "sharpening parameter shift, or JPEG / compression artefacts)"
        )
    if scale_decay_ratio < 1.5:
        return (
            "global/coarse drift — the distribution shift persists at all "
            "spatial scales (likely: brightness change, colour cast, or "
            "overall illumination / exposure shift)"
        )
    return (
        "mid-scale drift — the distribution shift is concentrated in "
        "intermediate spatial frequencies (likely: texture change or "
        "scene content shift)"
    )


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
    from src.common.device import resolve_device

    device = resolve_device()
    X_t = X.transpose(0, 3, 1, 2) if X.ndim == 4 else X
    tensor = torch.tensor(X_t, dtype=torch.float32, device=device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        feature_extractor = model.net[:-1]
        features = feature_extractor(tensor)

    return features.cpu().numpy()


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
