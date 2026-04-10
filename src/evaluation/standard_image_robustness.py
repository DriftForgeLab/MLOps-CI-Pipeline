# =============================================================================
# src/evaluation/standard_image_robustness.py — Augmentation robustness analysis
# =============================================================================
# Offline sensitivity profiling for standard (JPG/PNG) image pipelines.
#
# IMPORTANT DISTINCTION — two separate offline analyses exist in this system:
#
#   Raw images (DNG, raw_input=True):
#     ISP-based drift simulation in src/evaluation/image_drift_analysis.py.
#     Physically grounded — modifies actual ISP camera parameters to simulate
#     real-world sensor/pipeline drift. Results are interpretable in terms of
#     physical camera properties (white balance, gamma, noise level, etc.).
#
#   Standard images (JPG/PNG, raw_input=False):  ← THIS MODULE
#     Augmentation-based robustness testing. Applies geometric and photometric
#     transformations to the validation split and measures model accuracy under
#     each transformation. This is an APPROXIMATE method — augmented images are
#     not real incoming production data, and this analysis does NOT constitute
#     drift detection in the MLOps sense. It answers the question:
#       "How sensitive is this model to common image variations?"
#
# These two methods must NOT be presented as equivalent. Reports from this
# module are labelled "augmentation_robustness" throughout, never "drift".
#
# Real drift detection (production batches vs training reference) for both
# raw and standard image pipelines happens at monitoring time via
# monitor_image_batch() in src/monitoring/image_drift_monitor.py.
#
# Augmentation scenarios
# ----------------------
# Eight fixed scenarios are always evaluated. They are NOT configurable via
# YAML because there are no physical parameters to calibrate — augmentations
# are transformations on already-processed images, not camera pipeline choices.
#
#   horizontal_flip   Mirror left-right — opposite camera orientation
#   rotation_90       90° clockwise — camera tilt
#   rotation_180      180° — upside-down camera
#   brightness_high   +0.5 shift (z-scored) — overexposure / bright scene
#   brightness_low    −0.5 shift (z-scored) — underexposure / dark scene
#   contrast_low      ×0.5 scaling — low-contrast conditions
#   gaussian_noise    σ=0.1 additive noise — sensor / JPEG compression noise
#   gaussian_blur     σ=1.0 spatial blur — defocus or motion blur
#
# Output
# ------
#   drift_scenarios/robustness_report.json  — machine-readable full report
#   drift_scenarios/robustness_report.html  — human-readable summary table
#
# The caller (_run_standard_image_robustness_stage in steps.py) handles
# MLflow logging, so this module has no MLflow dependency.
# =============================================================================

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.common.io import atomic_write_json

logger = logging.getLogger(__name__)

# Default thresholds for classifying accuracy drop into sensitivity severity.
# Same values as ISP sensitivity to allow consistent cross-report comparisons.
DEFAULT_ROBUSTNESS_THRESHOLDS: dict[str, float] = {
    "high":   0.15,   # accuracy drop ≥ 15 percentage points
    "medium": 0.05,   # accuracy drop ≥ 5 percentage points
}

# Split evaluated for both baseline and scenario metrics. Must match the split
# that evaluate.py uses for image tasks (currently "val").
_EVALUATION_SPLIT = "val"


# =============================================================================
# Augmentation functions (pure numpy — no external augmentation library)
# =============================================================================

def _horizontal_flip(X: np.ndarray) -> np.ndarray:
    """Flip images left-right along the width axis."""
    if X.ndim == 4:
        return X[:, :, ::-1, :].copy()
    return X[:, ::-1].copy()


def _rotation(X: np.ndarray, k: int) -> np.ndarray:
    """Rotate images by k×90° counter-clockwise."""
    if X.ndim == 4:
        return np.rot90(X, k=k, axes=(1, 2)).copy()
    return np.rot90(X, k=k, axes=(0, 1)).copy()


def _brightness_shift(X: np.ndarray, delta: float) -> np.ndarray:
    """Additive brightness shift (works on z-scored images)."""
    return X + delta


def _contrast_scale(X: np.ndarray, factor: float) -> np.ndarray:
    """Multiplicative contrast change (compresses / expands value range)."""
    return X * factor


def _gaussian_noise(X: np.ndarray, sigma: float, seed: int = 42) -> np.ndarray:
    """Add Gaussian noise with fixed seed for reproducibility."""
    rng = np.random.default_rng(seed)
    return X + rng.normal(0.0, sigma, X.shape)


def _gaussian_blur(X: np.ndarray, sigma: float) -> np.ndarray:
    """Apply per-channel Gaussian blur using scipy."""
    from scipy.ndimage import gaussian_filter
    if X.ndim == 4:
        # Apply per-image, per-channel
        out = np.empty_like(X)
        for i in range(X.shape[0]):
            for c in range(X.shape[-1]):
                out[i, :, :, c] = gaussian_filter(X[i, :, :, c], sigma=sigma)
        return out
    # Grayscale (N, H, W)
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = gaussian_filter(X[i], sigma=sigma)
    return out


# Scenario definitions: (name, description, function)
_AUGMENTATION_SCENARIOS: list[tuple[str, str, object]] = [
    (
        "horizontal_flip",
        "Mirror images left-right — simulates opposite camera orientation",
        lambda X: _horizontal_flip(X),
    ),
    (
        "rotation_90",
        "90° clockwise rotation — simulates 90° camera tilt",
        lambda X: _rotation(X, k=3),
    ),
    (
        "rotation_180",
        "180° rotation — simulates upside-down camera mounting",
        lambda X: _rotation(X, k=2),
    ),
    (
        "brightness_high",
        "Positive brightness shift (+0.5 on z-scored images) — simulates overexposure",
        lambda X: _brightness_shift(X, +0.5),
    ),
    (
        "brightness_low",
        "Negative brightness shift (−0.5 on z-scored images) — simulates underexposure",
        lambda X: _brightness_shift(X, -0.5),
    ),
    (
        "contrast_low",
        "Contrast reduction (×0.5 on z-scored images) — simulates flat, low-contrast conditions",
        lambda X: _contrast_scale(X, 0.5),
    ),
    (
        "gaussian_noise",
        "Additive Gaussian noise (σ=0.1) — simulates sensor noise or JPEG compression artefacts",
        lambda X: _gaussian_noise(X, sigma=0.1),
    ),
    (
        "gaussian_blur",
        "Gaussian blur (σ=1.0 px) — simulates defocus or camera motion blur",
        lambda X: _gaussian_blur(X, sigma=1.0),
    ),
]


# =============================================================================
# Public API
# =============================================================================

def run_robustness_analysis(
    version_id: str,
    task_type: str,
    preprocessed_dir: Path,
    artifact_dir: Path = Path("artifacts/runs"),
    baseline_report_path: Path | None = None,
    robustness_thresholds: dict[str, float] | None = None,
) -> dict:
    """Evaluate model sensitivity to common image augmentations.

    Applies each augmentation scenario to the validation split and measures
    accuracy and F1 relative to the baseline evaluation report.

    IMPORTANT: This is augmentation-based robustness testing, NOT drift
    detection. Results tell you how the model handles image variations, not
    whether production data has shifted from training data.

    Args:
        version_id:          Dataset/run version hash (locates model artifact).
        task_type:           Must be "image_classification_cnn".
        preprocessed_dir:    Directory containing val.npz for this version.
        artifact_dir:        Base directory for model artifacts.
        baseline_report_path: Path to evaluation_report.json from the evaluation
                              stage. Baseline metrics come from the val split,
                              the same split evaluated here — deltas are valid.
        robustness_thresholds: Dict with "high" and "medium" thresholds.
                              Defaults to DEFAULT_ROBUSTNESS_THRESHOLDS.

    Returns:
        Robustness report dict. Saved to disk by the caller.

    Raises:
        FileNotFoundError: If model.pt or val.npz is missing.
    """
    import json

    thresholds = robustness_thresholds or DEFAULT_ROBUSTNESS_THRESHOLDS
    preprocessed_dir = Path(preprocessed_dir)

    # Load model
    model = _load_cnn_model(version_id, artifact_dir)

    # Load baseline metrics
    baseline_metrics: dict | None = None
    if baseline_report_path and Path(baseline_report_path).exists():
        with open(baseline_report_path) as f:
            eval_report = json.load(f)
        baseline_metrics = eval_report.get("metrics")
        logger.info(
            "  Robustness baseline (%s split) — accuracy=%.4f",
            _EVALUATION_SPLIT,
            baseline_metrics.get("accuracy", float("nan")),
        )
    else:
        logger.warning(
            "  No baseline evaluation report found — delta metrics will be omitted."
        )

    # Load validation images
    val_npz = preprocessed_dir / f"{_EVALUATION_SPLIT}.npz"
    if not val_npz.exists():
        raise FileNotFoundError(
            f"Validation split not found at '{val_npz}'. "
            "Run the preprocessing stage before robustness analysis."
        )
    data = np.load(val_npz)
    X_val, y_true = data["X"], data["y"]

    logger.info(
        "  Running augmentation robustness on %d images (%d scenarios)...",
        len(X_val), len(_AUGMENTATION_SCENARIOS),
    )

    # Evaluate each scenario
    scenario_results: list[dict] = []
    for name, description, aug_fn in _AUGMENTATION_SCENARIOS:
        result = _evaluate_augmentation(
            name=name,
            description=description,
            aug_fn=aug_fn,
            X_val=X_val,
            y_true=y_true,
            model=model,
            baseline_metrics=baseline_metrics,
            thresholds=thresholds,
        )
        if result is not None:
            scenario_results.append(result)
            logger.info(
                "  %-22s accuracy=%.4f  Δacc=%+.4f  sensitivity=%s",
                name,
                result["metrics"]["accuracy"],
                result["delta"].get("accuracy", float("nan")),
                result["sensitivity"],
            )

    if not scenario_results:
        raise ValueError("All augmentation scenarios failed evaluation.")

    ranked = sorted(
        scenario_results,
        key=lambda r: abs(r["delta"].get("accuracy", 0.0)),
        reverse=True,
    )

    report = {
        "schema_version":            "1.0.0",
        "report_type":               "image_augmentation_robustness",
        "generated_at":              datetime.now(timezone.utc).isoformat(),
        "task_type":                 task_type,
        "version_id":                version_id,
        "evaluation_split":          _EVALUATION_SPLIT,
        "baseline_metrics":          baseline_metrics,
        "robustness_thresholds":     thresholds,
        "total_images":              int(len(X_val)),
        "scenarios":                 scenario_results,
        "most_sensitive_augmentation":  ranked[0]["name"],
        "least_sensitive_augmentation": ranked[-1]["name"],
        "scenario_count":            len(scenario_results),
    }

    logger.info(
        "  Robustness analysis complete: most sensitive='%s', least='%s'",
        ranked[0]["name"], ranked[-1]["name"],
    )
    return report


def save_robustness_report_json(report: dict, output_dir: Path) -> Path:
    """Persist the robustness report as JSON.

    Returns:
        Path to the written file.
    """
    path = Path(output_dir) / "robustness_report.json"
    atomic_write_json(path, report)
    return path


def save_robustness_report_html(report: dict, output_dir: Path) -> Path:
    """Render and save the robustness report as a self-contained HTML file.

    Returns:
        Path to the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    html = _render_html_report(report)
    path = output_dir / "robustness_report.html"
    path.write_text(html, encoding="utf-8")
    return path


# =============================================================================
# Private helpers
# =============================================================================

def _load_cnn_model(version_id: str, artifact_dir: Path):
    """Load the trained PyTorch CNN model artifact.

    Raises:
        FileNotFoundError: If model.pt is not found.
    """
    model_dir = Path(artifact_dir) / version_id / "model"
    pt_path = model_dir / "model.pt"
    if not pt_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_dir}'. "
            "Run the training stage before robustness analysis. "
            "Expected: model.pt (image_classification_cnn)."
        )
    import torch
    logger.debug("  Loading PyTorch model from %s", pt_path)
    return torch.load(pt_path, weights_only=False, map_location="cpu")


def _evaluate_augmentation(
    name: str,
    description: str,
    aug_fn,
    X_val: np.ndarray,
    y_true: np.ndarray,
    model,
    baseline_metrics: dict | None,
    thresholds: dict[str, float],
) -> dict | None:
    """Apply one augmentation and evaluate the model on the result.

    Args:
        name:             Augmentation scenario name.
        description:      Human-readable description.
        aug_fn:           Callable (X: ndarray) → augmented X.
        X_val:            Validation images (N, H, W, C), label=-1 already filtered.
        y_true:           Ground-truth labels.
        model:            Trained PyTorch CNN.
        baseline_metrics: Val-split baseline from evaluation_report.json.
        thresholds:       Sensitivity classification thresholds.

    Returns:
        Scenario result dict, or None if evaluation fails.
    """
    try:
        X_aug = aug_fn(X_val)

        # NHWC → NCHW for PyTorch CNN
        if X_aug.ndim == 4:
            X_aug = X_aug.transpose(0, 3, 1, 2)

        y_pred = model.predict(X_aug)
        metrics = _compute_metrics(y_true, y_pred)

    except Exception as e:
        logger.error("  Augmentation '%s' failed: %s", name, e, exc_info=True)
        return None

    delta: dict[str, float] = {}
    if baseline_metrics:
        for key in ("accuracy", "f1_score"):
            if key in metrics and key in baseline_metrics:
                delta[key] = round(metrics[key] - baseline_metrics[key], 4)

    acc_drop = abs(delta.get("accuracy", 0.0))
    sensitivity = _classify_sensitivity(acc_drop, thresholds)

    return {
        "name":        name,
        "description": description,
        "metrics":     metrics,
        "delta":       delta,
        "sensitivity": sensitivity,
    }


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_score": round(
            float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), 4
        ),
    }


def _classify_sensitivity(accuracy_drop: float, thresholds: dict[str, float]) -> str:
    if accuracy_drop >= thresholds.get("high", DEFAULT_ROBUSTNESS_THRESHOLDS["high"]):
        return "high"
    if accuracy_drop >= thresholds.get("medium", DEFAULT_ROBUSTNESS_THRESHOLDS["medium"]):
        return "medium"
    return "low"


def _render_html_report(report: dict) -> str:
    """Render the robustness report as a self-contained HTML page."""
    generated_at = report.get("generated_at", "")[:19].replace("T", " ")
    version_id   = report.get("version_id", "")
    eval_split   = report.get("evaluation_split", _EVALUATION_SPLIT)
    baseline     = report.get("baseline_metrics") or {}
    scenarios    = report.get("scenarios", [])
    most_sens    = report.get("most_sensitive_augmentation", "—")
    least_sens   = report.get("least_sensitive_augmentation", "—")
    thresholds   = report.get("robustness_thresholds", DEFAULT_ROBUSTNESS_THRESHOLDS)
    total_imgs   = report.get("total_images", "?")

    high_pct   = int(thresholds.get("high", 0.15) * 100)
    medium_pct = int(thresholds.get("medium", 0.05) * 100)

    _SEVERITY_COLORS = {"high": "#d9534f", "medium": "#f0ad4e", "low": "#5cb85c"}

    def _fmt(val) -> str:
        return f"{val:.4f}" if isinstance(val, float) else "N/A"

    def _fmt_delta(delta: dict, key: str) -> str:
        v = delta.get(key)
        if v is None:
            return "N/A"
        return f"{'+'if v>=0 else ''}{v:.4f}"

    rows = ""
    for s in scenarios:
        sev = s.get("sensitivity", "low")
        color = _SEVERITY_COLORS.get(sev, "#888")
        rows += f"""
        <tr>
          <td><strong>{s['name']}</strong></td>
          <td style="font-size:0.85em;color:#555">{s.get('description','')}</td>
          <td>{_fmt(s['metrics'].get('accuracy'))}</td>
          <td>{_fmt(s['metrics'].get('f1_score'))}</td>
          <td>{_fmt_delta(s.get('delta',{}), 'accuracy')}</td>
          <td>{_fmt_delta(s.get('delta',{}), 'f1_score')}</td>
          <td style="color:{color};font-weight:bold">{sev.upper()}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Augmentation Robustness Report — {version_id[:8]}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 2em; color: #333; }}
    h1   {{ color: #2c3e50; }}
    .meta  {{ background:#f8f9fa; padding:1em; border-radius:4px; margin-bottom:1.5em; }}
    .meta span {{ margin-right:2em; }}
    .notice {{ background:#fff3cd; border-left:4px solid #f0ad4e; padding:0.8em 1.2em;
               border-radius:4px; margin-bottom:1.5em; }}
    .summary {{ background:#eef2ff; padding:0.8em 1.2em; border-left:4px solid #4a6cf7;
                border-radius:4px; margin-bottom:1.5em; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border:1px solid #ddd; padding:8px 12px; text-align:left; }}
    th {{ background:#2c3e50; color:white; }}
    tr:nth-child(even) {{ background:#f8f9fa; }}
    tr:hover {{ background:#eef2ff; }}
    .note {{ color:#888; font-size:0.85em; margin-top:1em; }}
  </style>
</head>
<body>
  <h1>Image Augmentation Robustness Report</h1>

  <div class="notice">
    <strong>Note:</strong> This report measures model sensitivity to image augmentations
    (geometric and photometric transformations). It is <strong>NOT</strong> a drift detection
    report. It does not indicate whether production data has shifted from training data.
    For actual drift detection, use <code>monitor-drift-image</code> with real production batches.
    <br><em>For raw-image (DNG) pipelines, see the ISP sensitivity report (sensitivity_report.html).</em>
  </div>

  <div class="meta">
    <span><strong>Task type:</strong> image_classification_cnn</span>
    <span><strong>Version:</strong> {version_id[:12]}</span>
    <span><strong>Evaluation split:</strong> {eval_split}</span>
    <span><strong>Generated:</strong> {generated_at} UTC</span>
    <span><strong>Images evaluated:</strong> {total_imgs}</span>
  </div>

  <div class="summary">
    <strong>Baseline ({eval_split} split)</strong> — accuracy: {_fmt(baseline.get('accuracy'))},
    F1: {_fmt(baseline.get('f1_score'))}<br>
    <strong>Most sensitive augmentation:</strong> {most_sens} &nbsp;|&nbsp;
    <strong>Least sensitive:</strong> {least_sens}
  </div>

  <h2>Augmentation Scenario Results</h2>
  <p class="note">
    Sensitivity classified by absolute accuracy drop vs baseline {eval_split} split:
    HIGH ≥ {high_pct} pp &nbsp;|&nbsp; MEDIUM ≥ {medium_pct} pp &nbsp;|&nbsp;
    LOW &lt; {medium_pct} pp.
  </p>

  <table>
    <thead>
      <tr>
        <th>Augmentation</th>
        <th>Description</th>
        <th>Accuracy</th>
        <th>F1 Score</th>
        <th>Δ Accuracy</th>
        <th>Δ F1</th>
        <th>Sensitivity</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>

  <p class="note">
    Report type: augmentation-based robustness (standard JPG/PNG pipeline, raw_input=false).<br>
    This analysis applies the same 8 augmentations to all images in the {eval_split} split.
    Augmentations are deterministic (fixed numpy seed for noise scenarios).
  </p>
</body>
</html>
"""
