# =============================================================================
# src/monitoring/image_drift_monitor.py — Online image batch drift orchestration
# =============================================================================
# Orchestrates image drift detection: checks batch size, dispatches to the
# appropriate computation method, assembles the result dict, and handles
# scenario matching for raw-ISP pipelines.
#
# Core computation (Wasserstein, MMD, severity classification) lives in
# src/drift/image_compute.py — mirroring how tabular drift computation lives
# in src/drift/compute.py.
#
# Two detection methods:
#   statistical (default) — per-channel Wasserstein distance. Fast, no model.
#   embedding             — CNN feature MMD. Requires a PyTorch SimpleCNN.
#
# Scenario matching (raw-ISP pipelines only):
#   Compares the batch's per-channel Wasserstein profile against pre-computed
#   signatures for known ISP scenarios. The closest match provides a
#   human-readable interpretation of the observed drift.
# =============================================================================

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config.schema import DriftConfig
from src.drift.interpret import _SEVERITY_ORD
from src.drift.image_compute import (
    DEFAULT_IMAGE_SEVERITY_THRESHOLDS,
    classify_image_severity,
    compute_statistical_drift,
    compute_embedding_drift,
)

logger = logging.getLogger(__name__)

# Maximum match_distance (Wasserstein-space) for a scenario match to be reported.
_SCENARIO_MATCH_DISTANCE_LIMIT = 0.20


# =============================================================================
# Public API
# =============================================================================

def monitor_image_batch(
    batch: np.ndarray,
    reference: np.ndarray,
    drift_config: DriftConfig,
    method: str = "statistical",
    model=None,
    task_type: str = "image_classification_cnn",
    drift_scenarios_dir: Path | None = None,
    sensitivity_report_path: Path | None = None,
    severity_thresholds: dict[str, float] | None = None,
) -> dict | None:
    """Detect drift in an incoming image batch by comparing it to the reference.

    Args:
        batch:                   Incoming images, shape (N, H, W, C) or (N, H, W),
                                 float64, preprocessed with the same pipeline as
                                 the reference (same normalization stats, same scale).
        reference:               Reference images from the training split.
        drift_config:            Drift config dataclass (min_batch_size, alert_severity).
        method:                  "statistical" (Wasserstein, default) or "embedding" (MMD).
        model:                   PyTorch CNN. Required when method="embedding".
        task_type:               Pipeline task type (only image_classification_cnn supported).
        drift_scenarios_dir:     If provided, attempts ISP scenario matching.
                                 Only meaningful for raw-image pipelines.
        sensitivity_report_path: Path to sensitivity_report.json. Used to look up
                                 estimated accuracy drop for matched scenarios.
        severity_thresholds:     Dict with "medium" and "high" Wasserstein thresholds.
                                 Defaults to DEFAULT_IMAGE_SEVERITY_THRESHOLDS.

    Returns:
        Drift result dict, or None if batch is below min_batch_size.
        Always contains "scenario_match"; None when method="embedding" or
        drift_scenarios_dir is not provided.

    Raises:
        ValueError: If method is unsupported, or method="embedding" and model is None.
    """
    if method not in ("statistical", "embedding"):
        raise ValueError(
            f"Unsupported drift detection method: '{method}'. "
            "Must be 'statistical' or 'embedding'."
        )
    if method == "embedding" and model is None:
        raise ValueError(
            "method='embedding' requires a trained model. "
            "Pass the loaded PyTorch CNN via the model= argument."
        )

    min_size = drift_config.monitoring.min_batch_size
    if len(batch) < min_size:
        logger.warning(
            "Image batch has %d images (< min_batch_size=%d) — skipping drift analysis.",
            len(batch), min_size,
        )
        return None

    thresholds = severity_thresholds or DEFAULT_IMAGE_SEVERITY_THRESHOLDS

    # --- Compute drift score ---
    if method == "statistical":
        drift_type = "image_statistical"
        channel_scores, overall_score = compute_statistical_drift(batch, reference)
    else:
        drift_type = "image_embedding"
        overall_score = compute_embedding_drift(batch, reference, model, task_type)
        channel_scores = {}

    overall_severity = classify_image_severity(overall_score, thresholds)
    dataset_drift_detected = overall_severity != "low"

    # --- Scenario matching (statistical method only; raw-ISP pipelines) ---
    # When method="embedding", scenario_match is always None — the embedding
    # method produces a single MMD scalar with no per-channel breakdown to match.
    scenario_match: dict | None = None
    if method == "statistical" and drift_scenarios_dir is not None:
        scenario_match = match_to_scenarios(
            batch_channel_scores=channel_scores,
            reference=reference,
            drift_scenarios_dir=Path(drift_scenarios_dir),
            sensitivity_report_path=sensitivity_report_path,
            thresholds=thresholds,
        )

    # --- Assemble result ---
    result: dict = {
        "schema_version": "1.0.0",
        "drift_type": drift_type,
        "method": method,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task_type": task_type,
        "reference": {"size": int(len(reference)), "source": "train"},
        "batch": {"size": int(len(batch))},
        "overall": {
            "drift_score": round(overall_score, 6),
            "dataset_drift_detected": dataset_drift_detected,
            "severity": overall_severity,
        },
        # scenario_interpretation: plausible explanation of observed drift
        # for raw-ISP pipelines. None when method="embedding" or no
        # drift_scenarios_dir was provided. This is interpretation, not
        # the detection itself — detection is always the reference vs batch comparison.
        "scenario_interpretation": scenario_match,
    }

    if channel_scores:
        result["channels"] = {
            ch: {
                "drift_score": round(score, 6),
                "severity": classify_image_severity(score, thresholds),
            }
            for ch, score in channel_scores.items()
        }

    # --- Log alert ---
    alert_severity = drift_config.monitoring.alert_severity
    if _SEVERITY_ORD.get(overall_severity, 0) >= _SEVERITY_ORD.get(alert_severity, 0):
        interpretation_info = ""
        if scenario_match:
            interpretation_info = (
                f" Closest ISP scenario: '{scenario_match['matched_scenario']}' "
                f"(estimated accuracy drop: {scenario_match.get('estimated_accuracy_drop', 'unknown')})"
            )
        logger.warning(
            "Drift alert: severity=%s (≥ threshold %s), "
            "drift_score=%.4f, batch=%d images.%s",
            overall_severity, alert_severity, overall_score, len(batch), interpretation_info,
        )

    return result


def match_to_scenarios(
    batch_channel_scores: dict[str, float],
    reference: np.ndarray,
    drift_scenarios_dir: Path,
    sensitivity_report_path: Path | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict | None:
    """Find the ISP scenario whose drift signature is closest to the observed batch drift.

    This provides an INTERPRETATION of detected drift, not additional detection.
    The drift itself has already been measured (Wasserstein distance). This function
    asks: "Which known ISP parameter change does the observed drift pattern most
    resemble?" A match gives a plausible physical explanation (e.g. "warm white
    balance shift"), but cannot confirm causation.

    Args:
        batch_channel_scores:    Channel-wise Wasserstein scores for the batch.
        reference:               Reference images (training split).
        drift_scenarios_dir:     Root directory for drift scenario NPZ files.
        sensitivity_report_path: Path to sensitivity_report.json (from ISP simulation).
        thresholds:              Severity thresholds (for logging only).

    Returns:
        Interpretation dict with keys: matched_scenario, description, match_distance,
        confidence, estimated_accuracy_drop. None if no match within limit or
        if no scenario signatures are available.
    """
    drift_scenarios_dir = Path(drift_scenarios_dir)
    thresholds = thresholds or DEFAULT_IMAGE_SEVERITY_THRESHOLDS

    accuracy_drops: dict[str, float | None] = {}
    descriptions: dict[str, str] = {}
    if sensitivity_report_path and Path(sensitivity_report_path).exists():
        with open(sensitivity_report_path) as f:
            sens_report = json.load(f)
        for s in sens_report.get("scenarios", []):
            name = s["name"]
            accuracy_drops[name] = s.get("delta", {}).get("accuracy")
            descriptions[name] = s.get("description", "")

    scenario_signatures: dict[str, dict[str, float]] = {}
    for scenario_dir in sorted(drift_scenarios_dir.iterdir()):
        if not scenario_dir.is_dir():
            continue
        meta_path = scenario_dir / "scenario_metadata.json"
        val_npz = scenario_dir / "val.npz"
        if not (meta_path.exists() and val_npz.exists()):
            continue

        with open(meta_path) as f:
            meta = json.load(f)
        scenario_name = meta.get("scenario_name", scenario_dir.name)

        if scenario_name not in descriptions:
            descriptions[scenario_name] = meta.get("description", "")

        try:
            scenario_data = np.load(val_npz)
            scenario_images = scenario_data["X"]
            sig, _ = compute_statistical_drift(scenario_images, reference)
            scenario_signatures[scenario_name] = sig
        except Exception as e:
            logger.warning("Could not compute signature for scenario '%s': %s", scenario_name, e)

    if not scenario_signatures:
        logger.debug("No scenario signatures available for matching — skipping.")
        return None

    batch_channels = sorted(batch_channel_scores.keys())
    best_name: str | None = None
    best_distance = float("inf")

    for scenario_name, sig in scenario_signatures.items():
        shared = sorted(set(batch_channels) & set(sig.keys()))
        if not shared:
            continue
        dist = float(np.sqrt(sum(
            (batch_channel_scores[ch] - sig[ch]) ** 2
            for ch in shared
        )))
        if dist < best_distance:
            best_distance = dist
            best_name = scenario_name

    if best_name is None or best_distance > _SCENARIO_MATCH_DISTANCE_LIMIT:
        logger.info(
            "No scenario match within limit (best_distance=%.4f, limit=%.4f)",
            best_distance, _SCENARIO_MATCH_DISTANCE_LIMIT,
        )
        return None

    confidence = "high" if best_distance < 0.05 else "medium" if best_distance < 0.10 else "low"
    estimated_drop = accuracy_drops.get(best_name)

    logger.info(
        "Scenario match: '%s' (distance=%.4f, confidence=%s, est_accuracy_drop=%s)",
        best_name, best_distance, confidence,
        f"{estimated_drop:+.4f}" if estimated_drop is not None else "unknown",
    )

    return {
        "matched_scenario": best_name,
        "description": descriptions.get(best_name, ""),
        "match_distance": round(best_distance, 6),
        "confidence": confidence,
        "estimated_accuracy_drop": estimated_drop,
    }


def load_reference_images(
    dataset_name: str,
    version_id: str,
    processed_dir: Path = Path("data/processed"),
    split: str = "train",
) -> np.ndarray:
    """Load reference images (NPZ) from the preprocessed dataset directory.

    Args:
        dataset_name:  Dataset identifier.
        version_id:    Content-addressed version hash.
        processed_dir: Base directory for versioned preprocessed data.
        split:         Which split to load (default "train").

    Returns:
        Float64 image array (N, H, W, C).

    Raises:
        FileNotFoundError: If the NPZ file does not exist.
    """
    npz_path = (
        Path(processed_dir) / dataset_name / version_id / "preprocessed" / f"{split}.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Reference images not found at '{npz_path}'. "
            "Run the preprocessing stage before drift monitoring."
        )
    return np.load(npz_path)["X"]

