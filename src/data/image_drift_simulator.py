# =============================================================================
# src/data/image_drift_simulator.py — ISP-based drift scenario simulator
# =============================================================================
# Generates controlled drift scenarios by re-processing the same raw DNG
# images through modified ISP pipeline configurations.
#
# Each scenario applies a specific set of ISP parameter overrides to the
# baseline config, producing a new set of processed images that represent a
# controlled shift in the image processing chain. The resulting NPZ files
# can then be evaluated against the trained model to measure how
# sensitive the model is to each type of ISP change.
#
# Output structure (per scenario):
#   drift_scenarios/<dataset>/<version_id>/<scenario_name>/train.npz   — (N, H, W, C) float64
#   drift_scenarios/<dataset>/<version_id>/<scenario_name>/val.npz
#   drift_scenarios/<dataset>/<version_id>/<scenario_name>/test.npz
#   drift_scenarios/<dataset>/<version_id>/<scenario_name>/scenario_metadata.json
#
# Only used when raw_input: true. For JPG/PNG datasets, see the Albumentations
# robustness path which is a separate analysis type.
# =============================================================================

from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.common.io import atomic_write_json, atomic_write_npz
from src.config.schema import ISPConfig

logger = logging.getLogger(__name__)

# Default parameter bounds used when the drift config does not provide a
# parameter_bounds section. These match the ranges documented in
# isp_pipeline.py and preprocessing_raw_image.yaml.
_DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "white_balance.r_gain":   (0.5, 2.5),
    "white_balance.g_gain":   (0.5, 2.0),
    "white_balance.b_gain":   (0.5, 2.0),
    "denoising.strength":     (0.0, 1.0),
    "sharpening.radius":      (0.1, 5.0),
    "sharpening.amount":      (0.0, 3.0),
    "gamma_correction.gamma": (1.0, 2.8),
}


# =============================================================================
# Public API
# =============================================================================

def run_drift_simulation(
    dataset_name: str,
    version_id: str,
    baseline_isp_config: ISPConfig,
    target_size: tuple[int, int],
    color_mode: str,
    normalize: bool,
    scenarios: list[dict],
    parameter_bounds: dict[str, list[float]],
    drift_scenarios_dir: Path,
    processed_dir: Path = Path("data/processed"),
    expected_formats: list[str] | None = None,
) -> list[dict]:
    """Generate drift scenarios by re-processing raw DNG images with modified ISP configs.

    For each scenario defined in the drift config, the same DNG images that
    were used during training are re-run through the ISP pipeline with the
    scenario's parameter overrides applied. Normalization uses the same mean
    and std as the baseline so the model receives comparable input.

    Folder structure created under drift_scenarios_dir:
        <scenario_name>/train.npz
        <scenario_name>/val.npz
        <scenario_name>/test.npz
        <scenario_name>/scenario_metadata.json

    Args:
        dataset_name:        Dataset identifier (matches preprocessing dataset dir).
        version_id:          Content-addressed version hash from the pipeline run.
        baseline_isp_config: ISP config used during baseline preprocessing.
        target_size:         (height, width) to resize images after ISP.
        color_mode:          "rgb" or "grayscale".
        normalize:           Whether to apply z-score normalization using baseline stats.
        scenarios:           List of scenario dicts from drift config. Each dict:
                               "name"        (str, required)
                               "description" (str, optional)
                               "isp"         (dict of ISP overrides, required)
        parameter_bounds:    Flat mapping "section.param" → [min, max].
                             Used to clip out-of-range override values.
                             Defaults are applied for any missing entry.
        drift_scenarios_dir: Root output directory for scenario NPZ files.
        processed_dir:       Base directory for versioned preprocessed data.
        expected_formats:    Allowed file extensions (e.g. [".dng"]). None = any.

    Returns:
        List of scenario result dicts (one per scenario), each containing:
          name, output_dir, isp_overrides (effective, after bound enforcement),
          splits (list of processed split names), image_counts, generated_at.

    Raises:
        FileNotFoundError: If the versioned dataset or feature_map.json is missing.
        ValueError: If no scenarios are provided or images cannot be processed.
    """
    if not scenarios:
        logger.info("  No drift scenarios configured — skipping simulation.")
        return []

    version_dir = Path(processed_dir) / dataset_name / version_id
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Versioned dataset not found at '{version_dir}'. "
            "Run the preprocessing stage before drift simulation."
        )

    # Load normalization stats from baseline preprocessing
    feature_map_path = version_dir / "preprocessed" / "feature_map.json"
    if not feature_map_path.exists():
        raise FileNotFoundError(
            f"Baseline feature_map.json not found at '{feature_map_path}'. "
            "Run the preprocessing stage before drift simulation."
        )
    with open(feature_map_path) as f:
        feature_map = json.load(f)

    norm_stats = feature_map.get("normalization_stats", {}) or {}
    norm_mean: list[float] | None = norm_stats.get("mean")
    norm_std: list[float] | None = norm_stats.get("std")
    baseline_class_to_index: dict[str, int] = feature_map.get("class_to_index", {})

    # Load expected_formats from dataset.yaml if not provided
    if expected_formats is None:
        dataset_yaml = version_dir / "dataset.yaml"
        if dataset_yaml.exists():
            with open(dataset_yaml) as f:
                dataset_meta = yaml.safe_load(f)
            image_props = dataset_meta.get("image_properties", {}) or {}
            expected_formats = image_props.get("expected_formats")

    # Merge caller-supplied bounds with defaults (caller takes precedence)
    effective_bounds: dict[str, tuple[float, float]] = dict(_DEFAULT_BOUNDS)
    for key, val in parameter_bounds.items():
        if isinstance(val, (list, tuple)) and len(val) == 2:
            effective_bounds[key] = (float(val[0]), float(val[1]))

    drift_scenarios_dir = Path(drift_scenarios_dir)
    results: list[dict] = []

    for scenario in scenarios:
        name = scenario.get("name")
        if not name:
            logger.warning("Scenario missing 'name' field — skipping")
            continue

        description = scenario.get("description", "")
        raw_overrides = scenario.get("isp", {})

        # Enforce parameter bounds: raises ValueError if any value is out of range.
        # Fail explicitly — silent clipping would make scenario names misleading.
        _enforce_bounds(raw_overrides, effective_bounds, name)
        isp_overrides = raw_overrides

        # Build the modified ISP config for this scenario
        scenario_isp = _merge_isp_overrides(baseline_isp_config, isp_overrides)

        scenario_dir = drift_scenarios_dir / name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        logger.info("  Generating drift scenario '%s'...", name)

        splits_processed: list[str] = []
        image_counts: dict[str, int] = {}

        for split_name in ("train", "val", "test"):
            split_images_dir = version_dir / split_name / "images"
            if not split_images_dir.exists():
                logger.debug("    Split '%s' not found at %s — skipping", split_name, split_images_dir)
                continue

            try:
                X, labels = _process_split_dng(
                    split_images_dir=split_images_dir,
                    isp_config=scenario_isp,
                    target_size=target_size,
                    color_mode=color_mode,
                    normalize=normalize,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    expected_formats=expected_formats,
                )

                # Encode labels using baseline class mapping for consistency
                y = _encode_labels(labels, baseline_class_to_index, split_name, name)

                atomic_write_npz(scenario_dir / f"{split_name}.npz", X=X, y=y)
                splits_processed.append(split_name)
                image_counts[split_name] = len(X)
                logger.info("    %s: %d images → %s", split_name, len(X), scenario_dir / f"{split_name}.npz")

            except Exception as e:
                logger.error(
                    "    Failed to process split '%s' for scenario '%s': %s",
                    split_name, name, e, exc_info=True,
                )

        if not splits_processed:
            logger.error("  Scenario '%s': no splits processed — check image paths", name)
            continue

        # Save scenario metadata for traceability
        metadata = {
            "scenario_name": name,
            "description": description,
            "isp_overrides": isp_overrides,
            "splits_processed": splits_processed,
            "image_counts": image_counts,
            "dataset_name": dataset_name,
            "version_id": version_id,
            "target_size": list(target_size),
            "color_mode": color_mode,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(scenario_dir / "scenario_metadata.json", metadata)

        results.append({
            "name": name,
            "output_dir": str(scenario_dir),
            "isp_overrides": isp_overrides,
            "splits": splits_processed,
            "image_counts": image_counts,
            "generated_at": metadata["generated_at"],
        })
        logger.info(
            "  Scenario '%s' complete (%d splits) → %s",
            name, len(splits_processed), scenario_dir,
        )

    return results


def load_image_scenarios_from_drift_yaml(drift_config_path: Path) -> tuple[list[dict], dict]:
    """Read image drift scenarios and parameter bounds from drift.yaml.

    The drift.yaml `image:` section is not parsed by load_drift_config() (it
    only reads the typed DriftConfig fields). This function reads the raw YAML
    to extract the additional image-specific configuration.

    Expected YAML structure under `drift.image`:
        image:
          scenarios:
            - name: "wb_shift_warm"
              description: "..."
              isp:
                white_balance:
                  r_gain: 1.8
          parameter_bounds:
            white_balance.r_gain: [0.5, 2.5]

    Args:
        drift_config_path: Path to drift.yaml.

    Returns:
        Tuple of (scenarios_list, parameter_bounds_dict).
        Both are empty if the `image:` section is absent.
    """
    with open(drift_config_path) as f:
        raw = yaml.safe_load(f)

    drift_section = raw.get("drift", {})
    image_section = drift_section.get("image", {})

    scenarios = image_section.get("scenarios", []) or []
    bounds = image_section.get("parameter_bounds", {}) or {}

    return scenarios, bounds


# =============================================================================
# Private helpers
# =============================================================================

def _enforce_bounds(
    overrides: dict,
    bounds: dict[str, tuple[float, float]],
    scenario_name: str,
) -> None:
    """Raise ValueError if any ISP override value violates its configured bound.

    Fails explicitly rather than clipping silently. Silent clipping would mean
    a scenario named "wb_shift_extreme" is actually processed with different
    values than declared, making reports and sensitivity profiles misleading.
    Fixing the config is the right response to an out-of-bounds value.

    Collects all violations before raising so the user sees the full picture
    in one error (fail-all-at-once pattern, consistent with the rest of the
    config validation in this codebase).

    Args:
        overrides:     Nested dict of ISP overrides from the scenario config.
        bounds:        Flat mapping "section.param" → (min, max).
        scenario_name: Used in the error message.

    Raises:
        ValueError: If one or more override values are outside their bounds.
    """
    violations: list[str] = []

    for section_key, section_overrides in overrides.items():
        if not isinstance(section_overrides, dict):
            continue
        for param_key, value in section_overrides.items():
            if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            flat_key = f"{section_key}.{param_key}"
            if flat_key not in bounds:
                continue
            lo, hi = bounds[flat_key]
            if value < lo or value > hi:
                violations.append(
                    f"  {flat_key} = {value}  (allowed range: [{lo}, {hi}])"
                )

    if violations:
        raise ValueError(
            f"Scenario '{scenario_name}' has parameter(s) outside defined bounds:\n"
            + "\n".join(violations)
            + "\nFix the scenario definition in drift.yaml or adjust parameter_bounds."
        )


def _merge_isp_overrides(baseline_isp: ISPConfig, overrides: dict) -> ISPConfig:
    """Return a new ISPConfig with scenario overrides applied to the baseline.

    Performs a key-level merge within each ISP section. Only the parameters
    explicitly listed in the override dict are changed; all other parameters
    in that section — and all other sections entirely — retain their baseline
    values.

    Example: overrides = {"white_balance": {"r_gain": 1.8}} changes r_gain
    but leaves g_gain and b_gain from the baseline untouched.

    This works correctly for ISP config because its structure is exactly
    2 levels deep (section → param). There is no deeper nesting that could
    be accidentally collapsed by a non-recursive update.

    Implementation:
        dataclasses.asdict() → deep copy → per-section dict.update() →
        _build_isp_config() reconstructs the frozen ISPConfig dataclass.

    Args:
        baseline_isp: The ISP config from the preprocessing config (frozen dataclass).
        overrides:    Nested dict of ISP overrides. Top-level keys must be ISP
                      section names: "black_level_correction", "demosaicing",
                      "white_balance", "color_correction", "denoising",
                      "sharpening", "gamma_correction".

    Returns:
        New ISPConfig dataclass with overrides applied, baseline otherwise.
    """
    from src.config.preprocessing_loader import _build_isp_config

    # Start from a full mutable copy of the baseline
    merged = dataclasses.asdict(baseline_isp)  # already a deep copy

    for section, section_overrides in overrides.items():
        if section not in merged:
            logger.warning(
                "Unknown ISP section in scenario overrides: '%s' — skipping", section
            )
            continue
        if isinstance(section_overrides, dict) and isinstance(merged[section], dict):
            # Key-level merge: updates only the keys present in section_overrides.
            # All other keys in merged[section] keep their baseline values.
            merged[section].update(section_overrides)
        else:
            # Non-dict section value (e.g. a scalar being replaced entirely)
            merged[section] = section_overrides

    return _build_isp_config(merged)


def _process_split_dng(
    split_images_dir: Path,
    isp_config: ISPConfig,
    target_size: tuple[int, int],
    color_mode: str,
    normalize: bool,
    norm_mean: list[float] | None,
    norm_std: list[float] | None,
    expected_formats: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    """Process one split's DNG images through a modified ISP config.

    Mirrors image_preprocess._load_and_transform_raw_images but accepts an
    explicit ISP config (the scenario variant) and applies the baseline
    normalization stats directly, so the model's expected input distribution
    is preserved.

    Args:
        split_images_dir: Path to the split/images/ directory.
        isp_config:       Modified ISP config for this scenario.
        target_size:      (height, width) to resize images.
        color_mode:       "rgb" or "grayscale".
        normalize:        Whether z-score normalization should be applied.
        norm_mean:        Baseline channel-wise mean (on [0,1] scale). None = skip.
        norm_std:         Baseline channel-wise std. None = skip.
        expected_formats: Allowed file extensions. None = any.

    Returns:
        (X, labels) where:
          X      — float64 array (N, H, W, C) or (N, H, W), normalized if requested.
          labels — list of string class names, one per image.

    Raises:
        ImportError: If rawpy or scikit-image is not installed.
        ValueError:  If no readable images are found.
    """
    try:
        import rawpy
        from skimage.transform import resize as skimage_resize
    except ImportError as exc:
        raise ImportError(
            "rawpy and scikit-image are required for raw image processing. "
            "Install with: pip install rawpy scikit-image"
        ) from exc

    from src.data.image_utils import scan_image_folder
    from src.data.isp_pipeline import run_isp, read_camera_params

    entries = scan_image_folder(split_images_dir, expected_formats)
    if not entries:
        raise ValueError(f"No images found in {split_images_dir}")

    target_h, target_w = target_size
    arrays: list[np.ndarray] = []
    labels: list[str] = []

    for img_path, class_name in entries:
        try:
            with rawpy.imread(str(img_path)) as raw:
                raw_array = raw.raw_image_visible.copy().astype(np.float32)
            camera_params = read_camera_params(img_path)
            # run_isp returns float64 in [0, 1]
            rgb = run_isp(raw_array, isp_config, camera_params)

            if color_mode == "grayscale":
                luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                arr = skimage_resize(luma, (target_h, target_w), anti_aliasing=True)
            else:
                arr = skimage_resize(rgb, (target_h, target_w, 3), anti_aliasing=True)
            # arr is float64 in [0, 1] (skimage_resize preserves range for [0,1] input)

            arrays.append(arr)
            labels.append(class_name)
        except Exception as e:
            logger.warning("Skipping unreadable image '%s': %s", img_path, e)

    if not arrays:
        raise ValueError(f"No readable images processed from {split_images_dir}")

    X = np.stack(arrays)  # (N, H, W, C) or (N, H, W), float64, [0, 1]

    if normalize:
        if norm_mean is not None and norm_std is not None:
            mean = np.array(norm_mean, dtype=np.float64)
            std = np.array(norm_std, dtype=np.float64)
            std = np.where(std == 0, 1.0, std)
            X = (X - mean) / std
    # else: normalize=False → X stays in [0, 1].
    # ISP + skimage_resize produces float64 in [0, 1]. The training pipeline
    # with normalize=False also outputs [0, 1] (it divides by 255 but skips
    # z-score). No further scaling is needed here.

    return X, labels


def _encode_labels(
    labels: list[str],
    class_to_index: dict[str, int],
    split_name: str,
    scenario_name: str,
) -> np.ndarray:
    """Encode string class names to integer indices using the baseline mapping.

    Args:
        labels:         String class names, one per image.
        class_to_index: Mapping from class name to index (from baseline feature_map.json).
        split_name:     Used only in warning messages.
        scenario_name:  Used only in warning messages.

    Returns:
        int64 array of class indices, one per image.

    Raises:
        ValueError: If any label is not present in the baseline class mapping.
            The simulator uses the same raw images as training, so an unknown
            class means a definite bug in the dataset or config — not a
            recoverable edge case.
    """
    encoded: list[int] = []
    unknown: list[str] = []
    for label in labels:
        idx = class_to_index.get(label)
        if idx is None:
            unknown.append(label)
        else:
            encoded.append(idx)

    if unknown:
        unique_unknown = sorted(set(unknown))
        raise ValueError(
            f"Scenario '{scenario_name}' / split '{split_name}': "
            f"the following class(es) are not in the baseline class mapping: "
            f"{unique_unknown}. "
            "The simulator must use the same classes as the training dataset."
        )

    return np.array(encoded, dtype=np.int64)
