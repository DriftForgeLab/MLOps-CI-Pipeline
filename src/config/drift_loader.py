"""Drift config loader — load and return a DriftConfig."""

from pathlib import Path

from src.config.schema import (
    VALID_DRIFT_FAIL_ON_SEVERITIES,
    VALID_DRIFT_SEVERITIES,
    VALID_DRIFT_STATTESTS,
    VALID_REFERENCE_SOURCES,
    DriftStatTestConfig,
    DriftStatTestThresholdConfig,
    DriftSeverityConfig,
    DriftFeatureSeverityConfig,
    DriftMonitoringConfig,
    DriftImageSeverityConfig,
    DriftImageConfig,
    DriftConfig,
)
from src.config.validation import _load_yaml


def load_drift_config(path: Path) -> DriftConfig:
    """
    Load, validate, and return a DriftConfig from a YAML file.

    Raises:
        FileNotFoundError: Config file does not exist.
        ValueError: Config file is empty, malformed, or fails validation.
    """
    raw = _load_yaml(path)
    drift = raw.get("drift", {})
    errors: list[str] = []

    # Validate enum fields
    stattest = drift.get("stattest", {})
    if stattest.get("numerical", "ks") not in VALID_DRIFT_STATTESTS:
        errors.append(f"Invalid numerical stattest: '{stattest['numerical']}'")
    if stattest.get("categorical", "chisquare") not in VALID_DRIFT_STATTESTS:
        errors.append(f"Invalid categorical stattest: '{stattest['categorical']}'")

    ref_source = drift.get("reference_source", "train")
    if ref_source not in VALID_REFERENCE_SOURCES:
        errors.append(f"Invalid reference_source: '{ref_source}'")

    # Validate severity references in monitoring config
    monitoring_cfg = drift.get("monitoring", {})
    alert_sev = monitoring_cfg.get("alert_severity", "medium")
    if alert_sev not in VALID_DRIFT_SEVERITIES:
        errors.append(f"Invalid monitoring.alert_severity: '{alert_sev}'")

    fail_on_sev = monitoring_cfg.get("fail_on_severity", "high")
    if fail_on_sev not in VALID_DRIFT_FAIL_ON_SEVERITIES:
        errors.append(
            f"Invalid monitoring.fail_on_severity: '{fail_on_sev}' "
            f"(expected one of {sorted(VALID_DRIFT_FAIL_ON_SEVERITIES)})"
        )

    # Validate image severity thresholds (medium < high, both > 0)
    image_cfg = drift.get("image", {}) or {}
    image_sev_cfg = image_cfg.get("severity", {}) or {}
    image_med = image_sev_cfg.get("medium", DriftImageSeverityConfig.medium)
    image_high = image_sev_cfg.get("high", DriftImageSeverityConfig.high)
    if not (isinstance(image_med, (int, float)) and image_med > 0):
        errors.append(f"image.severity.medium must be a positive number, got {image_med!r}")
    if not (isinstance(image_high, (int, float)) and image_high > 0):
        errors.append(f"image.severity.high must be a positive number, got {image_high!r}")
    if (
        isinstance(image_med, (int, float))
        and isinstance(image_high, (int, float))
        and image_med >= image_high
    ):
        errors.append(
            f"image.severity.medium ({image_med}) must be less than image.severity.high ({image_high})"
        )

    if errors:
        raise ValueError(
            "Drift config validation failed:\n  - " + "\n  - ".join(errors)
        )

    # Build frozen dataclass hierarchy
    return DriftConfig(
        enabled=drift.get("enabled", True),
        reference_source=ref_source,
        stattest=DriftStatTestConfig(**drift.get("stattest", {})),
        stattest_threshold=DriftStatTestThresholdConfig(
            **drift.get("stattest_threshold", {})
        ),
        drift_share=drift.get("drift_share", 0.5),
        severity=DriftSeverityConfig(**drift.get("severity", {})),
        feature_severity=DriftFeatureSeverityConfig(
            **drift.get("feature_severity", {})
        ),
        monitoring=DriftMonitoringConfig(**drift.get("monitoring", {})),
        image=DriftImageConfig(
            severity=DriftImageSeverityConfig(**image_sev_cfg),
        ),
    )
