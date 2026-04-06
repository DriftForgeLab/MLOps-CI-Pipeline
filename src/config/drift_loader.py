"""Drift config loader — load and return a DriftConfig."""

from pathlib import Path

from src.config.schema import (
    VALID_DRIFT_SEVERITIES,
    VALID_DRIFT_STATTESTS,
    VALID_REFERENCE_SOURCES,
    DriftStatTestConfig,
    DriftStatTestThresholdConfig,
    DriftSeverityConfig,
    DriftFeatureSeverityConfig,
    DriftMonitoringConfig,
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
    )
