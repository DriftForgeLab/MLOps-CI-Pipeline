import pytest
from pathlib import Path

from src.config.loader import load_drift_config, DriftConfig
from src.config.schema import (
    DriftStatTestConfig,
    DriftStatTestThresholdConfig,
    DriftSeverityConfig,
    DriftFeatureSeverityConfig,
    DriftMonitoringConfig,
    DriftImageSeverityConfig,
    DriftImageConfig,
)

CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "config"


# ---------------------------------------------------------------------------
# Happy path: load the real drift.yaml
# ---------------------------------------------------------------------------

def test_load_drift_config_valid():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    assert isinstance(config, DriftConfig)
    assert config.enabled is True
    assert config.reference_source == "train"
    assert config.drift_share == 0.5


def test_load_drift_config_stattest_values():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    assert config.stattest.numerical == "ks"
    assert config.stattest.categorical == "chisquare"
    assert config.stattest_threshold.numerical == 0.05
    assert config.stattest_threshold.categorical == 0.05


def test_load_drift_config_severity_values():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    assert config.severity.low_max == 0.25
    assert config.severity.medium_max == 0.50
    assert config.feature_severity.high_below == 0.001
    assert config.feature_severity.medium_below == 0.01


def test_load_drift_config_monitoring_values():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    assert config.monitoring.enabled is True
    assert config.monitoring.min_batch_size == 30
    assert config.monitoring.alert_severity == "medium"
    assert config.monitoring.fail_on_severity == "high"


def test_load_drift_config_is_frozen():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    with pytest.raises(AttributeError):
        config.enabled = False


# ---------------------------------------------------------------------------
# Defaults: minimal YAML with just the drift key
# ---------------------------------------------------------------------------

def test_load_drift_config_defaults(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text("drift:\n  enabled: true\n")
    config = load_drift_config(cfg)
    assert config == DriftConfig()


def test_load_drift_config_empty_drift_section(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text("drift: {}\n")
    config = load_drift_config(cfg)
    assert config == DriftConfig()


# ---------------------------------------------------------------------------
# Override individual fields
# ---------------------------------------------------------------------------

def test_load_drift_config_custom_stattest(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  stattest:\n"
        "    numerical: psi\n"
        "    categorical: jensenshannon\n"
    )
    config = load_drift_config(cfg)
    assert config.stattest.numerical == "psi"
    assert config.stattest.categorical == "jensenshannon"


def test_load_drift_config_disabled(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text("drift:\n  enabled: false\n")
    config = load_drift_config(cfg)
    assert config.enabled is False


def test_load_drift_config_custom_thresholds(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  drift_share: 0.3\n"
        "  severity:\n"
        "    low_max: 0.10\n"
        "    medium_max: 0.40\n"
    )
    config = load_drift_config(cfg)
    assert config.drift_share == 0.3
    assert config.severity.low_max == 0.10
    assert config.severity.medium_max == 0.40


# ---------------------------------------------------------------------------
# Image severity thresholds
# ---------------------------------------------------------------------------

def test_load_drift_config_image_severity_from_yaml():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    assert config.image.severity.medium == 0.10
    assert config.image.severity.high == 0.25


def test_load_drift_config_image_severity_defaults_when_absent(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text("drift:\n  enabled: true\n")
    config = load_drift_config(cfg)
    assert config.image == DriftImageConfig()
    assert config.image.severity == DriftImageSeverityConfig()
    assert config.image.severity.medium == 0.10
    assert config.image.severity.high == 0.25


def test_load_drift_config_image_severity_override(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  image:\n"
        "    severity:\n"
        "      medium: 0.05\n"
        "      high: 0.40\n"
    )
    config = load_drift_config(cfg)
    assert config.image.severity.medium == 0.05
    assert config.image.severity.high == 0.40


def test_load_drift_config_image_severity_partial_override(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  image:\n"
        "    severity:\n"
        "      medium: 0.08\n"
    )
    config = load_drift_config(cfg)
    assert config.image.severity.medium == 0.08
    assert config.image.severity.high == 0.25  # default preserved


def test_load_drift_config_image_severity_medium_not_less_than_high(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  image:\n"
        "    severity:\n"
        "      medium: 0.30\n"
        "      high: 0.25\n"
    )
    with pytest.raises(ValueError, match="image.severity.medium"):
        load_drift_config(cfg)


def test_load_drift_config_image_severity_negative_rejected(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  image:\n"
        "    severity:\n"
        "      medium: -0.05\n"
        "      high: 0.25\n"
    )
    with pytest.raises(ValueError, match="image.severity.medium"):
        load_drift_config(cfg)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_load_drift_config_invalid_numerical_stattest(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  stattest:\n"
        "    numerical: invalid_test\n"
    )
    with pytest.raises(ValueError, match="Invalid numerical stattest"):
        load_drift_config(cfg)


def test_load_drift_config_invalid_categorical_stattest(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  stattest:\n"
        "    categorical: bad_test\n"
    )
    with pytest.raises(ValueError, match="Invalid categorical stattest"):
        load_drift_config(cfg)


def test_load_drift_config_invalid_reference_source(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  reference_source: val\n"
    )
    with pytest.raises(ValueError, match="Invalid reference_source"):
        load_drift_config(cfg)


def test_load_drift_config_invalid_alert_severity(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  monitoring:\n"
        "    alert_severity: extreme\n"
    )
    with pytest.raises(ValueError, match="Invalid monitoring.alert_severity"):
        load_drift_config(cfg)


# ---------------------------------------------------------------------------
# fail_on_severity (CI gate)
# ---------------------------------------------------------------------------

def test_load_drift_config_fail_on_severity_default(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text("drift:\n  enabled: true\n")
    config = load_drift_config(cfg)
    assert config.monitoring.fail_on_severity == "high"


@pytest.mark.parametrize("level", ["low", "medium", "high", "never"])
def test_load_drift_config_fail_on_severity_valid(tmp_path, level):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  monitoring:\n"
        f"    fail_on_severity: {level}\n"
    )
    config = load_drift_config(cfg)
    assert config.monitoring.fail_on_severity == level


def test_load_drift_config_invalid_fail_on_severity(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  monitoring:\n"
        "    fail_on_severity: always\n"
    )
    with pytest.raises(ValueError, match="Invalid monitoring.fail_on_severity"):
        load_drift_config(cfg)


def test_load_drift_config_multiple_errors(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  reference_source: val\n"
        "  stattest:\n"
        "    numerical: bad\n"
    )
    with pytest.raises(ValueError, match="Drift config validation failed"):
        load_drift_config(cfg)


# ---------------------------------------------------------------------------
# File errors
# ---------------------------------------------------------------------------

def test_load_drift_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_drift_config(Path("nonexistent_drift.yaml"))


def test_load_drift_config_empty_file(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text("")
    with pytest.raises(ValueError, match="empty"):
        load_drift_config(cfg)


# ---------------------------------------------------------------------------
# Pipeline config integration: drift is optional in SubConfigPaths
# ---------------------------------------------------------------------------

def test_subconfigpaths_drift_is_required():
    from src.config.schema import SubConfigPaths
    with pytest.raises(TypeError):
        SubConfigPaths(
            preprocessing="a", training="b", evaluation="c",
            promotion="d", deployment="e",
        )


def test_subconfigpaths_custom_drift():
    from src.config.schema import SubConfigPaths
    paths = SubConfigPaths(
        preprocessing="a", training="b", evaluation="c",
        promotion="d", deployment="e", drift="custom/drift.yaml",
    )
    assert paths.drift == "custom/drift.yaml"


# ---------------------------------------------------------------------------
# VALID_PIPELINE_STAGES includes "model_analysis"
# ---------------------------------------------------------------------------

def test_model_analysis_in_valid_pipeline_stages():
    from src.config.schema import VALID_PIPELINE_STAGES
    assert "model_analysis" in VALID_PIPELINE_STAGES
