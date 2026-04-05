import pytest
from pathlib import Path

from src.config.loader import load_drift_config, DriftConfig
from src.config.schema import (
    DriftStatTestConfig,
    DriftStatTestThresholdConfig,
    DriftSeverityConfig,
    DriftFeatureSeverityConfig,
    DriftRecommendationConfig,
    DriftPipelineConfig,
    DriftMonitoringConfig,
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


def test_load_drift_config_recommendation_values():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    assert config.recommendations.retrain_min_severity == "high"
    assert config.recommendations.retrain_min_drift_share == 0.50
    assert config.recommendations.collect_data_min_severity == "medium"
    assert config.recommendations.collect_data_min_drift_share == 0.25


def test_load_drift_config_pipeline_values():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    assert config.pipeline.block_on_severity == "high"
    assert config.pipeline.require_approval_on_drift is True


def test_load_drift_config_monitoring_values():
    config = load_drift_config(CONFIG_DIR / "drift.yaml")
    assert config.monitoring.enabled is True
    assert config.monitoring.min_batch_size == 30
    assert config.monitoring.alert_severity == "medium"


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


def test_load_drift_config_invalid_block_on_severity(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  pipeline:\n"
        "    block_on_severity: critical\n"
    )
    with pytest.raises(ValueError, match="Invalid pipeline.block_on_severity"):
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


def test_load_drift_config_invalid_recommendation_severity(tmp_path):
    cfg = tmp_path / "drift.yaml"
    cfg.write_text(
        "drift:\n"
        "  recommendations:\n"
        "    retrain_min_severity: critical\n"
    )
    with pytest.raises(ValueError, match="Invalid recommendations.retrain_min_severity"):
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

def test_subconfigpaths_default_drift():
    from src.config.schema import SubConfigPaths
    paths = SubConfigPaths(
        preprocessing="a", training="b", evaluation="c",
        promotion="d", deployment="e",
    )
    assert paths.drift == "src/config/drift.yaml"


def test_subconfigpaths_custom_drift():
    from src.config.schema import SubConfigPaths
    paths = SubConfigPaths(
        preprocessing="a", training="b", evaluation="c",
        promotion="d", deployment="e", drift="custom/drift.yaml",
    )
    assert paths.drift == "custom/drift.yaml"


# ---------------------------------------------------------------------------
# VALID_PIPELINE_STAGES includes "drift"
# ---------------------------------------------------------------------------

def test_drift_in_valid_pipeline_stages():
    from src.config.schema import VALID_PIPELINE_STAGES
    assert "drift" in VALID_PIPELINE_STAGES
