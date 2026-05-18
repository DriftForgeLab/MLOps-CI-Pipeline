import pytest
from pathlib import Path

from src.config.loader import load_deployment_config
from src.config.schema import DeploymentConfig, ReloadConfig

CONFIG_DIR = Path(__file__).parent.parent.parent / "src" / "config"


# ---------------------------------------------------------------------------
# Happy path: load the real deployment.yaml
# ---------------------------------------------------------------------------

def test_load_deployment_config_valid():
    config = load_deployment_config(CONFIG_DIR / "deployment.yaml")
    assert isinstance(config, DeploymentConfig)
    assert config.server.port == 8000
    assert config.model.allowed_stage == "Production"


# ---------------------------------------------------------------------------
# Reload block
# ---------------------------------------------------------------------------

def test_load_deployment_config_reload_defaults(tmp_path):
    """A deployment config with no reload block yields safe reload defaults."""
    cfg = tmp_path / "deployment.yaml"
    cfg.write_text("server:\n  port: 8000\n")
    config = load_deployment_config(cfg)
    assert isinstance(config.reload, ReloadConfig)
    assert config.reload.enabled is False
    assert config.reload.url == "http://localhost:8000/admin/reload"
    assert config.reload.timeout_seconds == 5


def test_load_deployment_config_reload_from_yaml(tmp_path):
    """An explicit reload block is parsed into ReloadConfig."""
    cfg = tmp_path / "deployment.yaml"
    cfg.write_text(
        "reload:\n"
        "  enabled: true\n"
        "  url: \"http://api.internal:9000/admin/reload\"\n"
        "  timeout_seconds: 12\n"
    )
    config = load_deployment_config(cfg)
    assert config.reload.enabled is True
    assert config.reload.url == "http://api.internal:9000/admin/reload"
    assert config.reload.timeout_seconds == 12


def test_load_deployment_config_reload_invalid_enabled(tmp_path):
    cfg = tmp_path / "deployment.yaml"
    cfg.write_text("reload:\n  enabled: \"yes\"\n")
    with pytest.raises(ValueError, match="reload.enabled"):
        load_deployment_config(cfg)


def test_load_deployment_config_reload_invalid_timeout(tmp_path):
    cfg = tmp_path / "deployment.yaml"
    cfg.write_text("reload:\n  timeout_seconds: 0\n")
    with pytest.raises(ValueError, match="reload.timeout_seconds"):
        load_deployment_config(cfg)


def test_load_deployment_config_real_yaml_has_reload(tmp_path):
    """The shipped deployment.yaml defaults reload to disabled."""
    config = load_deployment_config(CONFIG_DIR / "deployment.yaml")
    assert config.reload.enabled is False
