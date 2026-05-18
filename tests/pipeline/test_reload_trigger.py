"""Tests for the opt-in pipeline-side API reload trigger."""

from unittest.mock import patch
from urllib.error import URLError

from src.config.schema import DeploymentConfig, ReloadConfig
from src.pipeline.run_pipeline import trigger_api_reload


def _deploy_config(*, enabled, url="http://localhost:8000/admin/reload", timeout=5):
    return DeploymentConfig(reload=ReloadConfig(enabled=enabled, url=url, timeout_seconds=timeout))


def test_noop_when_reload_disabled():
    """When reload.enabled is False, no HTTP request is made."""
    with patch("src.pipeline.run_pipeline.urlopen") as mock_urlopen:
        trigger_api_reload(_deploy_config(enabled=False))
    mock_urlopen.assert_not_called()


def test_posts_to_reload_url_when_enabled():
    """When enabled, the trigger POSTs to the configured reload URL."""
    cfg = _deploy_config(enabled=True, url="http://api.internal:9000/admin/reload")
    with patch("src.pipeline.run_pipeline.urlopen") as mock_urlopen:
        trigger_api_reload(cfg)
    mock_urlopen.assert_called_once()
    req = mock_urlopen.call_args[0][0]
    assert req.full_url == "http://api.internal:9000/admin/reload"
    assert req.get_method() == "POST"


def test_sends_admin_token_header_when_env_set(monkeypatch):
    """The X-Admin-Token header is attached when API_ADMIN_TOKEN is set."""
    monkeypatch.setenv("API_ADMIN_TOKEN", "s3cret")
    with patch("src.pipeline.run_pipeline.urlopen") as mock_urlopen:
        trigger_api_reload(_deploy_config(enabled=True))
    req = mock_urlopen.call_args[0][0]
    assert req.get_header("X-admin-token") == "s3cret"


def test_no_token_header_when_env_unset(monkeypatch):
    monkeypatch.delenv("API_ADMIN_TOKEN", raising=False)
    with patch("src.pipeline.run_pipeline.urlopen") as mock_urlopen:
        trigger_api_reload(_deploy_config(enabled=True))
    req = mock_urlopen.call_args[0][0]
    assert req.get_header("X-admin-token") is None


def test_ci_safe_when_api_unreachable():
    """A connection failure is logged and swallowed — the trigger never raises."""
    with patch(
        "src.pipeline.run_pipeline.urlopen",
        side_effect=URLError("connection refused"),
    ):
        # Must not raise.
        trigger_api_reload(_deploy_config(enabled=True))
