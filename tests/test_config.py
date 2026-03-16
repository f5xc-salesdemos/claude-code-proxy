"""Unit tests for Config (Pydantic BaseSettings)."""

import os

import pytest

from src.core.config import Config


class TestConfigDefaults:
    """Verify default field values when only required env vars are set."""

    def test_required_openai_api_key(self, monkeypatch):
        """Config requires OPENAI_API_KEY to be set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(Exception):  # noqa: B017 – ValidationError
            Config()

    def test_defaults_with_api_key(self, monkeypatch):
        """All optional fields have sensible defaults."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        # Clear optional env vars that may be set in .env
        for var in (
            "ANTHROPIC_API_KEY",
            "BIG_MODEL",
            "MIDDLE_MODEL",
            "SMALL_MODEL",
            "LOG_LEVEL",
            "PORT",
            "HOST",
            "MAX_TOKENS_LIMIT",
            "MIN_TOKENS_LIMIT",
            "REQUEST_TIMEOUT",
            "MAX_RETRIES",
        ):
            monkeypatch.delenv(var, raising=False)
        cfg = Config()
        assert cfg.openai_api_key == "test-key"
        assert cfg.port == 8082
        assert cfg.log_level == "INFO"
        assert cfg.big_model == "gpt-4o"
        assert cfg.small_model == "gpt-4o-mini"
        assert cfg.middle_model == "gpt-4o"  # defaults to big_model
        assert cfg.max_tokens_limit == 4096
        assert cfg.min_tokens_limit == 100
        assert cfg.request_timeout == 90
        assert cfg.max_retries == 2


class TestLogLevelValidation:
    """log_level field validator normalises and rejects bad values."""

    def test_valid_log_level(self, monkeypatch):
        """Lower-case input is normalised to upper case."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("LOG_LEVEL", "debug")
        cfg = Config()
        assert cfg.log_level == "DEBUG"

    def test_log_level_with_comment(self, monkeypatch):
        """Trailing comments after the level are stripped."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("LOG_LEVEL", "WARNING # be quiet")
        cfg = Config()
        assert cfg.log_level == "WARNING"

    def test_invalid_log_level_defaults_to_info(self, monkeypatch):
        """Unknown level strings fall back to INFO."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("LOG_LEVEL", "VERBOSE")
        cfg = Config()
        assert cfg.log_level == "INFO"


class TestMiddleModelDefault:
    """middle_model falls back to big_model when not set."""

    def test_middle_defaults_to_big(self, monkeypatch):
        """When MIDDLE_MODEL is unset, it inherits from BIG_MODEL."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("BIG_MODEL", "my-big")
        monkeypatch.delenv("MIDDLE_MODEL", raising=False)
        cfg = Config()
        assert cfg.middle_model == "my-big"

    def test_middle_overridden(self, monkeypatch):
        """Explicit MIDDLE_MODEL takes precedence over BIG_MODEL."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("BIG_MODEL", "my-big")
        monkeypatch.setenv("MIDDLE_MODEL", "my-mid")
        cfg = Config()
        assert cfg.middle_model == "my-mid"


class TestClientApiKeyValidation:
    """validate_client_api_key() behaviour."""

    def test_no_anthropic_key_accepts_all(self, monkeypatch):
        """Without ANTHROPIC_API_KEY, any client key is accepted."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = Config()
        assert cfg.validate_client_api_key("anything") is True

    def test_matching_key_accepted(self, monkeypatch):
        """Correct client key passes validation."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        cfg = Config()
        assert cfg.validate_client_api_key("secret") is True

    def test_wrong_key_rejected(self, monkeypatch):
        """Incorrect client key fails validation."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
        cfg = Config()
        assert cfg.validate_client_api_key("wrong") is False


class TestCustomHeaders:
    """get_custom_headers() scans CUSTOM_HEADER_* env vars."""

    def test_custom_headers(self, monkeypatch):
        """CUSTOM_HEADER_* env vars become HTTP headers with hyphens."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("CUSTOM_HEADER_X_MY_HEADER", "value1")
        monkeypatch.setenv("CUSTOM_HEADER_AUTH_TOKEN", "tok123")
        cfg = Config()
        headers = cfg.get_custom_headers()
        assert headers["X-MY-HEADER"] == "value1"
        assert headers["AUTH-TOKEN"] == "tok123"

    def test_no_custom_headers(self, monkeypatch):
        """No CUSTOM_HEADER_* vars returns empty dict."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        # Remove any CUSTOM_HEADER_ vars that may exist
        for key in list(os.environ):
            if key.startswith("CUSTOM_HEADER_"):
                monkeypatch.delenv(key, raising=False)
        cfg = Config()
        assert cfg.get_custom_headers() == {}
