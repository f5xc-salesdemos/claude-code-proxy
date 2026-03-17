"""Unit tests for ModelRegistry."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from src.core.model_registry import ModelLimits, ModelRegistry


def _make_config() -> MagicMock:
    """Create a minimal mock Config."""
    return MagicMock()


class TestModelRegistryDefaults:
    """Test default hardcoded model limits."""

    def test_known_model_limits(self):
        """claude-opus-4-6 returns ModelLimits(1000000, 128000)."""
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("claude-opus-4-6")
        assert limits == ModelLimits(
            max_input_tokens=1_000_000, max_output_tokens=128_000
        )

    def test_unknown_model_returns_none(self):
        """Unknown model name returns None."""
        registry = ModelRegistry(_make_config())
        assert registry.get_limits("nonexistent-xyz") is None

    def test_all_claude_defaults(self):
        """All hardcoded model names resolve to non-None limits."""
        registry = ModelRegistry(_make_config())
        expected_models = [
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-3-7-sonnet-20250219",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
        for model in expected_models:
            limits = registry.get_limits(model)
            assert limits is not None, f"Expected limits for {model!r}, got None"
            assert isinstance(limits, ModelLimits)


def _mock_response(
    status_code: int = 200, json_data: object = None, raise_on_json: bool = False
):
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if raise_on_json:
        resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
    else:
        resp.json.return_value = json_data
    return resp


class TestModelRegistryDiscovery:
    """Test async upstream model discovery."""

    @pytest.mark.asyncio
    async def test_discovery_valid_response(self):
        """Valid upstream response updates registry limits."""
        registry = ModelRegistry(_make_config())

        mock_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "model_group": "claude-opus-4-6",
                        "max_input_tokens": 500_000.0,
                        "max_output_tokens": 64_000.0,
                    },
                    {
                        "model_group": "new-upstream-model",
                        "max_input_tokens": 300_000.0,
                        "max_output_tokens": 16_000.0,
                    },
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        registry._client = mock_client

        await registry.discover_from_upstream(
            "https://f5ai.pd.f5net.com/api/v1", "test-key"
        )

        # Existing model updated
        limits = registry.get_limits("claude-opus-4-6")
        assert limits == ModelLimits(500_000, 64_000)

        # New model added
        limits = registry.get_limits("new-upstream-model")
        assert limits == ModelLimits(300_000, 16_000)

        # Verify URL construction stripped /v1
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://f5ai.pd.f5net.com/api/model_group/info"

    @pytest.mark.asyncio
    async def test_discovery_404_fallback(self):
        """404 response leaves defaults unchanged."""
        registry = ModelRegistry(_make_config())
        original_limits = registry.get_all_models()

        mock_resp = _mock_response(status_code=404)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        registry._client = mock_client

        await registry.discover_from_upstream("https://example.com/api/v1", "test-key")

        assert registry.get_all_models() == original_limits

    @pytest.mark.asyncio
    async def test_discovery_malformed_json(self):
        """JSON decode error leaves defaults unchanged."""
        registry = ModelRegistry(_make_config())
        original_limits = registry.get_all_models()

        mock_resp = _mock_response(status_code=200, raise_on_json=True)

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        registry._client = mock_client

        await registry.discover_from_upstream("https://example.com/api/v1", "test-key")

        assert registry.get_all_models() == original_limits

    @pytest.mark.asyncio
    async def test_discovery_null_fields(self):
        """Entries with null max_input_tokens preserve hardcoded defaults."""
        registry = ModelRegistry(_make_config())
        original_opus = registry.get_limits("claude-opus-4-6")

        mock_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "model_group": "claude-opus-4-6",
                        "max_input_tokens": None,
                        "max_output_tokens": 64_000.0,
                    },
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        registry._client = mock_client

        await registry.discover_from_upstream("https://example.com/api", "test-key")

        # Original limits preserved since max_input_tokens was null
        assert registry.get_limits("claude-opus-4-6") == original_opus


class TestModelRegistryEnvOverrides:
    """Test environment variable overrides for model limits."""

    def test_env_override_wins(self, monkeypatch):
        """MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6 overrides default."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6", "500000")
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 500000

    def test_no_env_override_fallthrough(self, monkeypatch):
        """Without env var, hardcoded default is used."""
        monkeypatch.delenv("MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6", raising=False)
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 1_000_000
