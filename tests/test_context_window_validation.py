"""Thorough validation tests for the context window management feature.

Covers gaps identified in the existing test suite:
- Config safety margin validator edge cases
- Model registry partial env overrides and new models via env vars
- Model registry non-integer env var handling
- Model registry discovery edge cases (zero/negative input tokens)
- Pre-flight validation boundary conditions (exact boundary, safety margin effects)
- Pre-flight with None model (request.model is None)
- Models enrichment when registry is None
- Token estimation with Unicode, large schemas, empty tools
- End-to-end integration through TestClient
"""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from src.core.config import Config
from src.core.model_registry import ModelLimits, ModelRegistry
from src.core.tokens import estimate_tokens

# ---------------------------------------------------------------------------
# Config: safety margin validator edge cases
# ---------------------------------------------------------------------------


class TestSafetyMarginValidator:
    """The _clamp_safety_margin validator must clamp to (0.0, 1.0]."""

    def test_zero_clamped_to_minimum(self, monkeypatch):
        """0.0 is clamped to 0.01."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("MODEL_REGISTRY_SAFETY_MARGIN", "0.0")
        cfg = Config()
        assert cfg.model_registry_safety_margin == 0.01

    def test_negative_clamped_to_minimum(self, monkeypatch):
        """Negative value is clamped to 0.01."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("MODEL_REGISTRY_SAFETY_MARGIN", "-0.5")
        cfg = Config()
        assert cfg.model_registry_safety_margin == 0.01

    def test_above_one_clamped_to_one(self, monkeypatch):
        """Values above 1.0 are clamped to 1.0."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("MODEL_REGISTRY_SAFETY_MARGIN", "1.5")
        cfg = Config()
        assert cfg.model_registry_safety_margin == 1.0

    def test_exactly_one_is_valid(self, monkeypatch):
        """1.0 is a valid safety margin (no clamping)."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("MODEL_REGISTRY_SAFETY_MARGIN", "1.0")
        cfg = Config()
        assert cfg.model_registry_safety_margin == 1.0

    def test_small_positive_accepted(self, monkeypatch):
        """A very small positive value like 0.01 is accepted."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("MODEL_REGISTRY_SAFETY_MARGIN", "0.01")
        cfg = Config()
        assert cfg.model_registry_safety_margin == 0.01

    def test_normal_value_passthrough(self, monkeypatch):
        """Normal values in range pass through unchanged."""
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("MODEL_REGISTRY_SAFETY_MARGIN", "0.8")
        cfg = Config()
        assert cfg.model_registry_safety_margin == 0.8


# ---------------------------------------------------------------------------
# Model Registry: partial env overrides
# ---------------------------------------------------------------------------


def _make_config() -> MagicMock:
    """Create a minimal mock Config."""
    return MagicMock()


class TestPartialEnvOverrides:
    """Test env var overrides that set only input OR output, not both."""

    def test_input_override_preserves_output(self, monkeypatch):
        """Setting MODEL_MAX_INPUT_TOKENS_* preserves existing output limit."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6", "500000")
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 500_000
        # Output should be preserved from the default (128_000)
        assert limits.max_output_tokens == 128_000

    def test_output_override_preserves_input(self, monkeypatch):
        """Setting MODEL_MAX_OUTPUT_TOKENS_* preserves existing input limit."""
        monkeypatch.setenv("MODEL_MAX_OUTPUT_TOKENS_CLAUDE_OPUS_4_6", "64000")
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_output_tokens == 64_000
        # Input should be preserved from the default (1_000_000)
        assert limits.max_input_tokens == 1_000_000

    def test_both_overrides_applied(self, monkeypatch):
        """Setting both input and output overrides for the same model."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6", "500000")
        monkeypatch.setenv("MODEL_MAX_OUTPUT_TOKENS_CLAUDE_OPUS_4_6", "64000")
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 500_000
        assert limits.max_output_tokens == 64_000


class TestNewModelViaEnvVar:
    """Test creating entirely new models via env vars that don't exist in defaults."""

    def test_new_model_input_only(self, monkeypatch):
        """A new model defined only by input tokens gets fallback output."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_MY_CUSTOM_MODEL", "100000")
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("my-custom-model")
        assert limits is not None
        assert limits.max_input_tokens == 100_000
        # Fallback output is 8192
        assert limits.max_output_tokens == 8_192

    def test_new_model_output_only(self, monkeypatch):
        """A new model defined only by output tokens gets fallback input."""
        monkeypatch.setenv("MODEL_MAX_OUTPUT_TOKENS_MY_CUSTOM_MODEL", "16000")
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("my-custom-model")
        assert limits is not None
        assert limits.max_output_tokens == 16_000
        # Fallback input is 200_000
        assert limits.max_input_tokens == 200_000

    def test_env_key_to_model_name_conversion(self, monkeypatch):
        """Env var underscores are converted to hyphens in model names."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_GPT_4O_MINI_2024", "128000")
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("gpt-4o-mini-2024")
        assert limits is not None
        assert limits.max_input_tokens == 128_000


class TestEnvVarEdgeCases:
    """Test malformed or unusual env var values."""

    def test_non_integer_env_var_skipped(self, monkeypatch):
        """Non-integer env var values are silently skipped."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6", "not-a-number")
        registry = ModelRegistry(_make_config())
        # Should still have the default
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 1_000_000

    def test_empty_string_env_var_skipped(self, monkeypatch):
        """Empty string env var values are silently skipped."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6", "")
        registry = ModelRegistry(_make_config())
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 1_000_000

    def test_float_env_var_truncated(self, monkeypatch):
        """Float env var values are rejected (int() fails on '500000.5')."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6", "500000.5")
        registry = ModelRegistry(_make_config())
        # int("500000.5") raises ValueError, so default should remain
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 1_000_000


# ---------------------------------------------------------------------------
# Model Registry: discovery edge cases
# ---------------------------------------------------------------------------


def _mock_response(status_code=200, json_data=None, raise_on_json=False):
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    if raise_on_json:
        resp.json.side_effect = json.JSONDecodeError("bad json", "", 0)
    else:
        resp.json.return_value = json_data
    return resp


class TestDiscoveryEdgeCases:
    """Edge cases in upstream model discovery."""

    @pytest.mark.asyncio
    async def test_discovery_zero_input_tokens_skipped(self):
        """Entries with max_input_tokens=0 are skipped."""
        registry = ModelRegistry(_make_config())
        original_opus = registry.get_limits("claude-opus-4-6")

        mock_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "model_group": "claude-opus-4-6",
                        "max_input_tokens": 0,
                        "max_output_tokens": 64_000,
                    },
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "src.core.model_registry.httpx.AsyncClient", return_value=mock_client
        ):
            await registry.discover_from_upstream("https://example.com/api", "key")

        # Original limits preserved since max_input_tokens was 0
        assert registry.get_limits("claude-opus-4-6") == original_opus

    @pytest.mark.asyncio
    async def test_discovery_negative_input_tokens_skipped(self):
        """Entries with negative max_input_tokens are skipped."""
        registry = ModelRegistry(_make_config())
        original_opus = registry.get_limits("claude-opus-4-6")

        mock_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "model_group": "claude-opus-4-6",
                        "max_input_tokens": -1,
                        "max_output_tokens": 64_000,
                    },
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "src.core.model_registry.httpx.AsyncClient", return_value=mock_client
        ):
            await registry.discover_from_upstream("https://example.com/api", "key")

        assert registry.get_limits("claude-opus-4-6") == original_opus

    @pytest.mark.asyncio
    async def test_discovery_missing_model_group_skipped(self):
        """Entries without model_group are skipped."""
        registry = ModelRegistry(_make_config())
        original_count = len(registry.get_all_models())

        mock_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "max_input_tokens": 500_000,
                        "max_output_tokens": 64_000,
                    },
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "src.core.model_registry.httpx.AsyncClient", return_value=mock_client
        ):
            await registry.discover_from_upstream("https://example.com/api", "key")

        # No new models should be added
        assert len(registry.get_all_models()) == original_count

    @pytest.mark.asyncio
    async def test_discovery_env_overrides_reapplied(self, monkeypatch):
        """Env overrides are re-applied after discovery to maintain priority."""
        monkeypatch.setenv("MODEL_MAX_INPUT_TOKENS_CLAUDE_OPUS_4_6", "999999")
        registry = ModelRegistry(_make_config())

        # Discovery tries to set a different value
        mock_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "model_group": "claude-opus-4-6",
                        "max_input_tokens": 500_000,
                        "max_output_tokens": 64_000,
                    },
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "src.core.model_registry.httpx.AsyncClient", return_value=mock_client
        ):
            await registry.discover_from_upstream("https://example.com/api", "key")

        # Env override should win over discovery
        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 999_999

    @pytest.mark.asyncio
    async def test_discovery_new_model_from_upstream(self):
        """Discovery can add entirely new models not in defaults."""
        registry = ModelRegistry(_make_config())
        assert registry.get_limits("brand-new-model") is None

        mock_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "model_group": "brand-new-model",
                        "max_input_tokens": 300_000,
                        "max_output_tokens": 16_000,
                    },
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "src.core.model_registry.httpx.AsyncClient", return_value=mock_client
        ):
            await registry.discover_from_upstream("https://example.com/api", "key")

        limits = registry.get_limits("brand-new-model")
        assert limits is not None
        assert limits.max_input_tokens == 300_000
        assert limits.max_output_tokens == 16_000

    @pytest.mark.asyncio
    async def test_discovery_missing_output_preserves_existing(self):
        """When upstream omits max_output_tokens, existing value is preserved."""
        registry = ModelRegistry(_make_config())
        original = registry.get_limits("claude-opus-4-6")

        mock_resp = _mock_response(
            json_data={
                "data": [
                    {
                        "model_group": "claude-opus-4-6",
                        "max_input_tokens": 500_000,
                        # No max_output_tokens
                    },
                ]
            }
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "src.core.model_registry.httpx.AsyncClient", return_value=mock_client
        ):
            await registry.discover_from_upstream("https://example.com/api", "key")

        limits = registry.get_limits("claude-opus-4-6")
        assert limits is not None
        assert limits.max_input_tokens == 500_000
        # Output should be preserved from original defaults
        assert limits.max_output_tokens == original.max_output_tokens

    @pytest.mark.asyncio
    async def test_discovery_url_strips_v1(self):
        """Discovery URL correctly strips /v1 suffix from base_url."""
        registry = ModelRegistry(_make_config())
        mock_resp = _mock_response(json_data={"data": []})

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "src.core.model_registry.httpx.AsyncClient", return_value=mock_client
        ):
            await registry.discover_from_upstream("https://example.com/api/v1", "key")

        call_url = mock_client.get.call_args[0][0]
        assert call_url == "https://example.com/api/model_group/info"

    @pytest.mark.asyncio
    async def test_discovery_url_no_v1_suffix(self):
        """Discovery URL works without /v1 suffix."""
        registry = ModelRegistry(_make_config())
        mock_resp = _mock_response(json_data={"data": []})

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch(
            "src.core.model_registry.httpx.AsyncClient", return_value=mock_client
        ):
            await registry.discover_from_upstream("https://example.com/api", "key")

        call_url = mock_client.get.call_args[0][0]
        assert call_url == "https://example.com/api/model_group/info"


# ---------------------------------------------------------------------------
# Token estimation: additional edge cases
# ---------------------------------------------------------------------------


def _msg(content: Any) -> SimpleNamespace:
    """Build a minimal message-like object."""
    return SimpleNamespace(content=content)


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool(name, description, input_schema):
    return SimpleNamespace(
        name=name, description=description, input_schema=input_schema
    )


class TestTokenEstimationEdgeCases:
    """Additional edge cases for the token estimation function."""

    def test_unicode_multibyte_counted_by_len(self):
        """Unicode characters are counted by Python len(), not byte length."""
        # Python len() counts codepoints, not bytes
        # "hello" = 5 chars, "こんにちは" = 5 chars
        ascii_result = estimate_tokens([_msg("hello")])
        unicode_result = estimate_tokens([_msg("こんにちは")])
        assert ascii_result == unicode_result  # Both 5 chars -> 1 token

    def test_emoji_counted_by_len(self):
        """Emoji are counted by Python len() (each emoji = 1-2 codepoints)."""
        # 4 basic emoji = 4 chars -> 1 token
        result = estimate_tokens([_msg("🎉🎊🎈🎁")])
        assert result == 1

    def test_large_schema_counted(self):
        """A large JSON schema contributes proportionally to the estimate."""
        large_schema = {
            "type": "object",
            "properties": {f"field_{i}": {"type": "string"} for i in range(100)},
        }
        schema_json = json.dumps(large_schema)
        tool = _tool(name="big_tool", description=None, input_schema=large_schema)
        result = estimate_tokens([], tools=[tool])
        # name (8) + schema json length
        expected = max(1, (len("big_tool") + len(schema_json)) // 4)
        assert result == expected

    def test_empty_tool_list(self):
        """An empty tools list contributes nothing."""
        result = estimate_tokens([_msg("abcd")], tools=[])
        assert result == 1  # 4 chars -> 1 token

    def test_tool_with_no_schema_or_description(self):
        """A tool with only a name still contributes its name length."""
        tool = _tool(name="x", description=None, input_schema=None)
        result = estimate_tokens([], tools=[tool])
        # Only name "x" = 1 char -> 1 token (floor)
        assert result == 1

    def test_very_long_message(self):
        """A very long message produces a proportional token estimate."""
        long_text = "a" * 100_000  # 100K chars -> 25K tokens
        result = estimate_tokens([_msg(long_text)])
        assert result == 25_000

    def test_mixed_content_block_types(self):
        """Only text blocks contribute; tool_use, tool_result blocks are ignored."""
        content = [
            SimpleNamespace(type="tool_use", id="t1", name="fn", input={}),
            _text_block("abcdefgh"),  # 8 chars
            SimpleNamespace(type="tool_result", tool_use_id="t1", content="result"),
        ]
        result = estimate_tokens([_msg(content)])
        assert result == 2  # 8 chars -> 2 tokens

    def test_system_empty_list(self):
        """An empty system list contributes nothing."""
        result = estimate_tokens([_msg("abcd")], system=[])
        assert result == 1

    def test_system_empty_string(self):
        """An empty string system prompt contributes nothing."""
        result = estimate_tokens([_msg("abcd")], system="")
        assert result == 1


# ---------------------------------------------------------------------------
# Pre-flight validation: boundary conditions
# ---------------------------------------------------------------------------


def _make_registry_mock(limits=None):
    """Build a mock ModelRegistry."""
    registry = MagicMock()
    registry.get_limits.return_value = limits
    return registry


class TestPreflightBoundary:
    """Test pre-flight validation at exact boundary conditions."""

    def test_exact_boundary_passes(self, monkeypatch):
        """Request at exactly the limit (not exceeding) should pass."""
        import src.api.endpoints as ep

        # context_limit = 1000 * 0.95 = 950
        # estimated_input (100) + max_tokens (850) = 950 <= 950 -> pass
        limits = ModelLimits(max_input_tokens=1000, max_output_tokens=100)
        mock_registry = _make_registry_mock(limits=limits)

        monkeypatch.setattr(ep, "model_registry", mock_registry)
        monkeypatch.setattr(ep, "estimate_tokens", lambda *a, **kw: 100)
        monkeypatch.setattr(ep.config, "model_registry_enabled", True)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 0.95)
        monkeypatch.setattr(ep.config, "anthropic_api_key", None)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 850,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        # At boundary: 100 + 850 = 950 == 950 -> not exceeding -> pass
        assert response.status_code != 400

    def test_one_over_boundary_rejected(self, monkeypatch):
        """Request one token over the limit should be rejected."""
        import src.api.endpoints as ep

        # context_limit = 1000 * 0.95 = 950
        # estimated_input (100) + max_tokens (851) = 951 > 950 -> reject
        limits = ModelLimits(max_input_tokens=1000, max_output_tokens=100)
        mock_registry = _make_registry_mock(limits=limits)

        monkeypatch.setattr(ep, "model_registry", mock_registry)
        monkeypatch.setattr(ep, "estimate_tokens", lambda *a, **kw: 100)
        monkeypatch.setattr(ep.config, "model_registry_enabled", True)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 0.95)
        monkeypatch.setattr(ep.config, "anthropic_api_key", None)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 851,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 400

    def test_safety_margin_1_0_uses_full_context(self, monkeypatch):
        """Safety margin of 1.0 uses the full context window."""
        import src.api.endpoints as ep

        # context_limit = 1000 * 1.0 = 1000
        # estimated_input (100) + max_tokens (900) = 1000 <= 1000 -> pass
        limits = ModelLimits(max_input_tokens=1000, max_output_tokens=100)
        mock_registry = _make_registry_mock(limits=limits)

        monkeypatch.setattr(ep, "model_registry", mock_registry)
        monkeypatch.setattr(ep, "estimate_tokens", lambda *a, **kw: 100)
        monkeypatch.setattr(ep.config, "model_registry_enabled", True)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 1.0)
        monkeypatch.setattr(ep.config, "anthropic_api_key", None)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 900,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code != 400

    def test_error_message_contains_calculation_details(self, monkeypatch):
        """The error message includes estimated tokens, max_tokens, and context limit."""
        import src.api.endpoints as ep

        limits = ModelLimits(max_input_tokens=1000, max_output_tokens=100)
        mock_registry = _make_registry_mock(limits=limits)

        monkeypatch.setattr(ep, "model_registry", mock_registry)
        monkeypatch.setattr(ep, "estimate_tokens", lambda *a, **kw: 800)
        monkeypatch.setattr(ep.config, "model_registry_enabled", True)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 0.95)
        monkeypatch.setattr(ep.config, "anthropic_api_key", None)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 400
        data = response.json()
        msg = data["error"]["message"]
        # Should contain the key numbers
        assert "800" in msg  # estimated input
        assert "200" in msg  # max_tokens
        assert "1,000" in msg  # max_input_tokens (formatted)
        assert "950" in msg  # context limit (1000 * 0.95)
        assert "claude-opus-4-6" in msg  # model name


# ---------------------------------------------------------------------------
# Pre-flight: registry None / model None cases
# ---------------------------------------------------------------------------


class TestPreflightNullGuards:
    """Test that pre-flight gracefully handles None registry and model."""

    def test_none_registry_passes(self, monkeypatch):
        """When model_registry is None, pre-flight is skipped entirely."""
        import src.api.endpoints as ep

        monkeypatch.setattr(ep, "model_registry", None)
        monkeypatch.setattr(ep.config, "model_registry_enabled", True)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 0.95)
        monkeypatch.setattr(ep.config, "anthropic_api_key", None)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 999_999,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        # Should NOT get 400 from pre-flight
        assert response.status_code != 400


# ---------------------------------------------------------------------------
# Models enrichment: registry None case
# ---------------------------------------------------------------------------


class TestModelsEnrichmentNullRegistry:
    """Test /v1/models when model_registry is None."""

    @pytest.mark.asyncio
    async def test_models_without_registry(self, monkeypatch):
        """When model_registry is None, upstream models are returned unchanged."""
        import src.api.endpoints as ep

        monkeypatch.setattr(ep, "model_registry", None)

        upstream_body = {
            "object": "list",
            "data": [
                {"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "openai"}
            ],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = upstream_body

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        monkeypatch.setattr(ep, "_get_httpx_client", lambda: mock_client)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()["data"]
        ids = [m["id"] for m in data]
        # Original model preserved
        assert "gpt-4o" in ids
        # No Claude aliases injected (registry is None)
        assert "claude-opus" not in ids
        assert "claude-sonnet" not in ids
        assert "claude-haiku" not in ids


# ---------------------------------------------------------------------------
# Count tokens endpoint: uses shared estimate_tokens
# ---------------------------------------------------------------------------


class TestCountTokensEndpoint:
    """Verify /v1/messages/count_tokens uses the shared estimate_tokens function."""

    def test_count_tokens_returns_estimate(self, monkeypatch):
        """The count_tokens endpoint returns the token estimate."""
        import src.api.endpoints as ep

        monkeypatch.setattr(ep.config, "anthropic_api_key", None)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        # "hello world" = 11 chars -> 11 // 4 = 2 tokens
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "hello world"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "input_tokens" in data
        assert data["input_tokens"] == 2  # 11 chars // 4 = 2

    def test_count_tokens_with_system(self, monkeypatch):
        """Count tokens includes system prompt in the estimate."""
        import src.api.endpoints as ep

        monkeypatch.setattr(ep.config, "anthropic_api_key", None)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        # system "1234567890" (10 chars) + message "12345678" (8 chars) = 18 chars -> 4 tokens
        response = client.post(
            "/v1/messages/count_tokens",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "system": "1234567890",
                "messages": [{"role": "user", "content": "12345678"}],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["input_tokens"] == 4  # 18 chars // 4 = 4
