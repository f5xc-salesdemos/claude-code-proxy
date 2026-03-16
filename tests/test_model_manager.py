"""Unit tests for ModelManager."""

from unittest.mock import MagicMock
from src.core.model_manager import ModelManager


def _make_config(
    big: str = "gpt-4o",
    middle: str = "gpt-4o",
    small: str = "gpt-4o-mini",
) -> MagicMock:
    """Create a mock Config with the given model settings."""
    cfg = MagicMock()
    cfg.big_model = big
    cfg.middle_model = middle
    cfg.small_model = small
    return cfg


class TestModelMapping:
    """Test model name mapping logic."""

    def test_haiku_maps_to_small(self):
        """Haiku models map to the small_model."""
        mm = ModelManager(_make_config(small="gpt-4o-mini"))
        assert mm.map_claude_model_to_openai("claude-haiku-4-5") == "gpt-4o-mini"
        assert (
            mm.map_claude_model_to_openai("claude-3-5-haiku-20241022") == "gpt-4o-mini"
        )

    def test_sonnet_maps_to_middle(self):
        """Sonnet models map to the middle_model."""
        mm = ModelManager(_make_config(middle="gpt-4o"))
        assert mm.map_claude_model_to_openai("claude-sonnet-4-6") == "gpt-4o"
        assert mm.map_claude_model_to_openai("claude-3-5-sonnet-20241022") == "gpt-4o"

    def test_opus_maps_to_big(self):
        """Opus models map to the big_model."""
        mm = ModelManager(_make_config(big="gpt-4o"))
        assert mm.map_claude_model_to_openai("claude-opus-4-6") == "gpt-4o"
        assert mm.map_claude_model_to_openai("claude-3-opus-20240229") == "gpt-4o"

    def test_unknown_defaults_to_big(self):
        """Unknown model names default to big_model."""
        mm = ModelManager(_make_config(big="fallback-model"))
        assert mm.map_claude_model_to_openai("claude-mystery") == "fallback-model"


class TestPassthroughModels:
    """Known provider prefixes are returned unchanged."""

    def test_gpt_passthrough(self):
        """GPT models pass through unchanged."""
        mm = ModelManager(_make_config())
        assert mm.map_claude_model_to_openai("gpt-4o") == "gpt-4o"
        assert mm.map_claude_model_to_openai("gpt-3.5-turbo") == "gpt-3.5-turbo"

    def test_o1_passthrough(self):
        """o1 models pass through unchanged."""
        mm = ModelManager(_make_config())
        assert mm.map_claude_model_to_openai("o1-preview") == "o1-preview"

    def test_o3_passthrough(self):
        """o3 models pass through unchanged."""
        mm = ModelManager(_make_config())
        assert mm.map_claude_model_to_openai("o3-mini") == "o3-mini"

    def test_deepseek_passthrough(self):
        """DeepSeek models pass through unchanged."""
        mm = ModelManager(_make_config())
        assert mm.map_claude_model_to_openai("deepseek-chat") == "deepseek-chat"

    def test_doubao_passthrough(self):
        """Doubao models pass through unchanged."""
        mm = ModelManager(_make_config())
        assert mm.map_claude_model_to_openai("doubao-pro") == "doubao-pro"

    def test_ark_passthrough(self):
        """ARK endpoint IDs pass through unchanged."""
        mm = ModelManager(_make_config())
        assert mm.map_claude_model_to_openai("ep-abc123") == "ep-abc123"
