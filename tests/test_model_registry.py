"""Unit tests for ModelRegistry."""

from unittest.mock import MagicMock

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
        assert limits == ModelLimits(max_input_tokens=1_000_000, max_output_tokens=128_000)

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
