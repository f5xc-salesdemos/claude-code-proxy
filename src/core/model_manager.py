"""Model name mapping between Claude and OpenAI families."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.config import Config

# Passthrough prefixes — models from these providers are returned unchanged.
_PASSTHROUGH_PREFIXES = ("gpt-", "o1-", "o3-", "o4-", "ep-", "doubao-", "deepseek-")


class ModelManager:
    """Maps Claude model names to configured OpenAI model names."""

    def __init__(self, config: "Config") -> None:
        self.config = config

    def map_claude_model_to_openai(self, claude_model: str) -> str:
        """Map Claude model names to OpenAI model names based on tier.

        Mapping:
          - haiku  -> SMALL_MODEL
          - sonnet -> MIDDLE_MODEL
          - opus   -> BIG_MODEL

        OpenAI, ARK, Doubao, and DeepSeek models are returned as-is.
        """
        if claude_model.startswith(_PASSTHROUGH_PREFIXES):
            return claude_model

        model_lower = claude_model.lower()
        if "haiku" in model_lower:
            return str(self.config.small_model)
        if "sonnet" in model_lower:
            return str(self.config.middle_model)
        if "opus" in model_lower:
            return str(self.config.big_model)
        # Default to big model for unknown models
        return str(self.config.big_model)
