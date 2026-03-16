"""Registry of known Claude model context window limits."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from src.core.config import Config


@dataclasses.dataclass
class ModelLimits:
    """Context window limits for a model."""

    max_input_tokens: int
    max_output_tokens: int


_DEFAULT_LIMITS: Dict[str, ModelLimits] = {
    "claude-opus-4-6": ModelLimits(1_000_000, 128_000),
    "claude-sonnet-4-6": ModelLimits(1_000_000, 128_000),
    "claude-3-7-sonnet-20250219": ModelLimits(200_000, 128_000),
    "claude-haiku-4-5": ModelLimits(200_000, 8_192),
    "claude-3-5-sonnet-20241022": ModelLimits(200_000, 8_192),
    "claude-3-5-haiku-20241022": ModelLimits(200_000, 8_192),
    "claude-3-opus-20240229": ModelLimits(200_000, 4_096),
}


class ModelRegistry:
    """Registry of known model context window limits."""

    def __init__(self, config: "Config") -> None:
        self.config = config
        self._limits: Dict[str, ModelLimits] = dict(_DEFAULT_LIMITS)

    def get_limits(self, model_name: str) -> Optional[ModelLimits]:
        """Return ModelLimits for the given model name, or None if unknown."""
        return self._limits.get(model_name)

    def get_all_models(self) -> Dict[str, ModelLimits]:
        """Return a copy of all registered model limits."""
        return dict(self._limits)
