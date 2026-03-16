"""Registry of known Claude model context window limits."""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import TYPE_CHECKING, Dict, Optional

import httpx

if TYPE_CHECKING:
    from src.core.config import Config

logger = logging.getLogger(__name__)


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
        self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Override model limits from ``MODEL_MAX_INPUT_TOKENS_*`` and ``MODEL_MAX_OUTPUT_TOKENS_*`` env vars."""
        input_prefix = "MODEL_MAX_INPUT_TOKENS_"
        output_prefix = "MODEL_MAX_OUTPUT_TOKENS_"
        for env_key, env_value in os.environ.items():
            if env_key.startswith(input_prefix):
                model_key = env_key[len(input_prefix):].lower().replace("_", "-")
            elif env_key.startswith(output_prefix):
                model_key = env_key[len(output_prefix):].lower().replace("_", "-")
            else:
                continue

            try:
                value = int(env_value)
            except (ValueError, TypeError):
                logger.debug("Skipping non-integer env var %s=%r", env_key, env_value)
                continue

            if model_key in self._limits:
                existing = self._limits[model_key]
                if env_key.startswith(input_prefix):
                    self._limits[model_key] = ModelLimits(value, existing.max_output_tokens)
                else:
                    self._limits[model_key] = ModelLimits(existing.max_input_tokens, value)
            else:
                if env_key.startswith(input_prefix):
                    self._limits[model_key] = ModelLimits(value, 0)
                else:
                    self._limits[model_key] = ModelLimits(0, value)

    def get_limits(self, model_name: str) -> Optional[ModelLimits]:
        """Return ModelLimits for the given model name, or None if unknown."""
        return self._limits.get(model_name)

    def get_all_models(self) -> Dict[str, ModelLimits]:
        """Return a copy of all registered model limits."""
        return dict(self._limits)

    async def discover_from_upstream(self, base_url: str, api_key: str) -> None:
        """Fetch model limits from upstream API and update the registry.

        Calls ``{base_url}/model_group/info`` (stripping trailing ``/v1``
        if present) and merges discovered limits into the registry.
        On any error the registry is left unchanged.
        """
        # Strip trailing /v1 to build the discovery endpoint
        url = base_url.rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        url = f"{url}/model_group/info"

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if resp.status_code != 200:
                    logger.warning(
                        "Upstream discovery returned status %d", resp.status_code
                    )
                    return

                payload = resp.json()
                entries = payload["data"]
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            logger.warning("Upstream model discovery failed: %s", exc)
            return

        for entry in entries:
            model_group = entry.get("model_group")
            max_input = entry.get("max_input_tokens")
            if model_group is None or max_input is None or max_input <= 0:
                continue

            # Determine max_output_tokens: use upstream value if present,
            # otherwise preserve existing limit's value (or default to 0).
            max_output = entry.get("max_output_tokens")
            if max_output is None:
                existing = self._limits.get(model_group)
                max_output = existing.max_output_tokens if existing else 0

            self._limits[model_group] = ModelLimits(
                int(max_input), int(max_output)
            )

        # Re-apply env overrides so they remain highest priority after discovery
        self._apply_env_overrides()
