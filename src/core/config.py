"""Application configuration loaded from environment variables."""

import logging
import os
import sys
from typing import Dict, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings

_VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


class Config(BaseSettings):
    """Proxy configuration sourced from environment variables.

    All fields are read from environment variables automatically.
    Prefix-less variable names match 1-to-1 with field names
    (case-insensitive on the env var side).
    """

    # Required
    openai_api_key: str

    # Optional API keys
    anthropic_api_key: Optional[str] = None

    # Server
    openai_base_url: str = "https://api.openai.com/v1"
    azure_api_version: Optional[str] = None
    host: str = "0.0.0.0"
    port: int = 8082
    log_level: str = "INFO"

    # Token limits
    max_tokens_limit: int = 4096
    min_tokens_limit: int = 100

    # Web search
    search_provider: Optional[str] = None
    tavily_api_key: Optional[str] = None

    # Connection
    request_timeout: int = 90
    max_retries: int = 2

    # Model mappings
    big_model: str = "gpt-4o"
    middle_model: Optional[str] = None
    small_model: str = "gpt-4o-mini"

    # Model registry
    model_registry_enabled: bool = True
    model_registry_refresh_interval: int = 300
    model_registry_safety_margin: float = 0.95

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        """Accept values like 'INFO # comment' and normalize to upper case."""
        level = str(v).split()[0].upper()
        if level not in _VALID_LOG_LEVELS:
            return "INFO"
        return level

    @field_validator("middle_model", mode="before")
    @classmethod
    def _default_middle_to_big(cls, v: Optional[str]) -> Optional[str]:
        """Keep None so model_post_init can resolve it from big_model."""
        return v if v else None

    def model_post_init(self, __context: object) -> None:
        """Resolve middle_model default after all fields are set."""
        if self.middle_model is None:
            # Use object.__setattr__ because the model is frozen-ish in v2
            object.__setattr__(self, "middle_model", self.big_model)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def validate_client_api_key(self, client_api_key: str) -> bool:
        """Validate client's Anthropic API key.

        If no ``ANTHROPIC_API_KEY`` is configured, all keys are accepted.
        """
        if not self.anthropic_api_key:
            return True
        return client_api_key == self.anthropic_api_key

    def get_custom_headers(self) -> Dict[str, str]:
        """Build custom headers from ``CUSTOM_HEADER_*`` env vars.

        ``CUSTOM_HEADER_X_MY_HEADER=value`` becomes ``X-MY-HEADER: value``.
        """
        custom_headers: Dict[str, str] = {}
        for env_key, env_value in os.environ.items():
            if env_key.startswith("CUSTOM_HEADER_"):
                header_name = env_key[14:]  # strip prefix
                if header_name:
                    header_name = header_name.replace("_", "-")
                    custom_headers[header_name] = env_value
        return custom_headers


try:
    config = Config()  # type: ignore[call-arg]
    if not config.anthropic_api_key:
        logging.getLogger(__name__).warning(
            "ANTHROPIC_API_KEY not set — client API key validation disabled"
        )
except Exception as e:
    print(f"Configuration Error: {e}")
    sys.exit(1)
