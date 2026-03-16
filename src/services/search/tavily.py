"""Tavily web search provider plugin.

Uses the Tavily REST API (https://api.tavily.com/search) directly
via httpx — no SDK dependency required.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from src.services.search.base import SearchProvider
from src.services.search.registry import register_provider

logger = logging.getLogger(__name__)

TAVILY_API_URL = "https://api.tavily.com/search"
WEB_SEARCH_RESULT_TYPE = "web_search_result"
WEB_SEARCH_TOOL_RESULT_ERROR_TYPE = "web_search_tool_result_error"

# Map Tavily HTTP status codes to Anthropic web_search spec error codes.
_HTTP_TO_ERROR_CODE: Dict[int, str] = {
    400: "invalid_tool_input",
    401: "unavailable",
    403: "unavailable",
    429: "too_many_requests",
}


class TavilySearchProvider(SearchProvider):
    """Search provider backed by the Tavily REST API."""

    def __init__(self, api_key: str, timeout: float = 15.0):
        self._api_key = api_key
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._available: Optional[bool] = None
        self._available_checked_at: float = 0.0
        self._availability_ttl: float = 60.0

    @property
    def provider_name(self) -> str:
        return "tavily"

    async def is_available(self) -> bool:
        """Check whether the Tavily API is reachable (cached for TTL).

        Returns ``False`` immediately if no API key is configured.
        """
        if not self._api_key:
            return False

        now = time.monotonic()
        if (
            self._available is not None
            and (now - self._available_checked_at) < self._availability_ttl
        ):
            return self._available

        try:
            # Tavily doesn't have a dedicated health endpoint, so we
            # make a lightweight search request to verify connectivity.
            resp = await self._client.post(
                TAVILY_API_URL,
                json={
                    "api_key": self._api_key,
                    "query": "test",
                    "max_results": 1,
                },
                timeout=5.0,
            )
            self._available = resp.status_code < 500
        except (httpx.HTTPError, httpx.TimeoutException, OSError):
            self._available = False

        self._available_checked_at = now
        return self._available

    async def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a Tavily search and return Anthropic-formatted results."""
        try:
            payload: Dict[str, Any] = {
                "api_key": self._api_key,
                "query": query,
                "max_results": max_results,
            }
            if allowed_domains:
                payload["include_domains"] = allowed_domains
            if blocked_domains:
                payload["exclude_domains"] = blocked_domains

            resp = await self._client.post(TAVILY_API_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

            results: List[Dict[str, Any]] = []
            for item in data.get("results", [])[:max_results]:
                result: Dict[str, Any] = {
                    "type": WEB_SEARCH_RESULT_TYPE,
                    "url": item.get("url", ""),
                    "title": item.get("title", ""),
                    "encrypted_content": item.get("content", ""),
                }
                page_age = item.get("published_date")
                if page_age:
                    result["page_age"] = str(page_age)
                results.append(result)

            return {"results": results}

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error_code = _HTTP_TO_ERROR_CODE.get(status, "unavailable")
            logger.warning("Tavily search failed (HTTP %s): %s", status, e)
            return {
                "error": {
                    "type": WEB_SEARCH_TOOL_RESULT_ERROR_TYPE,
                    "error_code": error_code,
                }
            }
        except (httpx.HTTPError, httpx.TimeoutException, OSError) as e:
            logger.warning("Tavily search failed: %s", e)
            return {
                "error": {
                    "type": WEB_SEARCH_TOOL_RESULT_ERROR_TYPE,
                    "error_code": "unavailable",
                }
            }

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()


def _tavily_factory() -> Optional[TavilySearchProvider]:
    """Create a TavilySearchProvider from environment configuration.

    Returns ``None`` if ``TAVILY_API_KEY`` is not set.
    """
    # Import config lazily to avoid circular imports during module loading
    from src.core.config import config

    api_key = getattr(config, "tavily_api_key", None)
    if not api_key:
        return None
    return TavilySearchProvider(api_key)


# Auto-register when module is imported
register_provider("tavily", _tavily_factory)
