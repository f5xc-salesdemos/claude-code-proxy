"""Firecrawl web search provider plugin.

Uses the local Firecrawl search API (http://localhost:3002/v1/search)
backed by SearXNG metasearch engine. No external API keys required.
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx
from src.services.search.base import SearchProvider
from src.services.search.registry import register_provider

logger = logging.getLogger(__name__)

WEB_SEARCH_RESULT_TYPE = "web_search_result"
WEB_SEARCH_TOOL_RESULT_ERROR_TYPE = "web_search_tool_result_error"


class FirecrawlSearchProvider(SearchProvider):
    """Search provider backed by a local Firecrawl instance."""

    def __init__(self, api_url: str = "http://localhost:3002", timeout: float = 30.0):
        self._api_url = api_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._available: Optional[bool] = None
        self._available_checked_at: float = 0.0
        self._availability_ttl: float = 60.0

    @property
    def provider_name(self) -> str:
        return "firecrawl"

    async def is_available(self) -> bool:
        now = time.monotonic()
        if (
            self._available is not None
            and (now - self._available_checked_at) < self._availability_ttl
        ):
            return self._available

        try:
            resp = await self._client.get(f"{self._api_url}/", timeout=5.0)
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
        """Execute a Firecrawl search and return Anthropic-formatted results."""
        try:
            # Build query with domain operators if specified
            enriched_query = query
            if allowed_domains:
                site_ops = " OR ".join(f"site:{d}" for d in allowed_domains)
                enriched_query = f"{query} ({site_ops})"
            if blocked_domains:
                exclude_ops = " ".join(f"-site:{d}" for d in blocked_domains)
                enriched_query = f"{enriched_query} {exclude_ops}"

            payload: Dict[str, Any] = {
                "query": enriched_query,
                "limit": max_results,
                "scrapeOptions": {
                    "formats": ["markdown"],
                    "onlyMainContent": True,
                },
            }

            resp = await self._client.post(
                f"{self._api_url}/v1/search", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

            results: List[Dict[str, Any]] = []
            for item in (data.get("data") or [])[:max_results]:
                metadata = item.get("metadata") or {}
                result: Dict[str, Any] = {
                    "type": WEB_SEARCH_RESULT_TYPE,
                    "url": item.get("url", metadata.get("sourceURL", "")),
                    "title": metadata.get("title", item.get("title", "")),
                    "encrypted_content": (item.get("markdown") or "")[:3000],
                }
                page_age = metadata.get("publishedTime")
                if page_age:
                    result["page_age"] = str(page_age)
                results.append(result)

            return {"results": results}

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.warning("Firecrawl search failed (HTTP %s): %s", status, e)
            return {
                "error": {
                    "type": WEB_SEARCH_TOOL_RESULT_ERROR_TYPE,
                    "error_code": "unavailable",
                }
            }
        except (httpx.HTTPError, httpx.TimeoutException, OSError) as e:
            logger.warning("Firecrawl search failed: %s", e)
            return {
                "error": {
                    "type": WEB_SEARCH_TOOL_RESULT_ERROR_TYPE,
                    "error_code": "unavailable",
                }
            }

    async def close(self) -> None:
        await self._client.aclose()


def _firecrawl_factory() -> Optional[FirecrawlSearchProvider]:
    """Create a FirecrawlSearchProvider from environment configuration."""
    from src.core.config import config

    api_url = getattr(config, "firecrawl_api_url", None) or "http://localhost:3002"
    return FirecrawlSearchProvider(api_url)


# Auto-register when module is imported
register_provider("firecrawl", _firecrawl_factory)
