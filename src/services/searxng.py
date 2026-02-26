import time
import logging
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

WEB_SEARCH_RESULT_TYPE = "web_search_result"
WEB_SEARCH_TOOL_RESULT_ERROR_TYPE = "web_search_tool_result_error"


class SearXNGClient:
    """Async client for querying a SearXNG instance."""

    def __init__(self, base_url: str, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
        self._available: Optional[bool] = None
        self._available_checked_at: float = 0.0
        self._availability_ttl: float = 60.0

    async def is_available(self) -> bool:
        now = time.monotonic()
        if self._available is not None and (now - self._available_checked_at) < self._availability_ttl:
            return self._available
        try:
            resp = await self._client.get(self.base_url, timeout=3.0)
            self._available = resp.status_code < 500
        except Exception:
            self._available = False
        self._available_checked_at = now
        return self._available

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Execute a search and return Anthropic-formatted result content.

        Returns the ``content`` value for a ``web_search_tool_result`` block —
        either a list of ``web_search_result`` dicts or an error dict.
        """
        try:
            resp = await self._client.get(
                f"{self.base_url}/search",
                params={"q": query, "format": "json"},
            )
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
                page_age = item.get("publishedDate")
                if page_age:
                    result["page_age"] = str(page_age)
                results.append(result)
            return {"results": results}
        except Exception as e:
            logger.warning(f"SearXNG search failed: {e}")
            return {
                "error": {
                    "type": WEB_SEARCH_TOOL_RESULT_ERROR_TYPE,
                    "error_code": "unavailable",
                }
            }

    async def close(self):
        await self._client.aclose()
