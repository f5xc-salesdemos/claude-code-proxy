"""Firecrawl-based web fetch provider.

Fetches URLs via the local Firecrawl scrape API and returns content
in Anthropic's web_fetch_tool_result format.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

WEB_FETCH_RESULT_TYPE = "web_fetch_result"
WEB_FETCH_ERROR_TYPE = "web_fetch_tool_error"


class FirecrawlFetchProvider:
    """Fetches web pages via local Firecrawl scrape endpoint."""

    def __init__(self, api_url: str = "http://localhost:3002", timeout: float = 30.0):
        self._api_url = api_url.rstrip("/")
        self._timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def fetch(
        self,
        url: str,
        *,
        allowed_domains: Optional[list] = None,
        blocked_domains: Optional[list] = None,
    ) -> Dict[str, Any]:
        """Fetch a URL and return Anthropic-formatted web_fetch result.

        Returns either a web_fetch_result dict or a web_fetch_tool_error dict.
        """
        # Domain filtering
        if allowed_domains or blocked_domains:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc.lower()
            if allowed_domains and not any(domain.endswith(d.lower()) for d in allowed_domains):
                return {"error": {"type": WEB_FETCH_ERROR_TYPE, "error_code": "url_not_allowed"}}
            if blocked_domains and any(domain.endswith(d.lower()) for d in blocked_domains):
                return {"error": {"type": WEB_FETCH_ERROR_TYPE, "error_code": "url_not_allowed"}}

        try:
            payload = {
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
            }

            resp = await self._client.post(
                f"{self._api_url}/v1/scrape", json=payload
            )
            resp.raise_for_status()
            data = resp.json()

            scrape_data = data.get("data", {})
            markdown = scrape_data.get("markdown", "")
            metadata = scrape_data.get("metadata", {})
            title = metadata.get("title", "")

            if not markdown:
                return {"error": {"type": WEB_FETCH_ERROR_TYPE, "error_code": "url_not_accessible"}}

            return {
                "result": {
                    "type": WEB_FETCH_RESULT_TYPE,
                    "url": url,
                    "content": {
                        "type": "document",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": markdown[:100000],  # Cap at ~100K chars
                        },
                        "title": title,
                        "citations": {"enabled": True},
                    },
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            }

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            logger.warning("Firecrawl fetch failed (HTTP %s): %s", status, e)
            error_code = "url_not_accessible" if status == 404 else "unavailable"
            return {"error": {"type": WEB_FETCH_ERROR_TYPE, "error_code": error_code}}

        except (httpx.HTTPError, httpx.TimeoutException, OSError) as e:
            logger.warning("Firecrawl fetch failed: %s", e)
            return {"error": {"type": WEB_FETCH_ERROR_TYPE, "error_code": "unavailable"}}

    async def close(self) -> None:
        await self._client.aclose()
