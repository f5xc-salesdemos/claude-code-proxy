"""Unit tests for SearXNGClient."""

import time

import httpx
import pytest
from src.services.searxng import SearXNGClient


class TestSearXNGAvailability:
    """Test availability checking and caching."""

    @pytest.mark.asyncio
    async def test_availability_cached_true(self):
        """Availability result is cached within TTL when True."""
        client = SearXNGClient("http://test-searxng:8080")
        client._available = True
        client._available_checked_at = time.monotonic()
        result = await client.is_available()
        assert result is True

    @pytest.mark.asyncio
    async def test_availability_cached_false(self):
        """Availability result is cached within TTL when False."""
        client = SearXNGClient("http://test-searxng:8080")
        client._available = False
        client._available_checked_at = time.monotonic()
        result = await client.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_unreachable_host_returns_false(self):
        """Unreachable host returns unavailable."""
        client = SearXNGClient("http://192.0.2.1:9999", timeout=0.1)
        client._available = None  # Clear cache
        result = await client.is_available()
        assert result is False


class TestSearXNGSearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self):
        """Successful search returns Anthropic-formatted results."""
        client = SearXNGClient("http://test:8080")

        mock_resp = httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com",
                        "title": "Example",
                        "content": "Example content",
                    },
                    {
                        "url": "https://other.com",
                        "title": "Other",
                        "content": "Other content",
                        "publishedDate": "2024-01-01",
                    },
                ]
            },
            request=httpx.Request("GET", "http://test:8080/search"),
        )

        async def mock_get(*args, **kwargs):
            """Return mock response."""
            return mock_resp

        client._client.get = mock_get  # type: ignore[method-assign]
        result = await client.search("test query", max_results=5)

        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["type"] == "web_search_result"
        assert result["results"][0]["url"] == "https://example.com"
        assert result["results"][0]["title"] == "Example"
        assert result["results"][0]["encrypted_content"] == "Example content"
        assert result["results"][1].get("page_age") == "2024-01-01"

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self):
        """Search truncates to max_results."""
        client = SearXNGClient("http://test:8080")

        mock_resp = httpx.Response(
            200,
            json={
                "results": [
                    {"url": f"https://r{i}.com", "title": f"R{i}", "content": ""}
                    for i in range(10)
                ]
            },
            request=httpx.Request("GET", "http://test:8080/search"),
        )

        async def mock_get(*args, **kwargs):
            """Return mock response."""
            return mock_resp

        client._client.get = mock_get  # type: ignore[method-assign]
        result = await client.search("test", max_results=3)
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_search_error_returns_error_dict(self):
        """Network error returns error dict instead of raising."""
        client = SearXNGClient("http://192.0.2.1:9999", timeout=0.1)
        result = await client.search("test")
        assert "error" in result
        assert result["error"]["type"] == "web_search_tool_result_error"
        assert result["error"]["error_code"] == "unavailable"

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        """close() completes without error."""
        client = SearXNGClient("http://test:8080")
        await client.close()

    @pytest.mark.asyncio
    async def test_trailing_slash_stripped(self):
        """Base URL trailing slash is stripped."""
        client = SearXNGClient("http://test:8080/")
        assert client.base_url == "http://test:8080"
