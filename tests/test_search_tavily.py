"""Tests for the Tavily search provider plugin."""

# pylint: disable=protected-access

import time
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from src.services.search.tavily import TavilySearchProvider


class TestTavilyProviderName:
    """Test provider identity."""

    def test_provider_name(self):
        """Provider identifies itself as 'tavily'."""
        provider = TavilySearchProvider(api_key="tvly-test")
        assert provider.provider_name == "tavily"


class TestTavilyAvailability:
    """Test availability checking and caching."""

    @pytest.mark.asyncio
    async def test_no_api_key_returns_false(self):
        """Provider without an API key is never available."""
        provider = TavilySearchProvider(api_key="")
        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_availability_cached_true(self):
        """Availability result is cached within TTL when True."""
        provider = TavilySearchProvider(api_key="tvly-test")
        provider._available = True
        provider._available_checked_at = time.monotonic()
        assert await provider.is_available() is True

    @pytest.mark.asyncio
    async def test_availability_cached_false(self):
        """Availability result is cached within TTL when False."""
        provider = TavilySearchProvider(api_key="tvly-test")
        provider._available = False
        provider._available_checked_at = time.monotonic()
        assert await provider.is_available() is False

    @pytest.mark.asyncio
    async def test_availability_expired_cache_rechecks(self):
        """Expired cache causes a fresh availability check."""
        provider = TavilySearchProvider(api_key="tvly-test")
        provider._available = True
        provider._available_checked_at = time.monotonic() - 120  # Expired

        mock_resp = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.is_available()
        assert result is True
        provider._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_availability_server_error_returns_false(self):
        """HTTP 500 from Tavily marks provider as unavailable."""
        provider = TavilySearchProvider(api_key="tvly-test")
        provider._available = None  # Force recheck

        mock_resp = httpx.Response(
            500,
            json={"error": "internal"},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_availability_network_error_returns_false(self):
        """Network error marks provider as unavailable."""
        provider = TavilySearchProvider(api_key="tvly-test")
        provider._available = None

        provider._client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await provider.is_available()
        assert result is False


class TestTavilySearch:
    """Test search functionality."""

    @pytest.mark.asyncio
    async def test_search_returns_formatted_results(self):
        """Successful search returns Anthropic-formatted results."""
        provider = TavilySearchProvider(api_key="tvly-test")

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
                        "published_date": "2024-01-15",
                    },
                ]
            },
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.search("test query")

        assert "results" in result
        assert len(result["results"]) == 2

        r0 = result["results"][0]
        assert r0["type"] == "web_search_result"
        assert r0["url"] == "https://example.com"
        assert r0["title"] == "Example"
        assert r0["encrypted_content"] == "Example content"
        assert "page_age" not in r0

        r1 = result["results"][1]
        assert r1["page_age"] == "2024-01-15"

    @pytest.mark.asyncio
    async def test_search_respects_max_results(self):
        """Search truncates to max_results."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            200,
            json={
                "results": [
                    {"url": f"https://r{i}.com", "title": f"R{i}", "content": ""}
                    for i in range(10)
                ]
            },
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.search("test", max_results=3)
        assert len(result["results"]) == 3

    @pytest.mark.asyncio
    async def test_search_passes_max_results_to_api(self):
        """max_results is sent to the Tavily API."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.search("query", max_results=7)

        call_kwargs = provider._client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["max_results"] == 7

    @pytest.mark.asyncio
    async def test_search_sends_api_key(self):
        """API key is included in the request payload."""
        provider = TavilySearchProvider(api_key="tvly-secret")

        mock_resp = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.search("query")

        call_kwargs = provider._client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["api_key"] == "tvly-secret"

    @pytest.mark.asyncio
    async def test_search_http_error_returns_error_dict(self):
        """HTTP error returns error dict instead of raising."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            401,
            json={"error": "unauthorized"},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.search("test")
        assert "error" in result
        assert result["error"]["type"] == "web_search_tool_result_error"
        assert result["error"]["error_code"] == "unavailable"

    @pytest.mark.asyncio
    async def test_search_network_error_returns_error_dict(self):
        """Network error returns error dict instead of raising."""
        provider = TavilySearchProvider(api_key="tvly-test")

        provider._client.post = AsyncMock(
            side_effect=httpx.ConnectError("Connection refused")
        )

        result = await provider.search("test")
        assert "error" in result
        assert result["error"]["type"] == "web_search_tool_result_error"

    @pytest.mark.asyncio
    async def test_search_timeout_returns_error_dict(self):
        """Timeout returns error dict instead of raising."""
        provider = TavilySearchProvider(api_key="tvly-test")

        provider._client.post = AsyncMock(side_effect=httpx.ReadTimeout("Timed out"))

        result = await provider.search("test")
        assert "error" in result


class TestTavilyDomainFilters:
    """Test allowed_domains and blocked_domains support."""

    @pytest.mark.asyncio
    async def test_allowed_domains_passed_as_include_domains(self):
        """allowed_domains are forwarded as include_domains to Tavily."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.search("query", allowed_domains=["example.com", "test.org"])

        call_kwargs = provider._client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["include_domains"] == ["example.com", "test.org"]
        assert "exclude_domains" not in payload

    @pytest.mark.asyncio
    async def test_blocked_domains_passed_as_exclude_domains(self):
        """blocked_domains are forwarded as exclude_domains to Tavily."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.search("query", blocked_domains=["bad.com"])

        call_kwargs = provider._client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["exclude_domains"] == ["bad.com"]
        assert "include_domains" not in payload

    @pytest.mark.asyncio
    async def test_both_domain_filters_passed(self):
        """Both domain filters are forwarded simultaneously."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.search(
            "query",
            allowed_domains=["good.com"],
            blocked_domains=["bad.com"],
        )

        call_kwargs = provider._client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["include_domains"] == ["good.com"]
        assert payload["exclude_domains"] == ["bad.com"]

    @pytest.mark.asyncio
    async def test_no_domain_filters_omits_fields(self):
        """When no domain filters are given, the fields are omitted."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            200,
            json={"results": []},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        await provider.search("query")

        call_kwargs = provider._client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "include_domains" not in payload
        assert "exclude_domains" not in payload


class TestTavilyErrorCodeMapping:
    """Test HTTP status code to spec error code mapping."""

    @pytest.mark.asyncio
    async def test_http_400_returns_invalid_tool_input(self):
        """HTTP 400 maps to invalid_tool_input error code."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            400,
            json={"error": "bad request"},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.search("test")
        assert result["error"]["error_code"] == "invalid_tool_input"

    @pytest.mark.asyncio
    async def test_http_429_returns_too_many_requests(self):
        """HTTP 429 maps to too_many_requests error code."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            429,
            json={"error": "rate limited"},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.search("test")
        assert result["error"]["error_code"] == "too_many_requests"

    @pytest.mark.asyncio
    async def test_http_401_returns_unavailable(self):
        """HTTP 401 maps to unavailable error code."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            401,
            json={"error": "unauthorized"},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.search("test")
        assert result["error"]["error_code"] == "unavailable"

    @pytest.mark.asyncio
    async def test_http_403_returns_unavailable(self):
        """HTTP 403 maps to unavailable error code."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            403,
            json={"error": "forbidden"},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.search("test")
        assert result["error"]["error_code"] == "unavailable"

    @pytest.mark.asyncio
    async def test_http_500_returns_unavailable_fallback(self):
        """HTTP 500 falls back to unavailable error code."""
        provider = TavilySearchProvider(api_key="tvly-test")

        mock_resp = httpx.Response(
            500,
            json={"error": "internal"},
            request=httpx.Request("POST", "https://api.tavily.com/search"),
        )
        provider._client.post = AsyncMock(return_value=mock_resp)

        result = await provider.search("test")
        assert result["error"]["error_code"] == "unavailable"


class TestTavilyLifecycle:
    """Test resource management."""

    @pytest.mark.asyncio
    async def test_close_does_not_raise(self):
        """close() completes without error."""
        provider = TavilySearchProvider(api_key="tvly-test")
        await provider.close()

    @pytest.mark.asyncio
    async def test_close_closes_http_client(self):
        """close() closes the underlying httpx client."""
        provider = TavilySearchProvider(api_key="tvly-test")
        provider._client.aclose = AsyncMock()
        await provider.close()
        provider._client.aclose.assert_called_once()


class TestTavilyRegistration:
    """Test auto-registration in the registry."""

    def test_tavily_registered(self):
        """Tavily provider is registered when the module is imported."""
        from src.services.search.registry import _registry

        assert "tavily" in _registry

    def test_factory_returns_none_without_api_key(self):
        """Factory returns None when TAVILY_API_KEY is not configured."""
        from src.services.search.tavily import _tavily_factory

        # The factory does a lazy import of config inside the function
        with patch("src.core.config.config") as mock_config:
            mock_config.tavily_api_key = None
            result = _tavily_factory()
            assert result is None

    def test_factory_returns_provider_with_api_key(self):
        """Factory returns a TavilySearchProvider when key is configured."""
        from src.services.search.tavily import _tavily_factory

        with patch("src.core.config.config") as mock_config:
            mock_config.tavily_api_key = "tvly-test-key"
            result = _tavily_factory()
            assert isinstance(result, TavilySearchProvider)
