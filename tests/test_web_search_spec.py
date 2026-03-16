"""Tests verifying web search implementation against the Anthropic spec.

Covers:
- Fix 1: streaming server_tool_use content_block_start includes "input": {}
- Fix 2: non-streaming handles multiple web_search tool calls
- Fix 3: non-streaming content block ordering (tool blocks before text)
- Fix 4: non-streaming dynamic web_search_requests count
- Fix 5: domain filters are passed through to search provider
"""

import logging

import pytest
from tests.conftest import (
    FakeSearchProvider,
    collect_streaming_events,
    fake_openai_stream,
    make_request,
    openai_response,
    parse_sse_events,
)
from src.conversion.response_converter import convert_openai_streaming_to_claude
from src.api.endpoints import _build_non_streaming_web_search_response


# ---------------------------------------------------------------------------
# Fix 1: streaming server_tool_use content_block_start includes "input": {}
# ---------------------------------------------------------------------------


class TestStreamingServerToolUseInput:
    """Spec section 6 step 2: server_tool_use must include 'input': {}."""

    @pytest.mark.asyncio
    async def test_server_tool_use_has_input_empty_dict(self):
        """The server_tool_use content_block_start must contain 'input': {}."""
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {"name": "web_search", "arguments": ""},
                                    "type": "function",
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"query":"test"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]

        provider = FakeSearchProvider()
        request = make_request()

        events = await collect_streaming_events(
            convert_openai_streaming_to_claude(
                fake_openai_stream(chunks),
                request,
                logging.getLogger("test"),
                web_search_config={"type": "web_search_20250305"},
                search_provider=provider,
            )
        )
        parsed = parse_sse_events(events)

        # Find the content_block_start with type server_tool_use
        block_starts = [
            (et, d)
            for et, d in parsed
            if et == "content_block_start"
            and d.get("content_block", {}).get("type") == "server_tool_use"
        ]
        assert len(block_starts) >= 1, "Expected at least one server_tool_use block"

        content_block = block_starts[0][1]["content_block"]
        assert "input" in content_block, "server_tool_use must include 'input' field"
        assert content_block["input"] == {}, "server_tool_use input must be empty dict"

    @pytest.mark.asyncio
    async def test_regular_tool_use_still_has_input(self):
        """Non-web-search tool_use blocks should still include 'input': {}."""
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_xyz",
                                    "function": {"name": "get_weather", "arguments": ""},
                                    "type": "function",
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"city":"SF"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
            },
        ]

        request = make_request()
        events = await collect_streaming_events(
            convert_openai_streaming_to_claude(
                fake_openai_stream(chunks),
                request,
                logging.getLogger("test"),
            )
        )
        parsed = parse_sse_events(events)

        block_starts = [
            (et, d)
            for et, d in parsed
            if et == "content_block_start"
            and d.get("content_block", {}).get("type") == "tool_use"
        ]
        assert len(block_starts) >= 1
        assert block_starts[0][1]["content_block"]["input"] == {}


# ---------------------------------------------------------------------------
# Fixes 2-4: non-streaming multiple web search, ordering, dynamic count
# ---------------------------------------------------------------------------


class TestNonStreamingWebSearchOrdering:
    """Spec section 5: tool blocks first, then text."""

    def test_single_search_tool_blocks_before_text(self):
        """With one search, order is: server_tool_use, result, text."""
        resp = openai_response(content="Here is the answer", tool_calls=[])
        request = make_request()
        search_results = [("test query", {"results": [{"type": "web_search_result", "url": "https://example.com", "title": "Ex", "encrypted_content": "content"}]})]

        result = _build_non_streaming_web_search_response(resp, request, search_results)
        content = result["content"]

        assert len(content) == 3
        assert content[0]["type"] == "server_tool_use"
        assert content[1]["type"] == "web_search_tool_result"
        assert content[2]["type"] == "text"

    def test_text_last_even_with_no_text_content(self):
        """When there's no text content, only tool blocks appear."""
        resp = openai_response(content=None, tool_calls=[])
        request = make_request()
        search_results = [("query", {"results": []})]

        result = _build_non_streaming_web_search_response(resp, request, search_results)
        content = result["content"]

        # No text block when content is None
        assert len(content) == 2
        assert content[0]["type"] == "server_tool_use"
        assert content[1]["type"] == "web_search_tool_result"


class TestNonStreamingMultipleSearches:
    """Spec allows multiple server_tool_use/result pairs."""

    def test_multiple_search_results_all_present(self):
        """All searches produce paired server_tool_use + result blocks."""
        resp = openai_response(content="Combined answer", tool_calls=[])
        request = make_request()
        search_results = [
            ("query one", {"results": [{"type": "web_search_result", "url": "https://a.com", "title": "A", "encrypted_content": "a"}]}),
            ("query two", {"results": [{"type": "web_search_result", "url": "https://b.com", "title": "B", "encrypted_content": "b"}]}),
            ("query three", {"results": []}),
        ]

        result = _build_non_streaming_web_search_response(resp, request, search_results)
        content = result["content"]

        # 3 pairs (server_tool_use + result) + 1 text = 7 blocks
        assert len(content) == 7

        # Verify ordering: pairs first, text last
        assert content[0]["type"] == "server_tool_use"
        assert content[0]["input"]["query"] == "query one"
        assert content[1]["type"] == "web_search_tool_result"
        assert content[2]["type"] == "server_tool_use"
        assert content[2]["input"]["query"] == "query two"
        assert content[3]["type"] == "web_search_tool_result"
        assert content[4]["type"] == "server_tool_use"
        assert content[4]["input"]["query"] == "query three"
        assert content[5]["type"] == "web_search_tool_result"
        assert content[6]["type"] == "text"

    def test_tool_use_ids_link_pairs(self):
        """Each server_tool_use id matches its paired result's tool_use_id."""
        resp = openai_response(content="answer", tool_calls=[])
        request = make_request()
        search_results = [
            ("q1", {"results": []}),
            ("q2", {"results": []}),
        ]

        result = _build_non_streaming_web_search_response(resp, request, search_results)
        content = result["content"]

        # First pair
        assert content[0]["id"] == content[1]["tool_use_id"]
        # Second pair
        assert content[2]["id"] == content[3]["tool_use_id"]
        # Different IDs between pairs
        assert content[0]["id"] != content[2]["id"]


class TestNonStreamingDynamicCount:
    """web_search_requests count must match actual number of searches."""

    def test_single_search_count_is_1(self):
        """One search yields web_search_requests: 1."""
        resp = openai_response(content="answer", tool_calls=[])
        request = make_request()
        search_results = [("q", {"results": []})]

        result = _build_non_streaming_web_search_response(resp, request, search_results)
        assert result["usage"]["server_tool_use"]["web_search_requests"] == 1

    def test_three_searches_count_is_3(self):
        """Three searches yield web_search_requests: 3."""
        resp = openai_response(content="answer", tool_calls=[])
        request = make_request()
        search_results = [
            ("q1", {"results": []}),
            ("q2", {"results": []}),
            ("q3", {"results": []}),
        ]

        result = _build_non_streaming_web_search_response(resp, request, search_results)
        assert result["usage"]["server_tool_use"]["web_search_requests"] == 3


class TestNonStreamingSearchErrorContent:
    """Error results are correctly placed in the result block."""

    def test_error_result_in_content(self):
        """When search returns an error, the error dict is the result content."""
        resp = openai_response(content="fallback", tool_calls=[])
        request = make_request()
        error = {"type": "web_search_tool_result_error", "error_code": "unavailable"}
        search_results = [("q", {"error": error})]

        result = _build_non_streaming_web_search_response(resp, request, search_results)
        content = result["content"]

        assert content[1]["type"] == "web_search_tool_result"
        assert content[1]["content"] == error


# ---------------------------------------------------------------------------
# Fix 5: Domain filters passed through in streaming
# ---------------------------------------------------------------------------


class TestStreamingDomainFilters:
    """Domain filters from web_search_config are passed to the search provider."""

    @pytest.mark.asyncio
    async def test_domain_filters_passed_to_provider(self):
        """allowed_domains and blocked_domains reach the search provider."""
        provider = FakeSearchProvider()

        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {"name": "web_search", "arguments": ""},
                                    "type": "function",
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"query":"test"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]

        request = make_request()
        web_search_config = {
            "type": "web_search_20250305",
            "allowed_domains": ["example.com"],
            "blocked_domains": ["blocked.com"],
        }

        await collect_streaming_events(
            convert_openai_streaming_to_claude(
                fake_openai_stream(chunks),
                request,
                logging.getLogger("test"),
                web_search_config=web_search_config,
                search_provider=provider,
            )
        )

        assert provider.last_allowed_domains == ["example.com"]
        assert provider.last_blocked_domains == ["blocked.com"]

    @pytest.mark.asyncio
    async def test_no_domain_filters_passes_none(self):
        """When web_search_config has no domain filters, None is passed."""
        provider = FakeSearchProvider()

        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {"name": "web_search", "arguments": ""},
                                    "type": "function",
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"query":"test"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]

        request = make_request()
        web_search_config = {"type": "web_search_20250305"}

        await collect_streaming_events(
            convert_openai_streaming_to_claude(
                fake_openai_stream(chunks),
                request,
                logging.getLogger("test"),
                web_search_config=web_search_config,
                search_provider=provider,
            )
        )

        assert provider.last_allowed_domains is None
        assert provider.last_blocked_domains is None


# ---------------------------------------------------------------------------
# Streaming: web_search_requests count
# ---------------------------------------------------------------------------


class TestStreamingWebSearchCount:
    """Streaming web_search_requests matches actual search count."""

    @pytest.mark.asyncio
    async def test_single_search_count(self):
        """One web_search call yields web_search_requests: 1 in message_delta."""
        provider = FakeSearchProvider()
        chunks = [
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {"name": "web_search", "arguments": ""},
                                    "type": "function",
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"query":"q1"}'},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-1",
                "choices": [{"delta": {}, "finish_reason": "stop"}],
            },
        ]

        request = make_request()
        events = await collect_streaming_events(
            convert_openai_streaming_to_claude(
                fake_openai_stream(chunks),
                request,
                logging.getLogger("test"),
                web_search_config={"type": "web_search_20250305"},
                search_provider=provider,
            )
        )
        parsed = parse_sse_events(events)

        # Find message_delta
        msg_deltas = [d for et, d in parsed if et == "message_delta"]
        assert len(msg_deltas) == 1
        usage = msg_deltas[0].get("usage", {})
        assert usage.get("server_tool_use", {}).get("web_search_requests") == 1
