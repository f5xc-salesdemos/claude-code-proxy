"""Tests for empty content sanitization — prevents LiteLLM placeholder leaks.

LiteLLM's ``_sanitize_empty_text_content()`` replaces empty/whitespace-only
string content in "user" and "assistant" messages with a visible placeholder:

    [System: Empty message content sanitised to satisfy protocol]

These tests verify that the proxy:
1. Never emits empty text deltas or empty text blocks (Fix A / A2)
2. Strips known placeholder strings from incoming request history (Fix B / C)
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.conversion.request_converter import (
    convert_claude_assistant_message,
    convert_claude_user_message,
)
from src.conversion.response_converter import (
    convert_openai_streaming_to_claude,
    convert_openai_streaming_to_claude_with_cancellation,
    convert_openai_to_claude_response,
)
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

LITELLM_PLACEHOLDER = "[System: Empty message content sanitised to satisfy protocol]"
NO_CONTENT_PLACEHOLDER = "[no content]"


def _make_request(**overrides) -> ClaudeMessagesRequest:
    """Build a minimal ClaudeMessagesRequest for testing."""
    defaults = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(overrides)
    return ClaudeMessagesRequest(**defaults)


def _openai_response(content=None, tool_calls=None, finish_reason="stop"):
    """Build a minimal OpenAI non-streaming response dict."""
    message = {}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


async def _collect_streaming_events(async_gen):
    """Collect all SSE events from the streaming generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


def _parse_sse_events(raw_events):
    """Parse raw SSE strings into (event_type, data_dict) tuples."""
    parsed = []
    for raw in raw_events:
        lines = raw.strip().split("\n")
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        if event_type and data:
            parsed.append((event_type, data))
    return parsed


async def _fake_openai_stream(chunks):
    """Yield SSE-formatted lines from a list of chunk dicts."""
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}"
    yield "data: [DONE]"


# ===========================================================================
# 1. Streaming empty content filtering (Fix A)
# ===========================================================================


class TestStreamingEmptyContentFiltering:
    """Empty string deltas must not open text blocks or produce events."""

    @pytest.mark.asyncio
    async def test_empty_string_delta_does_not_start_text_block(self):
        """An empty-string content delta must not emit content_block_start."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {"content": ""},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ]
            },
        ]
        request = _make_request()
        logger = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude(_fake_openai_stream(chunks), request, logger)
        )
        parsed = _parse_sse_events(events)

        # Should not have any content_block_start for text
        text_block_starts = [
            (et, d)
            for et, d in parsed
            if et == "content_block_start" and d.get("content_block", {}).get("type") == "text"
        ]
        assert text_block_starts == [], "Empty string delta should not open a text block"

    @pytest.mark.asyncio
    async def test_empty_string_delta_does_not_produce_text_delta(self):
        """An empty-string content delta must not emit content_block_delta."""
        chunks = [
            {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude(_fake_openai_stream(chunks), request, logger)
        )
        parsed = _parse_sse_events(events)

        text_deltas = [
            d
            for et, d in parsed
            if et == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
        ]
        # Only the "hello" delta should appear
        assert len(text_deltas) == 1
        assert text_deltas[0]["delta"]["text"] == "hello"

    @pytest.mark.asyncio
    async def test_none_content_delta_is_also_skipped(self):
        """A None content delta (already handled) must not emit text events."""
        chunks = [
            {"choices": [{"delta": {"content": None}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude(_fake_openai_stream(chunks), request, logger)
        )
        parsed = _parse_sse_events(events)

        text_events = [
            (et, d)
            for et, d in parsed
            if et in ("content_block_start", "content_block_delta")
            and (
                d.get("content_block", {}).get("type") == "text"
                or d.get("delta", {}).get("type") == "text_delta"
            )
        ]
        assert text_events == []

    @pytest.mark.asyncio
    async def test_real_content_after_empty_still_works(self):
        """Real text appearing after empty deltas must still be emitted."""
        chunks = [
            {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude(_fake_openai_stream(chunks), request, logger)
        )
        parsed = _parse_sse_events(events)

        text_deltas = [
            d["delta"]["text"]
            for et, d in parsed
            if et == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
        ]
        assert text_deltas == ["Hi"]

    @pytest.mark.asyncio
    async def test_tool_only_stream_no_text_block(self):
        """A stream with only tool calls must not emit any text blocks."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"city":"NYC"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        request = _make_request()
        logger = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude(_fake_openai_stream(chunks), request, logger)
        )
        parsed = _parse_sse_events(events)

        text_block_starts = [
            d
            for et, d in parsed
            if et == "content_block_start" and d.get("content_block", {}).get("type") == "text"
        ]
        assert text_block_starts == [], "Tool-only stream should not emit text blocks"


class TestStreamingWithCancellationEmptyContent:
    """Same empty-string filtering for the cancellation-aware variant."""

    @pytest.mark.asyncio
    async def test_empty_string_delta_filtered_with_cancellation(self):
        """Empty string deltas must be filtered in the cancellation variant."""
        chunks = [
            {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "world"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()

        # Mock the http_request with is_disconnected returning False
        http_request = AsyncMock()
        http_request.is_disconnected = AsyncMock(return_value=False)
        openai_client = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _fake_openai_stream(chunks),
                request,
                logger,
                http_request,
                openai_client,
                "req-123",
            )
        )
        parsed = _parse_sse_events(events)

        text_deltas = [
            d["delta"]["text"]
            for et, d in parsed
            if et == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
        ]
        assert text_deltas == ["world"]

    @pytest.mark.asyncio
    async def test_only_empty_deltas_no_text_block_with_cancellation(self):
        """Stream of only empty deltas should produce no text block."""
        chunks = [
            {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()
        http_request = AsyncMock()
        http_request.is_disconnected = AsyncMock(return_value=False)
        openai_client = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _fake_openai_stream(chunks),
                request,
                logger,
                http_request,
                openai_client,
                "req-456",
            )
        )
        parsed = _parse_sse_events(events)

        text_blocks = [
            d
            for et, d in parsed
            if et == "content_block_start" and d.get("content_block", {}).get("type") == "text"
        ]
        assert text_blocks == []


# ===========================================================================
# 2. Request placeholder stripping (Fix B + C)
# ===========================================================================


class TestUserMessagePlaceholderStripping:
    """LiteLLM's placeholder must be stripped from user messages."""

    def test_litellm_placeholder_string_content(self):
        """String content equal to the LiteLLM placeholder returns empty."""
        msg = ClaudeMessage(role="user", content=LITELLM_PLACEHOLDER)
        result = convert_claude_user_message(msg)
        assert result["content"] == ""

    def test_litellm_placeholder_with_whitespace(self):
        """Placeholder with surrounding whitespace is also stripped."""
        msg = ClaudeMessage(role="user", content=f"  {LITELLM_PLACEHOLDER}  ")
        result = convert_claude_user_message(msg)
        assert result["content"] == ""

    def test_no_content_placeholder_string(self):
        """Our own '[no content]' placeholder is also stripped."""
        msg = ClaudeMessage(role="user", content=NO_CONTENT_PLACEHOLDER)
        result = convert_claude_user_message(msg)
        assert result["content"] == ""

    def test_normal_user_message_unchanged(self):
        """Normal user text must pass through unmodified."""
        msg = ClaudeMessage(role="user", content="Hello, world!")
        result = convert_claude_user_message(msg)
        assert result["content"] == "Hello, world!"

    def test_placeholder_in_list_content_block(self):
        """Placeholder text blocks in list content are skipped."""
        msg = ClaudeMessage(
            role="user",
            content=[
                {"type": "text", "text": LITELLM_PLACEHOLDER},
                {"type": "text", "text": "actual question"},
            ],
        )
        result = convert_claude_user_message(msg)
        # Single remaining text block collapses to a plain string
        assert result["content"] == "actual question"

    def test_all_placeholder_blocks_returns_empty(self):
        """If all text blocks are placeholders, content should be empty string."""
        msg = ClaudeMessage(
            role="user",
            content=[
                {"type": "text", "text": LITELLM_PLACEHOLDER},
            ],
        )
        result = convert_claude_user_message(msg)
        # With the empty-list guard, this must collapse to an empty string
        assert result["content"] == ""


class TestAssistantMessagePlaceholderStripping:
    """Placeholders must be stripped from assistant message history."""

    def test_litellm_placeholder_string_returns_none(self):
        """String content equal to placeholder returns content=None."""
        msg = ClaudeMessage(role="assistant", content=LITELLM_PLACEHOLDER)
        result = convert_claude_assistant_message(msg)
        assert result["message"]["content"] is None

    def test_no_content_placeholder_string_returns_none(self):
        """'[no content]' placeholder returns content=None."""
        msg = ClaudeMessage(role="assistant", content=NO_CONTENT_PLACEHOLDER)
        result = convert_claude_assistant_message(msg)
        assert result["message"]["content"] is None

    def test_placeholder_text_block_skipped_in_list(self):
        """Placeholder text blocks in list content are excluded."""
        msg = ClaudeMessage(
            role="assistant",
            content=[
                {"type": "text", "text": LITELLM_PLACEHOLDER},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "get_weather",
                    "input": {"city": "NYC"},
                },
            ],
        )
        result = convert_claude_assistant_message(msg)
        # Content should be None (no valid text parts), tool_calls should exist
        assert result["message"]["content"] is None
        assert len(result["message"]["tool_calls"]) == 1

    def test_normal_assistant_message_unchanged(self):
        """Normal assistant text must pass through unmodified."""
        msg = ClaudeMessage(role="assistant", content="Here is the answer.")
        result = convert_claude_assistant_message(msg)
        assert result["message"]["content"] == "Here is the answer."


# ===========================================================================
# 3. Non-streaming empty content (Fix A2)
# ===========================================================================


class TestNonStreamingEmptyContent:
    """Empty string content from upstream must not create empty text blocks."""

    def test_empty_string_content_with_tool_calls(self):
        """Empty string content should be omitted when tool calls exist."""
        openai_resp = _openai_response(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path": "/tmp/x"}',
                    },
                }
            ],
            finish_reason="tool_calls",
        )
        request = _make_request()
        result = convert_openai_to_claude_response(openai_resp, request)

        # Should have tool_use block but no text block
        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        tool_blocks = [b for b in result["content"] if b["type"] == "tool_use"]
        assert text_blocks == [], "Empty string should not create text block"
        assert len(tool_blocks) == 1

    def test_whitespace_only_content_with_tool_calls(self):
        """Whitespace-only content should be omitted when tool calls exist."""
        openai_resp = _openai_response(
            content="   \n  ",
            tool_calls=[
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "write_file",
                        "arguments": "{}",
                    },
                }
            ],
            finish_reason="tool_calls",
        )
        request = _make_request()
        result = convert_openai_to_claude_response(openai_resp, request)

        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert text_blocks == [], "Whitespace-only content should not create text block"

    def test_real_content_still_included(self):
        """Non-empty text content should still produce a text block."""
        openai_resp = _openai_response(content="Hello there")
        request = _make_request()
        result = convert_openai_to_claude_response(openai_resp, request)

        text_blocks = [b for b in result["content"] if b["type"] == "text"]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Hello there"

    def test_none_content_no_tool_calls_gets_placeholder(self):
        """None content with no tool calls should still get the fallback."""
        openai_resp = _openai_response(content=None, tool_calls=None)
        request = _make_request()
        result = convert_openai_to_claude_response(openai_resp, request)

        # Should have the [no content] fallback
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "[no content]"


# ===========================================================================
# 4. Full round-trip: stream -> history -> next request
# ===========================================================================


class TestFullRoundTrip:
    """Simulate streaming response -> reconstructed history -> next request."""

    @pytest.mark.asyncio
    async def test_empty_deltas_do_not_pollute_round_trip(self):
        """A stream with empty deltas should produce no text in the
        assistant history that LiteLLM could sanitize."""
        # Simulate: upstream model returns empty content + tool call
        chunks = [
            {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_abc",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": "",
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"path":"/tmp"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        request = _make_request()
        logger = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude(_fake_openai_stream(chunks), request, logger)
        )
        parsed = _parse_sse_events(events)

        # Extract any text content that was emitted
        text_deltas = [
            d["delta"]["text"]
            for et, d in parsed
            if et == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
        ]
        # No text should have been streamed
        assert text_deltas == []

        # Now simulate: if LiteLLM had injected the placeholder into the
        # assistant message history, our request converter should strip it
        polluted_msg = ClaudeMessage(
            role="assistant",
            content=[
                {"type": "text", "text": LITELLM_PLACEHOLDER},
                {
                    "type": "tool_use",
                    "id": "call_abc",
                    "name": "read_file",
                    "input": {"path": "/tmp"},
                },
            ],
        )
        result = convert_claude_assistant_message(polluted_msg)
        assert (
            result["message"]["content"] is None
        ), "Placeholder text should be stripped from assistant history"
        assert len(result["message"]["tool_calls"]) == 1

    @pytest.mark.asyncio
    async def test_placeholder_stripped_from_user_in_round_trip(self):
        """If LiteLLM pollutes a user message, the proxy strips it."""
        polluted_user = ClaudeMessage(role="user", content=LITELLM_PLACEHOLDER)
        result = convert_claude_user_message(polluted_user)
        # Should be empty, not the placeholder
        assert LITELLM_PLACEHOLDER not in str(result["content"])


# ===========================================================================
# 5. Whitespace-only streaming delta filtering (M3)
# ===========================================================================


class TestStreamingWhitespaceOnlyDelta:
    """Whitespace-only deltas must not open text blocks or produce events."""

    @pytest.mark.asyncio
    async def test_whitespace_only_delta_does_not_start_text_block(self):
        """A whitespace-only content delta must not emit content_block_start."""
        chunks = [
            {"choices": [{"delta": {"content": "   "}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude(_fake_openai_stream(chunks), request, logger)
        )
        parsed = _parse_sse_events(events)

        text_block_starts = [
            (et, d)
            for et, d in parsed
            if et == "content_block_start" and d.get("content_block", {}).get("type") == "text"
        ]
        assert text_block_starts == [], "Whitespace-only delta should not open a text block"

    @pytest.mark.asyncio
    async def test_whitespace_only_delta_filtered_with_cancellation(self):
        """Whitespace-only deltas must be filtered in the cancellation variant."""
        chunks = [
            {"choices": [{"delta": {"content": "   "}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()
        http_request = AsyncMock()
        http_request.is_disconnected = AsyncMock(return_value=False)
        openai_client = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _fake_openai_stream(chunks),
                request,
                logger,
                http_request,
                openai_client,
                "req-ws-1",
            )
        )
        parsed = _parse_sse_events(events)

        text_deltas = [
            d["delta"]["text"]
            for et, d in parsed
            if et == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
        ]
        # Only "hello" should appear; whitespace-only "   " is filtered
        assert text_deltas == ["hello"]


# ===========================================================================
# 6. Non-streaming empty string with no tool calls (M4)
# ===========================================================================


class TestNonStreamingEmptyNoToolCalls:
    """Empty string content with no tool calls should get the fallback."""

    def test_empty_string_no_tool_calls_gets_placeholder(self):
        """Empty string content with no tool calls → [no content] fallback."""
        openai_resp = _openai_response(content="", tool_calls=None)
        request = _make_request()
        result = convert_openai_to_claude_response(openai_resp, request)

        # Should have the [no content] fallback since no text and no tool calls
        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "[no content]"

    def test_whitespace_only_no_tool_calls_gets_placeholder(self):
        """Whitespace-only content with no tool calls → [no content] fallback."""
        openai_resp = _openai_response(content="  \n  ", tool_calls=None)
        request = _make_request()
        result = convert_openai_to_claude_response(openai_resp, request)

        assert len(result["content"]) == 1
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "[no content]"


# ===========================================================================
# 7. Cancellation variant parity tests (M5)
# ===========================================================================


class TestCancellationVariantParity:
    """Ensure the cancellation streaming variant has parity with the main one."""

    @pytest.mark.asyncio
    async def test_none_content_delta_filtered_with_cancellation(self):
        """A None content delta must not emit text events in cancellation variant."""
        chunks = [
            {"choices": [{"delta": {"content": None}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()
        http_request = AsyncMock()
        http_request.is_disconnected = AsyncMock(return_value=False)
        openai_client = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _fake_openai_stream(chunks),
                request,
                logger,
                http_request,
                openai_client,
                "req-none-1",
            )
        )
        parsed = _parse_sse_events(events)

        text_events = [
            (et, d)
            for et, d in parsed
            if et in ("content_block_start", "content_block_delta")
            and (
                d.get("content_block", {}).get("type") == "text"
                or d.get("delta", {}).get("type") == "text_delta"
            )
        ]
        assert text_events == []

    @pytest.mark.asyncio
    async def test_real_content_after_empty_with_cancellation(self):
        """Real text after empty deltas must still be emitted in cancellation variant."""
        chunks = [
            {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": ""}, "finish_reason": None}]},
            {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ]
        request = _make_request()
        logger = MagicMock()
        http_request = AsyncMock()
        http_request.is_disconnected = AsyncMock(return_value=False)
        openai_client = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _fake_openai_stream(chunks),
                request,
                logger,
                http_request,
                openai_client,
                "req-real-1",
            )
        )
        parsed = _parse_sse_events(events)

        text_deltas = [
            d["delta"]["text"]
            for et, d in parsed
            if et == "content_block_delta" and d.get("delta", {}).get("type") == "text_delta"
        ]
        assert text_deltas == ["Hi"]

    @pytest.mark.asyncio
    async def test_tool_only_stream_no_text_block_with_cancellation(self):
        """A stream with only tool calls must not emit text blocks in cancellation variant."""
        chunks = [
            {
                "choices": [
                    {
                        "delta": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_cancel_1",
                                    "function": {
                                        "name": "get_weather",
                                        "arguments": "",
                                    },
                                }
                            ],
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {
                                        "arguments": '{"city":"NYC"}',
                                    },
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ]
            },
            {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        ]
        request = _make_request()
        logger = MagicMock()
        http_request = AsyncMock()
        http_request.is_disconnected = AsyncMock(return_value=False)
        openai_client = MagicMock()

        events = await _collect_streaming_events(
            convert_openai_streaming_to_claude_with_cancellation(
                _fake_openai_stream(chunks),
                request,
                logger,
                http_request,
                openai_client,
                "req-tool-1",
            )
        )
        parsed = _parse_sse_events(events)

        text_block_starts = [
            d
            for et, d in parsed
            if et == "content_block_start" and d.get("content_block", {}).get("type") == "text"
        ]
        assert (
            text_block_starts == []
        ), "Tool-only stream should not emit text blocks in cancellation variant"
