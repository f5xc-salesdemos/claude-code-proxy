"""Shared fixtures and helpers for the test suite."""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.models.claude import ClaudeMessagesRequest

# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_request(**overrides: Any) -> ClaudeMessagesRequest:
    """Build a minimal ``ClaudeMessagesRequest`` with sensible defaults.

    Any keyword argument overrides the corresponding default field.
    """
    defaults: Dict[str, Any] = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(overrides)
    return ClaudeMessagesRequest(**defaults)


def mock_model_manager(model_name: str = "gpt-4o") -> MagicMock:
    """Build a mock ``ModelManager`` that returns a fixed model name."""
    mm = MagicMock()
    mm.map_claude_model_to_openai.return_value = model_name
    return mm


def openai_response(
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: str = "stop",
) -> Dict[str, Any]:
    """Build a minimal OpenAI non-streaming response dict."""
    message: Dict[str, Any] = {}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {
        "id": "chatcmpl-test",
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------


async def fake_openai_stream(chunks: List[Dict[str, Any]]):
    """Yield SSE-formatted lines from a list of chunk dicts."""
    for chunk in chunks:
        yield f"data: {json.dumps(chunk)}"
    yield "data: [DONE]"


async def collect_streaming_events(async_gen):
    """Collect all SSE events from an async streaming generator."""
    events = []
    async for event in async_gen:
        events.append(event)
    return events


def parse_sse_events(raw_events: List[str]):
    """Parse raw SSE strings into ``(event_type, data_dict)`` tuples."""
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


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_request():
    """Return a default ``ClaudeMessagesRequest``."""
    return make_request()


@pytest.fixture()
def default_model_manager():
    """Return a mock ``ModelManager`` mapping to ``gpt-4o``."""
    return mock_model_manager()
