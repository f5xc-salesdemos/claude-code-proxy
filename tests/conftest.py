"""Shared fixtures and helpers for the test suite."""

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from src.models.claude import ClaudeMessagesRequest
from src.services.search.base import SearchProvider

# Re-export List so conftest users don't need to import it separately

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


class FakeSearchProvider(SearchProvider):
    """In-memory search provider for integration tests.

    By default returns a single canned result.  Override
    ``results`` or ``error`` to customise behaviour.
    """

    def __init__(
        self,
        *,
        available: bool = True,
        results: Optional[List[Dict[str, Any]]] = None,
        error: Optional[Dict[str, Any]] = None,
    ):
        self._available = available
        self._error = error
        self._results = results if results is not None else [
            {
                "type": "web_search_result",
                "url": "https://example.com",
                "title": "Example",
                "encrypted_content": "Example content",
            }
        ]

    @property
    def provider_name(self) -> str:
        return "fake"

    async def is_available(self) -> bool:
        return self._available

    async def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        # Store the last call's domain filters for test assertions
        self.last_allowed_domains = allowed_domains
        self.last_blocked_domains = blocked_domains
        if self._error:
            return {"error": self._error}
        return {"results": self._results[:max_results]}

    async def close(self) -> None:
        pass


@pytest.fixture()
def default_request():
    """Return a default ``ClaudeMessagesRequest``."""
    return make_request()


@pytest.fixture()
def default_model_manager():
    """Return a mock ``ModelManager`` mapping to ``gpt-4o``."""
    return mock_model_manager()


@pytest.fixture()
def fake_search_provider():
    """Return a FakeSearchProvider with default canned results."""
    return FakeSearchProvider()
