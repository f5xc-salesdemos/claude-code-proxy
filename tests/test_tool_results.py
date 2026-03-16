"""Unit tests for convert_claude_tool_results() in request_converter."""

from src.conversion.request_converter import convert_claude_tool_results
from src.models.claude import ClaudeMessage


class TestConvertClaudeToolResults:
    """Tests for convert_claude_tool_results covering various inputs."""

    def test_single_tool_result(self):
        """Single tool_result block is converted correctly."""
        msg = ClaudeMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "result text",
                }
            ],
        )
        results = convert_claude_tool_results(msg)
        assert len(results) == 1
        assert results[0]["role"] == "tool"
        assert results[0]["tool_call_id"] == "call_1"
        assert results[0]["content"] == "result text"

    def test_multiple_tool_results(self):
        """Multiple tool_result blocks are all converted."""
        msg = ClaudeMessage(
            role="user",
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "first",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_2",
                    "content": "second",
                },
            ],
        )
        results = convert_claude_tool_results(msg)
        assert len(results) == 2
        assert results[0]["tool_call_id"] == "call_1"
        assert results[1]["tool_call_id"] == "call_2"

    def test_mixed_content_filters_non_tool_result(self):
        """Non-tool_result blocks in the content list are filtered out."""
        msg = ClaudeMessage(
            role="user",
            content=[
                {"type": "text", "text": "some text"},
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "result",
                },
            ],
        )
        results = convert_claude_tool_results(msg)
        assert len(results) == 1
        assert results[0]["tool_call_id"] == "call_1"

    def test_string_content_returns_empty(self):
        """String content (not a list) returns empty list."""
        msg = ClaudeMessage(role="user", content="just a string")
        results = convert_claude_tool_results(msg)
        assert results == []

    def test_none_content_returns_empty(self):
        """None content returns empty list."""
        msg = ClaudeMessage(role="user", content=None)
        results = convert_claude_tool_results(msg)
        assert results == []
