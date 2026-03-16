"""Unit tests for parse_tool_result_content() in request_converter."""

import json

from src.conversion.request_converter import parse_tool_result_content


class TestParseToolResultContent:
    """Tests for parse_tool_result_content covering all type paths."""

    def test_none_returns_placeholder(self):
        """None content returns 'No content provided'."""
        assert parse_tool_result_content(None) == "No content provided"

    def test_string_passthrough(self):
        """String content is returned as-is."""
        assert parse_tool_result_content("hello world") == "hello world"

    def test_empty_string(self):
        """Empty string is returned as-is."""
        assert parse_tool_result_content("") == ""

    def test_list_of_text_blocks(self):
        """List of text-type dicts extracts text fields."""
        content = [
            {"type": "text", "text": "line one"},
            {"type": "text", "text": "line two"},
        ]
        result = parse_tool_result_content(content)
        assert result == "line one\nline two"

    def test_list_of_strings(self):
        """List of plain strings are joined."""
        content = ["foo", "bar"]
        result = parse_tool_result_content(content)
        assert result == "foo\nbar"

    def test_list_of_dicts_with_text_key(self):
        """List of dicts with 'text' key (but not text type) extracts text."""
        content = [{"text": "hello", "extra": "ignored"}]
        result = parse_tool_result_content(content)
        assert result == "hello"

    def test_list_of_dicts_without_text(self):
        """List of dicts without 'text' key are JSON-serialized."""
        content = [{"key": "value"}]
        result = parse_tool_result_content(content)
        assert json.loads(result) == {"key": "value"}

    def test_list_mixed_types(self):
        """Mixed list of strings, text blocks, and dicts."""
        content = [
            {"type": "text", "text": "typed"},
            "plain string",
            {"other": 123},
        ]
        result = parse_tool_result_content(content)
        lines = result.split("\n")
        assert lines[0] == "typed"
        assert lines[1] == "plain string"
        assert json.loads(lines[2]) == {"other": 123}

    def test_dict_text_type(self):
        """Dict with type=text returns the text field."""
        content = {"type": "text", "text": "hello"}
        assert parse_tool_result_content(content) == "hello"

    def test_dict_other_type(self):
        """Dict with non-text type is JSON-serialized."""
        content = {"type": "image", "url": "http://example.com"}
        result = parse_tool_result_content(content)
        assert json.loads(result) == content

    def test_arbitrary_object(self):
        """Arbitrary objects are converted via str()."""
        result = parse_tool_result_content(42)
        assert result == "42"

    def test_empty_list(self):
        """Empty list returns empty string (stripped)."""
        result = parse_tool_result_content([])
        assert result == ""

    def test_list_with_empty_text_block(self):
        """Text block with empty text."""
        content = [{"type": "text", "text": ""}]
        result = parse_tool_result_content(content)
        assert result == ""
