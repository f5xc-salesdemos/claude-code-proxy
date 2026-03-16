"""Unit tests for src.core.tokens.estimate_tokens."""

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from src.core.tokens import estimate_tokens


# ---------------------------------------------------------------------------
# Helpers: lightweight stand-ins for Pydantic models
# ---------------------------------------------------------------------------


def _msg(content: Any) -> SimpleNamespace:
    """Build a minimal message-like object."""
    return SimpleNamespace(content=content)


def _text_block(text: str) -> SimpleNamespace:
    """Build a text content block."""
    return SimpleNamespace(type="text", text=text)


def _image_block() -> SimpleNamespace:
    """Build an image content block (no ``text`` attribute)."""
    return SimpleNamespace(
        type="image",
        source=SimpleNamespace(
            type="base64",
            media_type="image/png",
            data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDQAHGGF2JgAAAABJRU5ErkJggg==",
        ),
    )


def _sys_block(text: str) -> SimpleNamespace:
    """Build a system content block with a ``text`` attribute."""
    return SimpleNamespace(type="text", text=text)


def _tool(name: str, description: Optional[str], input_schema: Dict[str, Any]) -> SimpleNamespace:
    """Build a tool definition object."""
    return SimpleNamespace(name=name, description=description, input_schema=input_schema)


# ---------------------------------------------------------------------------
# Tests: empty / minimal input
# ---------------------------------------------------------------------------


def test_empty_messages():
    """Empty message list with no system or tools returns 1 (floor)."""
    result = estimate_tokens([])
    assert result == 1


def test_none_content_message():
    """Messages with None content contribute 0 characters."""
    result = estimate_tokens([_msg(None)])
    assert result == 1


def test_empty_string_content():
    """A message with an empty string contributes 0 chars → floor to 1."""
    result = estimate_tokens([_msg("")])
    assert result == 1


# ---------------------------------------------------------------------------
# Tests: text message estimation
# ---------------------------------------------------------------------------


def test_text_message_estimation():
    """Simple string content: chars / 4."""
    # 8 chars → 2 tokens
    result = estimate_tokens([_msg("12345678")])
    assert result == 2


def test_text_message_estimation_formula():
    """Verify formula: max(1, total_chars // 4)."""
    # "hello" = 5 chars → 5 // 4 = 1
    assert estimate_tokens([_msg("hello")]) == 1
    # 100-char string → 25 tokens
    text = "a" * 100
    assert estimate_tokens([_msg(text)]) == 25


def test_multiple_messages_summed():
    """Characters from multiple messages are summed."""
    msgs = [_msg("abcd"), _msg("efgh")]  # 8 chars total → 2 tokens
    assert estimate_tokens(msgs) == 2


def test_list_content_text_blocks():
    """Content as list of text blocks: each block's text is counted."""
    content = [_text_block("abcd"), _text_block("efgh")]  # 8 chars → 2
    result = estimate_tokens([_msg(content)])
    assert result == 2


def test_dict_message_content():
    """Messages passed as dicts (not objects) are handled correctly."""
    msg = {"content": "abcdefgh"}  # 8 chars → 2
    assert estimate_tokens([msg]) == 2


# ---------------------------------------------------------------------------
# Tests: image blocks skipped
# ---------------------------------------------------------------------------


def test_image_blocks_skipped():
    """Image content blocks must NOT contribute to the token count."""
    # Only the text block's 4 chars count → 1 token
    content = [_text_block("abcd"), _image_block()]
    result = estimate_tokens([_msg(content)])
    assert result == 1


def test_only_image_block_gives_floor():
    """A message containing only an image block yields the floor of 1."""
    result = estimate_tokens([_msg([_image_block()])])
    assert result == 1


def test_mixed_text_and_image():
    """Multiple text blocks and image blocks: only text chars counted."""
    content = [
        _image_block(),
        _text_block("aaaa"),  # 4 chars
        _image_block(),
        _text_block("bbbb"),  # 4 chars
    ]
    # 8 total text chars → 2 tokens
    assert estimate_tokens([_msg(content)]) == 2


# ---------------------------------------------------------------------------
# Tests: system prompt counted
# ---------------------------------------------------------------------------


def test_system_and_tools_counted():
    """System text and tool definitions are included in the estimate."""
    system_text = "a" * 40  # 40 chars
    tool = _tool(
        name="t",           # 1 char
        description=None,
        input_schema={"type": "object"},  # json → '{"type": "object"}' = 18 chars
    )
    # name=1, schema=18, system=40 → 59 chars // 4 = 14
    result = estimate_tokens([], system=system_text, tools=[tool])
    expected = max(1, (40 + 1 + len(json.dumps({"type": "object"}))) // 4)
    assert result == expected


def test_system_as_string():
    """System prompt as a plain string is counted."""
    # 16 chars → 4 tokens
    result = estimate_tokens([], system="a" * 16)
    assert result == 4


def test_system_as_list_of_blocks():
    """System prompt as a list of text blocks: each block's text is counted."""
    blocks = [_sys_block("aaaa"), _sys_block("bbbb")]  # 8 chars → 2 tokens
    result = estimate_tokens([], system=blocks)
    assert result == 2


def test_system_none_no_contribution():
    """system=None does not add any characters."""
    result = estimate_tokens([_msg("abcd")], system=None)
    assert result == 1  # 4 chars // 4 = 1


# ---------------------------------------------------------------------------
# Tests: tools counted
# ---------------------------------------------------------------------------


def test_tools_schema_counted():
    """Tool input_schema is JSON-serialised and its length counted."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    tool = _tool(name="foo", description="bar", input_schema=schema)
    schema_chars = len(json.dumps(schema))
    name_chars = len("foo")
    desc_chars = len("bar")
    expected = max(1, (schema_chars + name_chars + desc_chars) // 4)
    result = estimate_tokens([], tools=[tool])
    assert result == expected


def test_tools_as_dicts():
    """Tools passed as plain dicts are handled the same as objects."""
    schema = {"type": "object"}
    tool_dict = {"name": "mytool", "description": "does stuff", "input_schema": schema}
    schema_chars = len(json.dumps(schema))
    name_chars = len("mytool")
    desc_chars = len("does stuff")
    expected = max(1, (schema_chars + name_chars + desc_chars) // 4)
    result = estimate_tokens([], tools=[tool_dict])
    assert result == expected


def test_tools_none_no_contribution():
    """tools=None does not add any characters."""
    result = estimate_tokens([_msg("abcd")], tools=None)
    assert result == 1


def test_multiple_tools_summed():
    """Characters from multiple tools are accumulated."""
    schema = {"type": "object"}
    tools = [
        _tool(name="a", description=None, input_schema=schema),
        _tool(name="b", description=None, input_schema=schema),
    ]
    schema_chars = len(json.dumps(schema))
    # name "a" (1) + schema + name "b" (1) + schema
    expected = max(1, (1 + schema_chars + 1 + schema_chars) // 4)
    result = estimate_tokens([], tools=tools)
    assert result == expected


# ---------------------------------------------------------------------------
# Tests: combined input
# ---------------------------------------------------------------------------


def test_all_inputs_combined():
    """Messages + system + tools all contribute to the final estimate."""
    msg = _msg("abcdefgh")            # 8 chars
    system = "sys "                   # 4 chars
    schema = {"type": "object"}       # e.g. 18 chars when serialised
    tool = _tool(name="t", description=None, input_schema=schema)
    schema_chars = len(json.dumps(schema))
    total = 8 + 4 + 1 + schema_chars  # messages + system + name + schema
    expected = max(1, total // 4)
    result = estimate_tokens([msg], system=system, tools=[tool])
    assert result == expected


def test_floor_is_one():
    """Result is always at least 1, even for 0-character inputs."""
    assert estimate_tokens([]) == 1
    assert estimate_tokens([_msg(None)]) == 1
    assert estimate_tokens([_msg([])]) == 1
