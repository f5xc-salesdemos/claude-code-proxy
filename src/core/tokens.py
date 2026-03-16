"""Pure-function token estimation utilities.

Provides a simple character-based heuristic (4 chars ≈ 1 token) that mirrors
the logic in the ``/v1/messages/count_tokens`` endpoint without importing
any FastAPI or application-startup dependencies.
"""

import json
from typing import Any, List, Optional


def estimate_tokens(
    messages: List[Any],
    system: Any = None,
    tools: Any = None,
) -> int:
    """Estimate the number of input tokens for a conversation.

    Uses a 4-characters-per-token heuristic identical to the existing
    ``count_tokens`` endpoint.  Image content blocks are intentionally
    skipped — counting base64 data would produce wildly inflated estimates.

    Parameters
    ----------
    messages:
        List of message objects.  Each object must expose a ``content``
        attribute (or ``["content"]`` key) that is either a plain string or
        a list of content blocks.  Content blocks are counted only when they
        have a ``text`` attribute/key; other block types (image, tool_use,
        tool_result …) are silently ignored.
    system:
        Optional system prompt.  Accepted as:
        * a plain ``str``
        * a list of objects with a ``text`` attribute (``ClaudeSystemContent``)
        * ``None`` — ignored
    tools:
        Optional list of tool definitions.  Each tool's ``input_schema`` is
        JSON-serialised and its length added to the character total.  Accepted
        as a list of objects with ``name``, optional ``description``, and
        ``input_schema`` attributes, or plain dicts with the same keys.

    Returns
    -------
    int
        Estimated token count — always at least 1.
    """
    total_chars = 0

    # ------------------------------------------------------------------ system
    if system is not None:
        if isinstance(system, str):
            total_chars += len(system)
        elif isinstance(system, list):
            for block in system:
                text = _get_text(block)
                if text is not None:
                    total_chars += len(text)

    # ---------------------------------------------------------------- messages
    for msg in messages:
        content = _get_content(msg)
        if content is None:
            continue
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for block in content:
                text = _get_text(block)
                if text is not None:
                    total_chars += len(text)

    # ------------------------------------------------------------------ tools
    if tools is not None:
        for tool in tools:
            schema = _get_input_schema(tool)
            if schema is not None:
                total_chars += len(json.dumps(schema))
            name = _get_attr_or_key(tool, "name")
            if name:
                total_chars += len(str(name))
            description = _get_attr_or_key(tool, "description")
            if description:
                total_chars += len(str(description))

    # 4 characters ≈ 1 token; floor at 1
    return max(1, total_chars // 4)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_attr_or_key(obj: Any, key: str) -> Any:
    """Return ``obj.key`` if it exists, else ``obj[key]``, else ``None``."""
    if hasattr(obj, key):
        return getattr(obj, key)
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _get_text(block: Any) -> Optional[str]:
    """Extract text from a content block, or return None for non-text blocks."""
    text = _get_attr_or_key(block, "text")
    if isinstance(text, str):
        return text
    return None


def _get_content(msg: Any) -> Any:
    """Extract the ``content`` field from a message object or dict."""
    return _get_attr_or_key(msg, "content")


def _get_input_schema(tool: Any) -> Any:
    """Extract the ``input_schema`` from a tool definition."""
    return _get_attr_or_key(tool, "input_schema")
