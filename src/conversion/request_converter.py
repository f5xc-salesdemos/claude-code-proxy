"""Convert Claude Messages API requests to OpenAI Chat Completions format."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from src.core.config import config
from src.core.constants import Constants
from src.models.claude import ClaudeMessage, ClaudeMessagesRequest, ClaudeTool

logger = logging.getLogger(__name__)

# Known placeholder strings injected by LiteLLM or our own proxy that
# should be stripped from incoming conversation history to prevent
# "[System: Empty message content sanitised to satisfy protocol]" from
# leaking into the UI.
_PLACEHOLDER_STRINGS = frozenset(
    {
        "[System: Empty message content sanitised to satisfy protocol]",
        "[no content]",
    }
)


def _is_placeholder_text(text: str) -> bool:
    """Return True if text is a known placeholder that should be stripped."""
    if not isinstance(text, str) or not text:
        return False
    return text.strip() in _PLACEHOLDER_STRINGS


# Web search tool definition injected into OpenAI requests
_WEB_SEARCH_OPENAI_TOOL = {
    "type": Constants.TOOL_FUNCTION,
    Constants.TOOL_FUNCTION: {
        "name": "web_search",
        "description": (
            "Search the web for current information."
            " Returns results with titles, URLs, and content snippets."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query",
                }
            },
            "required": ["query"],
        },
    },
}


def _is_web_search_tool(tool: Any) -> bool:
    """Check if a tool dict is an Anthropic server-side web_search tool."""
    if isinstance(tool, dict):
        tool_type = tool.get("type", "")
        return isinstance(tool_type, str) and tool_type.startswith("web_search")
    return False


def _block_type(block: Any) -> str:
    """Get the type of a content block, whether Pydantic model or dict."""
    if hasattr(block, "type"):
        return str(block.type)
    if isinstance(block, dict):
        return str(block.get("type", ""))
    return ""


def _get_block_text(block: Any) -> str:
    """Extract the text value from a content block (dict or Pydantic model)."""
    if isinstance(block, dict):
        return block.get("text", "")
    return getattr(block, "text", "")


def _is_tool_result_message(msg: ClaudeMessage) -> bool:
    """Return True if the message contains at least one tool_result block."""
    return (
        msg.role == Constants.ROLE_USER
        and isinstance(msg.content, list)
        and any(
            _block_type(block) == Constants.CONTENT_TOOL_RESULT for block in msg.content
        )
    )


# ---------------------------------------------------------------------------
# convert_claude_to_openai — helpers
# ---------------------------------------------------------------------------


def _convert_system_message(
    system: Any,
) -> Optional[Dict[str, Any]]:
    """Convert Claude system prompt to an OpenAI system message dict.

    Returns None if the system prompt is empty or whitespace-only.
    """
    if not system:
        return None

    if isinstance(system, str):
        system_text = system
    elif isinstance(system, list):
        text_parts = []
        for block in system:
            if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                text_parts.append(block.text)
            elif (
                isinstance(block, dict) and block.get("type") == Constants.CONTENT_TEXT
            ):
                text_parts.append(block.get("text", ""))
        system_text = "\n\n".join(text_parts)
    else:
        return None

    stripped = system_text.strip()
    if not stripped:
        return None
    return {"role": Constants.ROLE_SYSTEM, "content": stripped}


def _convert_messages_list(
    messages: List[ClaudeMessage],
) -> List[Dict[str, Any]]:
    """Convert Claude messages to OpenAI messages, pairing tool results."""
    openai_messages: List[Dict[str, Any]] = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_messages.append(convert_claude_user_message(msg))
        elif msg.role == Constants.ROLE_ASSISTANT:
            result = convert_claude_assistant_message(msg)
            openai_messages.append(result["message"])
            if result["extra_tool_messages"]:
                openai_messages.extend(result["extra_tool_messages"])

            # Check if next message contains tool results
            if i + 1 < len(messages) and _is_tool_result_message(messages[i + 1]):
                i += 1  # Skip to tool result message
                openai_messages.extend(convert_claude_tool_results(messages[i]))

        i += 1
    return openai_messages


def _convert_tools(
    tools: Optional[List[Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Convert Claude tools to OpenAI function tools.

    Returns (openai_tools, web_search_config).
    """
    if not tools:
        return [], None

    openai_tools: List[Dict[str, Any]] = []
    web_search_config: Optional[Dict[str, Any]] = None

    for tool in tools:
        if _is_web_search_tool(tool):
            web_search_config = tool if isinstance(tool, dict) else None
            openai_tools.append(_WEB_SEARCH_OPENAI_TOOL)
        elif isinstance(tool, ClaudeTool):
            if tool.name and tool.name.strip():
                openai_tools.append(
                    {
                        "type": Constants.TOOL_FUNCTION,
                        Constants.TOOL_FUNCTION: {
                            "name": tool.name,
                            "description": tool.description or "",
                            "parameters": tool.input_schema,
                        },
                    }
                )
        elif isinstance(tool, dict) and tool.get("name"):
            logger.debug("Skipping unknown tool type: %s", tool.get("type", "none"))

    return openai_tools, web_search_config


def _convert_tool_choice(
    tool_choice: Optional[Dict[str, Any]],
) -> Optional[Any]:
    """Map Claude tool_choice to OpenAI tool_choice."""
    if not tool_choice:
        return None

    choice_type = tool_choice.get("type")
    if choice_type in ("auto", "any"):
        return "auto"
    if choice_type == "tool" and "name" in tool_choice:
        return {
            "type": Constants.TOOL_FUNCTION,
            Constants.TOOL_FUNCTION: {"name": tool_choice["name"]},
        }
    return "auto"


# ---------------------------------------------------------------------------
# convert_claude_to_openai — main entry point
# ---------------------------------------------------------------------------


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager: Any
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Convert Claude API request format to OpenAI format.

    Returns (openai_request, web_search_config) where web_search_config
    is the original Anthropic web_search tool dict if one was present,
    or None otherwise.
    """
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Build messages list
    openai_messages: List[Dict[str, Any]] = []
    system_msg = _convert_system_message(claude_request.system)
    if system_msg:
        openai_messages.append(system_msg)
    openai_messages.extend(_convert_messages_list(claude_request.messages))

    # Build base request
    openai_request: Dict[str, Any] = {
        "model": openai_model,
        "messages": openai_messages,
        "max_tokens": min(
            max(claude_request.max_tokens, config.min_tokens_limit),
            config.max_tokens_limit,
        ),
        "temperature": claude_request.temperature,
        "stream": claude_request.stream,
    }
    logger.debug(
        "Converted Claude request to OpenAI format: %s",
        json.dumps(openai_request, indent=2, ensure_ascii=False),
    )

    # Optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Tools
    openai_tools, web_search_config = _convert_tools(claude_request.tools)
    if openai_tools:
        openai_request["tools"] = openai_tools

    # Tool choice
    mapped_choice = _convert_tool_choice(claude_request.tool_choice)
    if mapped_choice is not None:
        openai_request["tool_choice"] = mapped_choice

    return openai_request, web_search_config


# ---------------------------------------------------------------------------
# convert_claude_user_message — helpers
# ---------------------------------------------------------------------------


def _convert_image_block(block: Any) -> Optional[Dict[str, Any]]:
    """Convert a Claude image content block to OpenAI image_url format.

    Returns None if the source is not a valid base64 image.
    """
    if isinstance(block, dict):
        source = block.get("source", {})
    else:
        source = getattr(block, "source", {})

    if not (
        isinstance(source, dict)
        and source.get("type") == "base64"
        and "media_type" in source
        and "data" in source
    ):
        return None

    return {
        "type": "image_url",
        "image_url": {"url": f"data:{source['media_type']};base64,{source['data']}"},
    }


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}

    if isinstance(msg.content, str):
        if _is_placeholder_text(msg.content):
            return {"role": Constants.ROLE_USER, "content": ""}
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content: List[Dict[str, Any]] = []
    for block in msg.content:
        btype = _block_type(block)
        if btype == Constants.CONTENT_TEXT:
            text = _get_block_text(block)
            if _is_placeholder_text(text):
                continue
            openai_content.append({"type": "text", "text": text})
        elif btype == Constants.CONTENT_IMAGE:
            image_block = _convert_image_block(block)
            if image_block:
                openai_content.append(image_block)

    if not openai_content:
        return {"role": Constants.ROLE_USER, "content": ""}

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    return {"role": Constants.ROLE_USER, "content": openai_content}


# ---------------------------------------------------------------------------
# convert_claude_assistant_message — helpers
# ---------------------------------------------------------------------------


def _empty_assistant_result() -> Dict[str, Any]:
    """Return a result dict with content=None and no extra messages."""
    return {
        "message": {"role": Constants.ROLE_ASSISTANT, "content": None},
        "extra_tool_messages": [],
    }


def _convert_tool_use_block(block: Any) -> Dict[str, Any]:
    """Convert a tool_use or server_tool_use block to OpenAI tool_call format."""
    if isinstance(block, dict):
        block_id = block.get("id", "")
        block_name = block.get("name", "")
        block_input = block.get("input", {})
    else:
        block_id = getattr(block, "id", "")
        block_name = getattr(block, "name", "")
        block_input = getattr(block, "input", {})
    return {
        "id": block_id,
        "type": Constants.TOOL_FUNCTION,
        Constants.TOOL_FUNCTION: {
            "name": block_name,
            "arguments": json.dumps(block_input, ensure_ascii=False),
        },
    }


def _convert_web_search_result_block(block: Any) -> Dict[str, Any]:
    """Convert a web_search_tool_result block to an OpenAI tool message."""
    tool_use_id = block.get("tool_use_id", "") if isinstance(block, dict) else ""
    content = block.get("content", []) if isinstance(block, dict) else []

    if isinstance(content, list):
        result_texts = []
        for r in content:
            if isinstance(r, dict) and r.get("type") == "web_search_result":
                title = r.get("title", "")
                url = r.get("url", "")
                snippet = r.get("encrypted_content", "")
                result_texts.append(f"[{title}]({url})\n{snippet}")
        summary = "\n\n".join(result_texts) if result_texts else "No results"
    elif (
        isinstance(content, dict)
        and content.get("type") == "web_search_tool_result_error"
    ):
        summary = f"Search error: {content.get('error_code', 'unknown')}"
    else:
        summary = str(content)

    return {
        "role": Constants.ROLE_TOOL,
        "tool_call_id": tool_use_id,
        "content": summary,
    }


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format.

    Returns a dict with keys:
      - "message": the OpenAI assistant message dict
      - "extra_tool_messages": list of OpenAI tool-role messages for
        web_search_tool_result blocks (which Claude embeds in the
        assistant message but OpenAI needs as separate tool messages)
    """
    if msg.content is None:
        return _empty_assistant_result()

    if isinstance(msg.content, str):
        if _is_placeholder_text(msg.content):
            return _empty_assistant_result()
        return {
            "message": {"role": Constants.ROLE_ASSISTANT, "content": msg.content},
            "extra_tool_messages": [],
        }

    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    extra_tool_messages: List[Dict[str, Any]] = []

    for block in msg.content:
        btype = _block_type(block)

        if btype == Constants.CONTENT_TEXT:
            text = _get_block_text(block)
            if text and text.strip() and not _is_placeholder_text(text):
                text_parts.append(text)

        elif btype in (Constants.CONTENT_TOOL_USE, Constants.CONTENT_SERVER_TOOL_USE):
            tool_calls.append(_convert_tool_use_block(block))

        elif btype == Constants.CONTENT_WEB_SEARCH_RESULT:
            extra_tool_messages.append(_convert_web_search_result_block(block))

    openai_message: Dict[str, Any] = {"role": Constants.ROLE_ASSISTANT}
    openai_message["content"] = "".join(text_parts) if text_parts else None
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return {
        "message": openai_message,
        "extra_tool_messages": extra_tool_messages,
    }


# ---------------------------------------------------------------------------
# Tool results + content parsing
# ---------------------------------------------------------------------------


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages: List[Dict[str, Any]] = []

    if isinstance(msg.content, list):
        for block in msg.content:
            btype = _block_type(block)
            if btype == Constants.CONTENT_TOOL_RESULT:
                if isinstance(block, dict):
                    content_val = block.get("content")
                    tool_use_id = block.get("tool_use_id", "")
                else:
                    content_val = getattr(block, "content", None)
                    tool_use_id = getattr(block, "tool_use_id", "")
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": tool_use_id,
                        "content": parse_tool_result_content(content_val),
                    }
                )

    return tool_messages


def _safe_json_dumps(obj: Any) -> str:
    """JSON-serialize *obj*, falling back to str() on failure."""
    try:
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(obj)


def _parse_list_content(content: list) -> str:
    """Parse a list of content items into a joined string."""
    result_parts: List[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
            result_parts.append(item.get("text", ""))
        elif isinstance(item, str):
            result_parts.append(item)
        elif isinstance(item, dict):
            result_parts.append(
                item.get("text", "") if "text" in item else _safe_json_dumps(item)
            )
    return "\n".join(result_parts).strip()


def _parse_dict_content(content: dict) -> str:
    """Parse a dict content item into a string."""
    if content.get("type") == Constants.CONTENT_TEXT:
        return str(content.get("text", ""))
    return _safe_json_dumps(content)


def parse_tool_result_content(content: Any) -> str:
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return _parse_list_content(content)

    if isinstance(content, dict):
        return _parse_dict_content(content)

    try:
        return str(content)
    except Exception:  # pylint: disable=broad-exception-caught
        return "Unparsable content"
