import json
from typing import Dict, Any, List, Optional, Tuple
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest, ClaudeMessage
from src.core.config import config
import logging

logger = logging.getLogger(__name__)

# Web search tool definition injected into OpenAI requests
_WEB_SEARCH_OPENAI_TOOL = {
    "type": Constants.TOOL_FUNCTION,
    Constants.TOOL_FUNCTION: {
        "name": "web_search",
        "description": "Search the web for current information. Returns results with titles, URLs, and content snippets.",
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
        return block.type
    if isinstance(block, dict):
        return block.get("type", "")
    return ""


def convert_claude_to_openai(
    claude_request: ClaudeMessagesRequest, model_manager
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Convert Claude API request format to OpenAI format.

    Returns (openai_request, web_search_config) where web_search_config
    is the original Anthropic web_search tool dict if one was present,
    or None otherwise.
    """

    # Map model
    openai_model = model_manager.map_claude_model_to_openai(claude_request.model)

    # Convert messages
    openai_messages = []

    # Add system message if present
    if claude_request.system:
        system_text = ""
        if isinstance(claude_request.system, str):
            system_text = claude_request.system
        elif isinstance(claude_request.system, list):
            text_parts = []
            for block in claude_request.system:
                if hasattr(block, "type") and block.type == Constants.CONTENT_TEXT:
                    text_parts.append(block.text)
                elif (
                    isinstance(block, dict)
                    and block.get("type") == Constants.CONTENT_TEXT
                ):
                    text_parts.append(block.get("text", ""))
            system_text = "\n\n".join(text_parts)

        if system_text.strip():
            openai_messages.append(
                {"role": Constants.ROLE_SYSTEM, "content": system_text.strip()}
            )

    # Process Claude messages
    i = 0
    while i < len(claude_request.messages):
        msg = claude_request.messages[i]

        if msg.role == Constants.ROLE_USER:
            openai_message = convert_claude_user_message(msg)
            openai_messages.append(openai_message)
        elif msg.role == Constants.ROLE_ASSISTANT:
            result = convert_claude_assistant_message(msg)
            openai_messages.append(result["message"])
            if result["extra_tool_messages"]:
                openai_messages.extend(result["extra_tool_messages"])

            # Check if next message contains tool results
            if i + 1 < len(claude_request.messages):
                next_msg = claude_request.messages[i + 1]
                if (
                    next_msg.role == Constants.ROLE_USER
                    and isinstance(next_msg.content, list)
                    and any(
                        _block_type(block) == Constants.CONTENT_TOOL_RESULT
                        for block in next_msg.content
                    )
                ):
                    # Process tool results
                    i += 1  # Skip to tool result message
                    tool_results = convert_claude_tool_results(next_msg)
                    openai_messages.extend(tool_results)

        i += 1

    # Build OpenAI request
    openai_request = {
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
        f"Converted Claude request to OpenAI format: {json.dumps(openai_request, indent=2, ensure_ascii=False)}"
    )
    # Add optional parameters
    if claude_request.stop_sequences:
        openai_request["stop"] = claude_request.stop_sequences
    if claude_request.top_p is not None:
        openai_request["top_p"] = claude_request.top_p

    # Convert tools — separate server tools from regular tools
    web_search_config: Optional[Dict[str, Any]] = None
    if claude_request.tools:
        openai_tools = []
        for tool in claude_request.tools:
            if _is_web_search_tool(tool):
                web_search_config = tool if isinstance(tool, dict) else None
                # Inject synthetic OpenAI function tool for web_search
                openai_tools.append(_WEB_SEARCH_OPENAI_TOOL)
            elif hasattr(tool, "name") and hasattr(tool, "input_schema"):
                # Regular ClaudeTool Pydantic model
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
                # Unknown dict-based tool — skip silently
                logger.debug(f"Skipping unknown tool type: {tool.get('type', 'none')}")
        if openai_tools:
            openai_request["tools"] = openai_tools

    # Convert tool choice
    if claude_request.tool_choice:
        choice_type = claude_request.tool_choice.get("type")
        if choice_type == "auto":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "any":
            openai_request["tool_choice"] = "auto"
        elif choice_type == "tool" and "name" in claude_request.tool_choice:
            openai_request["tool_choice"] = {
                "type": Constants.TOOL_FUNCTION,
                Constants.TOOL_FUNCTION: {"name": claude_request.tool_choice["name"]},
            }
        else:
            openai_request["tool_choice"] = "auto"

    return openai_request, web_search_config


def convert_claude_user_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude user message to OpenAI format."""
    if msg.content is None:
        return {"role": Constants.ROLE_USER, "content": ""}

    if isinstance(msg.content, str):
        return {"role": Constants.ROLE_USER, "content": msg.content}

    # Handle multimodal content
    openai_content = []
    for block in msg.content:
        btype = _block_type(block)
        if btype == Constants.CONTENT_TEXT:
            text = block.text if hasattr(block, "text") else block.get("text", "")
            openai_content.append({"type": "text", "text": text})
        elif btype == Constants.CONTENT_IMAGE:
            # Convert Claude image format to OpenAI format
            source = block.source if hasattr(block, "source") else block.get("source", {})
            if (
                isinstance(source, dict)
                and source.get("type") == "base64"
                and "media_type" in source
                and "data" in source
            ):
                openai_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{source['media_type']};base64,{source['data']}"
                        },
                    }
                )

    if len(openai_content) == 1 and openai_content[0]["type"] == "text":
        return {"role": Constants.ROLE_USER, "content": openai_content[0]["text"]}
    else:
        return {"role": Constants.ROLE_USER, "content": openai_content}


def convert_claude_assistant_message(msg: ClaudeMessage) -> Dict[str, Any]:
    """Convert Claude assistant message to OpenAI format.

    Returns a dict with keys:
      - "message": the OpenAI assistant message dict
      - "extra_tool_messages": list of OpenAI tool-role messages for
        web_search_tool_result blocks (which Claude embeds in the
        assistant message but OpenAI needs as separate tool messages)
    """
    text_parts = []
    tool_calls = []
    extra_tool_messages: List[Dict[str, Any]] = []

    if msg.content is None:
        return {
            "message": {"role": Constants.ROLE_ASSISTANT, "content": None},
            "extra_tool_messages": [],
        }

    if isinstance(msg.content, str):
        return {
            "message": {"role": Constants.ROLE_ASSISTANT, "content": msg.content},
            "extra_tool_messages": [],
        }

    for block in msg.content:
        btype = _block_type(block)

        if btype == Constants.CONTENT_TEXT:
            text = block.text if hasattr(block, "text") else block.get("text", "")
            text_parts.append(text)

        elif btype == Constants.CONTENT_TOOL_USE:
            block_id = block.id if hasattr(block, "id") else block.get("id", "")
            block_name = block.name if hasattr(block, "name") else block.get("name", "")
            block_input = block.input if hasattr(block, "input") else block.get("input", {})
            tool_calls.append(
                {
                    "id": block_id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block_name,
                        "arguments": json.dumps(block_input, ensure_ascii=False),
                    },
                }
            )

        elif btype == Constants.CONTENT_SERVER_TOOL_USE:
            # server_tool_use blocks (e.g. web_search) — treat like tool_use
            block_id = block.get("id", "") if isinstance(block, dict) else ""
            block_name = block.get("name", "") if isinstance(block, dict) else ""
            block_input = block.get("input", {}) if isinstance(block, dict) else {}
            tool_calls.append(
                {
                    "id": block_id,
                    "type": Constants.TOOL_FUNCTION,
                    Constants.TOOL_FUNCTION: {
                        "name": block_name,
                        "arguments": json.dumps(block_input, ensure_ascii=False),
                    },
                }
            )

        elif btype == Constants.CONTENT_WEB_SEARCH_RESULT:
            # web_search_tool_result — convert to an OpenAI tool message
            tool_use_id = block.get("tool_use_id", "") if isinstance(block, dict) else ""
            content = block.get("content", []) if isinstance(block, dict) else []
            # Summarise results as text for the upstream model
            if isinstance(content, list):
                result_texts = []
                for r in content:
                    if isinstance(r, dict) and r.get("type") == "web_search_result":
                        title = r.get("title", "")
                        url = r.get("url", "")
                        snippet = r.get("encrypted_content", "")
                        result_texts.append(f"[{title}]({url})\n{snippet}")
                summary = "\n\n".join(result_texts) if result_texts else "No results"
            elif isinstance(content, dict) and content.get("type") == "web_search_tool_result_error":
                summary = f"Search error: {content.get('error_code', 'unknown')}"
            else:
                summary = str(content)
            extra_tool_messages.append(
                {
                    "role": Constants.ROLE_TOOL,
                    "tool_call_id": tool_use_id,
                    "content": summary,
                }
            )

    openai_message: Dict[str, Any] = {"role": Constants.ROLE_ASSISTANT}

    # Set content
    if text_parts:
        openai_message["content"] = "".join(text_parts)
    else:
        openai_message["content"] = None

    # Set tool calls
    if tool_calls:
        openai_message["tool_calls"] = tool_calls

    return {
        "message": openai_message,
        "extra_tool_messages": extra_tool_messages,
    }


def convert_claude_tool_results(msg: ClaudeMessage) -> List[Dict[str, Any]]:
    """Convert Claude tool results to OpenAI format."""
    tool_messages = []

    if isinstance(msg.content, list):
        for block in msg.content:
            btype = _block_type(block)
            if btype == Constants.CONTENT_TOOL_RESULT:
                content_val = block.content if hasattr(block, "content") else block.get("content")
                tool_use_id = block.tool_use_id if hasattr(block, "tool_use_id") else block.get("tool_use_id", "")
                content = parse_tool_result_content(content_val)
                tool_messages.append(
                    {
                        "role": Constants.ROLE_TOOL,
                        "tool_call_id": tool_use_id,
                        "content": content,
                    }
                )

    return tool_messages


def parse_tool_result_content(content):
    """Parse and normalize tool result content into a string format."""
    if content is None:
        return "No content provided"

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        result_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == Constants.CONTENT_TEXT:
                result_parts.append(item.get("text", ""))
            elif isinstance(item, str):
                result_parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    result_parts.append(item.get("text", ""))
                else:
                    try:
                        result_parts.append(json.dumps(item, ensure_ascii=False))
                    except:
                        result_parts.append(str(item))
        return "\n".join(result_parts).strip()

    if isinstance(content, dict):
        if content.get("type") == Constants.CONTENT_TEXT:
            return content.get("text", "")
        try:
            return json.dumps(content, ensure_ascii=False)
        except:
            return str(content)

    try:
        return str(content)
    except:
        return "Unparseable content"
