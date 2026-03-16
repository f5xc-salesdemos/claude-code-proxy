"""Convert OpenAI Chat Completions responses to Claude Messages API format."""

import json
import logging
import traceback
import uuid
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import HTTPException, Request
from src.core.constants import Constants
from src.models.claude import ClaudeMessagesRequest
from src.services.search.base import SearchProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_server_tool_id() -> str:
    """Generate a unique server tool ID."""
    return f"srvtoolu_{uuid.uuid4().hex[:24]}"


def _sse_event(event_type: str, data: Dict[str, Any]) -> str:
    """Format a single SSE frame."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _map_finish_reason(finish_reason: Optional[str]) -> str:
    """Map an OpenAI finish_reason to a Claude stop_reason."""
    return {
        "stop": Constants.STOP_END_TURN,
        "length": Constants.STOP_MAX_TOKENS,
        "tool_calls": Constants.STOP_TOOL_USE,
        "function_call": Constants.STOP_TOOL_USE,
    }.get(finish_reason or "stop", Constants.STOP_END_TURN)


# ---------------------------------------------------------------------------
# Non-streaming conversion
# ---------------------------------------------------------------------------


def convert_openai_to_claude_response(
    openai_response: Dict[str, Any],
    original_request: ClaudeMessagesRequest,
    web_search_config: Optional[Dict[str, Any]] = None,
    search_provider: Optional[SearchProvider] = None,
) -> Dict[str, Any]:
    """Convert OpenAI response to Claude format.

    NOTE: For web_search interception in non-streaming mode the caller
    must have already executed the search and injected the results (since
    this function is synchronous).  The ``web_search_config`` and
    ``search_provider`` params are accepted for signature consistency but
    web_search handling is done in the endpoint layer for non-streaming.
    """

    # Extract response data
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in OpenAI response")

    choice = choices[0]
    message = choice.get("message", {})

    # Build Claude content blocks
    content_blocks = []

    # Add text content — skip empty/whitespace-only strings to avoid
    # creating text blocks that LiteLLM would later sanitise with a
    # visible placeholder.
    text_content = message.get("content")
    if isinstance(text_content, str) and text_content.strip():
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": text_content})

    # Add tool calls
    tool_calls = message.get("tool_calls", []) or []
    for tool_call in tool_calls:
        if tool_call.get("type") == Constants.TOOL_FUNCTION:
            function_data = tool_call.get(Constants.TOOL_FUNCTION, {})
            try:
                arguments = json.loads(function_data.get("arguments", "{}"))
            except json.JSONDecodeError:
                arguments = {"raw_arguments": function_data.get("arguments", "")}

            content_blocks.append(
                {
                    "type": Constants.CONTENT_TOOL_USE,
                    "id": tool_call.get("id", f"tool_{uuid.uuid4()}"),
                    "name": function_data.get("name", ""),
                    "input": arguments,
                }
            )

    # The Anthropic Messages API requires at least one content block.
    if not content_blocks:
        content_blocks.append({"type": Constants.CONTENT_TEXT, "text": "[no content]"})

    # Map finish reason
    stop_reason = _map_finish_reason(choice.get("finish_reason", "stop"))

    # Build Claude response
    claude_response = {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": Constants.ROLE_ASSISTANT,
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
            "output_tokens": openai_response.get("usage", {}).get(
                "completion_tokens", 0
            ),
        },
    }

    return claude_response


# ---------------------------------------------------------------------------
# Streaming conversion (unified)
# ---------------------------------------------------------------------------


async def convert_openai_streaming_to_claude(
    openai_stream: AsyncGenerator[str, None],
    original_request: ClaudeMessagesRequest,
    logger: logging.Logger,  # pylint: disable=redefined-outer-name
    http_request: Optional[Request] = None,
    openai_client: Optional[Any] = None,
    request_id: Optional[str] = None,
    *,
    web_search_config: Optional[Dict[str, Any]] = None,
    search_provider: Optional[SearchProvider] = None,
) -> AsyncGenerator[str, None]:
    """Convert OpenAI streaming response to Claude streaming format.

    When ``http_request`` is provided, client disconnection is checked on
    every chunk and the request is cancelled via ``openai_client``.  When
    it is ``None``, cancellation checking is skipped (simple mode).
    """
    cancellation_enabled = http_request is not None

    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    # Send initial SSE events
    msg_start = {
        "type": Constants.EVENT_MESSAGE_START,
        "message": {
            "id": message_id,
            "type": "message",
            "role": Constants.ROLE_ASSISTANT,
            "model": original_request.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }
    yield _sse_event(Constants.EVENT_MESSAGE_START, msg_start)
    yield _sse_event(Constants.EVENT_PING, {"type": Constants.EVENT_PING})

    # Process streaming chunks — defer text content_block_start until we
    # actually receive text.  Avoids triggering LiteLLM's empty-message
    # sanitisation.
    text_block_started = False
    text_block_index = 0
    tool_block_counter = 0
    current_tool_calls: Dict[int, Dict[str, Any]] = {}
    final_stop_reason = Constants.STOP_END_TURN
    usage_data: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}
    web_search_count = 0

    try:
        async for line in openai_stream:
            # Check if client disconnected (cancellation mode only)
            if cancellation_enabled and await http_request.is_disconnected():  # type: ignore[union-attr]
                logger.info("Client disconnected, cancelling request %s", request_id)
                if openai_client is not None:
                    openai_client.cancel_request(request_id)
                break

            if not line.strip():
                continue
            if not line.startswith("data: "):
                continue

            chunk_data = line[6:]
            if chunk_data.strip() == "[DONE]":
                break

            try:
                chunk = json.loads(chunk_data)
                # Track usage when present
                usage = chunk.get("usage")
                if usage:
                    cache_read = 0
                    details = usage.get("prompt_tokens_details") or {}
                    if details:
                        cache_read = details.get("cached_tokens", 0)
                    usage_data = {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "cache_read_input_tokens": cache_read,
                    }
                choices = chunk.get("choices", [])
                if not choices:
                    continue
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse chunk: %s, error: %s", chunk_data, e)
                continue

            choice = choices[0]
            delta = choice.get("delta", {})
            finish_reason = choice.get("finish_reason")

            # --- Text deltas ---
            # Skip empty/whitespace-only deltas to avoid opening a text
            # block that LiteLLM later sanitises.
            if delta and delta.get("content") and delta["content"].strip():
                if not text_block_started:
                    yield _sse_event(
                        Constants.EVENT_CONTENT_BLOCK_START,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_START,
                            "index": 0,
                            "content_block": {
                                "type": Constants.CONTENT_TEXT,
                                "text": "",
                            },
                        },
                    )
                    text_block_started = True
                yield _sse_event(
                    Constants.EVENT_CONTENT_BLOCK_DELTA,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                        "index": text_block_index,
                        "delta": {
                            "type": Constants.DELTA_TEXT,
                            "text": delta["content"],
                        },
                    },
                )

            # --- Tool call deltas ---
            if "tool_calls" in delta and delta["tool_calls"]:
                for tc_delta in delta["tool_calls"]:
                    for evt in _handle_tool_delta(
                        tc_delta,
                        current_tool_calls,
                        tool_block_counter,
                        text_block_started,
                        text_block_index,
                        web_search_config,
                    ):
                        yield evt
                    # Update counter (may have been incremented)
                    tool_block_counter = max(
                        tool_block_counter,
                        sum(1 for tc in current_tool_calls.values() if tc["started"]),
                    )

            # Handle finish reason
            if finish_reason:
                final_stop_reason = _map_finish_reason(finish_reason)
                if cancellation_enabled:
                    # Don't break — let the stream naturally end
                    pass
                else:
                    break

    except HTTPException as e:
        if e.status_code == 499:
            logger.info("Request %s was cancelled", request_id)
            yield _sse_event(
                "error",
                {
                    "type": "error",
                    "error": {
                        "type": "cancelled",
                        "message": "Request was cancelled by client",
                    },
                },
            )
            return
        # Emit SSE error event instead of re-raising (which causes
        # RuntimeError: "response already started").
        logger.error("HTTP %s during streaming: %s", e.status_code, e.detail)
        yield _sse_event(
            "error",
            {
                "type": "error",
                "error": {"type": "api_error", "message": str(e.detail)},
            },
        )
        return
    except Exception as e:
        logger.error("Streaming error: %s", e)
        logger.error(traceback.format_exc())
        yield _sse_event(
            "error",
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"Streaming error: {str(e)}",
                },
            },
        )
        return

    # --- Finalize ---

    # Close text block if opened
    if text_block_started:
        yield _sse_event(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {"type": Constants.EVENT_CONTENT_BLOCK_STOP, "index": text_block_index},
        )

    # Close tool call blocks and handle web_search interception
    for tool_data in current_tool_calls.values():
        if not tool_data.get("started") or tool_data.get("claude_index") is None:
            continue

        yield _sse_event(
            Constants.EVENT_CONTENT_BLOCK_STOP,
            {
                "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                "index": tool_data["claude_index"],
            },
        )

        # For web_search calls, execute search and emit result block
        if tool_data.get("is_web_search") and search_provider is not None:
            web_search_count += 1
            query = ""
            try:
                args = json.loads(tool_data.get("args_buffer", "{}"))
                query = args.get("query", "")
            except json.JSONDecodeError:
                query = tool_data.get("args_buffer", "").strip('"')

            logger.info("Executing web search for: %s", query)
            # Extract domain filters from web_search_config
            _allowed = (
                web_search_config.get("allowed_domains") if web_search_config else None
            )
            _blocked = (
                web_search_config.get("blocked_domains") if web_search_config else None
            )
            search_result = await search_provider.search(
                query,
                allowed_domains=_allowed or None,
                blocked_domains=_blocked or None,
            )

            if "error" in search_result:
                result_content = search_result["error"]
            else:
                result_content = search_result.get("results", [])

            tool_block_counter += 1
            result_index = (
                (text_block_index + tool_block_counter)
                if text_block_started
                else (tool_block_counter - 1)
            )
            server_tool_id = tool_data.get("server_tool_id", _generate_server_tool_id())

            yield _sse_event(
                Constants.EVENT_CONTENT_BLOCK_START,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_START,
                    "index": result_index,
                    "content_block": {
                        "type": Constants.CONTENT_WEB_SEARCH_RESULT,
                        "tool_use_id": server_tool_id,
                        "content": result_content,
                    },
                },
            )
            yield _sse_event(
                Constants.EVENT_CONTENT_BLOCK_STOP,
                {
                    "type": Constants.EVENT_CONTENT_BLOCK_STOP,
                    "index": result_index,
                },
            )
            final_stop_reason = Constants.STOP_END_TURN

    # Add server_tool_use usage tracking
    if web_search_count > 0:
        usage_data["server_tool_use"] = {"web_search_requests": web_search_count}

    yield _sse_event(
        Constants.EVENT_MESSAGE_DELTA,
        {
            "type": Constants.EVENT_MESSAGE_DELTA,
            "delta": {
                "stop_reason": final_stop_reason,
                "stop_sequence": None,
            },
            "usage": usage_data,
        },
    )
    yield _sse_event(
        Constants.EVENT_MESSAGE_STOP,
        {"type": Constants.EVENT_MESSAGE_STOP},
    )


def _handle_tool_delta(
    tc_delta: Dict[str, Any],
    current_tool_calls: Dict[int, Dict[str, Any]],
    tool_block_counter: int,
    text_block_started: bool,
    text_block_index: int,
    web_search_config: Optional[Dict[str, Any]],
) -> list:
    """Process a single tool call delta, returning SSE events to yield."""
    events = []
    tc_index = tc_delta.get("index", 0)

    # Initialize tool call tracking
    if tc_index not in current_tool_calls:
        current_tool_calls[tc_index] = {
            "id": None,
            "name": None,
            "args_buffer": "",
            "json_sent": False,
            "claude_index": None,
            "started": False,
            "is_web_search": False,
        }

    tool_call = current_tool_calls[tc_index]

    if tc_delta.get("id"):
        tool_call["id"] = tc_delta["id"]

    function_data = tc_delta.get(Constants.TOOL_FUNCTION, {})
    if function_data.get("name"):
        tool_call["name"] = function_data["name"]
        if function_data["name"] == "web_search" and web_search_config:
            tool_call["is_web_search"] = True

    # Start content block when we have complete initial data
    if tool_call["id"] and tool_call["name"] and not tool_call["started"]:
        started_count = sum(1 for tc in current_tool_calls.values() if tc["started"])
        claude_index = (
            (text_block_index + started_count + 1)
            if text_block_started
            else started_count
        )
        tool_call["claude_index"] = claude_index
        tool_call["started"] = True

        if tool_call["is_web_search"]:
            server_tool_id = _generate_server_tool_id()
            tool_call["server_tool_id"] = server_tool_id
            events.append(
                _sse_event(
                    Constants.EVENT_CONTENT_BLOCK_START,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_START,
                        "index": claude_index,
                        "content_block": {
                            "type": Constants.CONTENT_SERVER_TOOL_USE,
                            "id": server_tool_id,
                            "name": "web_search",
                            "input": {},
                        },
                    },
                )
            )
        else:
            events.append(
                _sse_event(
                    Constants.EVENT_CONTENT_BLOCK_START,
                    {
                        "type": Constants.EVENT_CONTENT_BLOCK_START,
                        "index": claude_index,
                        "content_block": {
                            "type": Constants.CONTENT_TOOL_USE,
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "input": {},
                        },
                    },
                )
            )

    # Handle function arguments
    if (
        "arguments" in function_data
        and tool_call["started"]
        and function_data["arguments"] is not None
    ):
        tool_call["args_buffer"] += function_data["arguments"]

        try:
            json.loads(tool_call["args_buffer"])
            if not tool_call["json_sent"]:
                events.append(
                    _sse_event(
                        Constants.EVENT_CONTENT_BLOCK_DELTA,
                        {
                            "type": Constants.EVENT_CONTENT_BLOCK_DELTA,
                            "index": tool_call["claude_index"],
                            "delta": {
                                "type": Constants.DELTA_INPUT_JSON,
                                "partial_json": tool_call["args_buffer"],
                            },
                        },
                    )
                )
                tool_call["json_sent"] = True
        except json.JSONDecodeError:
            pass

    return events


# Backward-compatible alias — callers using the old name keep working.
convert_openai_streaming_to_claude_with_cancellation = (
    convert_openai_streaming_to_claude
)
