"""
Responses API → Chat Completions translation layer.

Converts OpenAI Responses API requests (used by Codex CLI) into
Chat Completions API requests that the upstream server supports,
then converts the responses back.
"""

import json
import logging
import uuid

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


def convert_responses_to_chat_completions(body: dict) -> dict:
    """Convert a Responses API request body to Chat Completions format.

    Handles:
      - ``input`` (string or list of items) → ``messages``
      - ``instructions`` → system message
      - ``tools`` → Chat Completions function tools
      - ``max_output_tokens`` → ``max_tokens``
      - Pass-through: model, temperature, top_p, stream, tool_choice
      - Strip: reasoning, previous_response_id, truncation, text, store
    """
    messages: list[dict] = []

    # System message from instructions
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # Convert input → messages
    raw_input = body.get("input", "")
    if isinstance(raw_input, str):
        messages.append({"role": "user", "content": raw_input})
    elif isinstance(raw_input, list):
        _convert_input_items(raw_input, messages)

    # Ensure at least one message
    if not messages:
        messages.append({"role": "user", "content": ""})

    # Build request
    cc_request: dict = {
        "model": body.get("model", ""),
        "messages": messages,
    }

    # Token limits
    max_output = body.get("max_output_tokens")
    if max_output is not None:
        cc_request["max_tokens"] = max_output

    # Pass-through scalars
    for key in ("temperature", "top_p"):
        if key in body and body[key] is not None:
            cc_request[key] = body[key]

    cc_request["stream"] = body.get("stream", False)

    # Tools
    tools = body.get("tools")
    if tools:
        cc_tools = []
        for tool in tools:
            tool_type = tool.get("type", "")
            if tool_type == "function":
                fn = tool.get("function") or tool
                cc_tools.append({
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    },
                })
        if cc_tools:
            cc_request["tools"] = cc_tools

    # Tool choice
    tool_choice = body.get("tool_choice")
    if tool_choice is not None:
        cc_request["tool_choice"] = tool_choice

    return cc_request


def _convert_input_items(items: list, messages: list[dict]) -> None:
    """Walk the Responses API ``input`` array and append to *messages*."""
    # We may need to coalesce consecutive user content parts
    pending_assistant: dict | None = None

    def _flush_assistant():
        nonlocal pending_assistant
        if pending_assistant is not None:
            messages.append(pending_assistant)
            pending_assistant = None

    for item in items:
        if isinstance(item, str):
            _flush_assistant()
            messages.append({"role": "user", "content": item})
            continue

        item_type = item.get("type", "")

        if item_type == "message":
            _flush_assistant()
            role = item.get("role", "user")
            content = item.get("content", "")
            # content can be a string or list of content parts
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, str):
                        text_parts.append(part)
                    elif isinstance(part, dict):
                        ptype = part.get("type", "")
                        if ptype in ("input_text", "output_text", "text"):
                            text_parts.append(part.get("text", ""))
                        elif ptype == "refusal":
                            text_parts.append(part.get("refusal", ""))
                content = "\n".join(text_parts) if text_parts else ""
            messages.append({"role": role, "content": content})

        elif item_type == "function_call":
            # Assistant made a tool call — emit as assistant message with tool_calls
            call_id = item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:24]}"))
            name = item.get("name", "")
            arguments = item.get("arguments", "{}")
            if pending_assistant is None:
                pending_assistant = {"role": "assistant", "content": None, "tool_calls": []}
            elif "tool_calls" not in pending_assistant:
                pending_assistant["tool_calls"] = []
            pending_assistant["tool_calls"].append({
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            })

        elif item_type == "function_call_output":
            _flush_assistant()
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output if isinstance(output, str) else json.dumps(output),
            })

        elif item_type in ("local_shell_call",):
            # Treat like function_call
            call_id = item.get("call_id", item.get("id", f"call_{uuid.uuid4().hex[:24]}"))
            args = json.dumps({"command": item.get("command", [])})
            if pending_assistant is None:
                pending_assistant = {"role": "assistant", "content": None, "tool_calls": []}
            elif "tool_calls" not in pending_assistant:
                pending_assistant["tool_calls"] = []
            pending_assistant["tool_calls"].append({
                "id": call_id,
                "type": "function",
                "function": {"name": "local_shell", "arguments": args},
            })

        elif item_type == "local_shell_call_output":
            _flush_assistant()
            call_id = item.get("call_id", "")
            output = item.get("output", "")
            messages.append({
                "role": "tool",
                "tool_call_id": call_id,
                "content": output if isinstance(output, str) else json.dumps(output),
            })

        elif item_type == "reasoning":
            # Skip reasoning items
            continue

        else:
            # Unknown item — try to treat as user message
            text = item.get("text", item.get("content", ""))
            if text:
                _flush_assistant()
                messages.append({"role": "user", "content": text if isinstance(text, str) else json.dumps(text)})

    _flush_assistant()


def build_response_object(openai_response: dict, original_body: dict) -> dict:
    """Convert a Chat Completions response to Responses API format."""
    choices = openai_response.get("choices", [])
    if not choices:
        raise HTTPException(status_code=500, detail="No choices in upstream response")

    choice = choices[0]
    message = choice.get("message", {})
    finish_reason = choice.get("finish_reason", "stop")

    output: list[dict] = []
    response_id = f"resp_{uuid.uuid4().hex[:24]}"

    # Text content → output_text item
    text_content = message.get("content")
    if text_content is not None:
        output.append({
            "type": "message",
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text_content, "annotations": []}],
        })

    # Tool calls → function_call items
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function", {})
        output.append({
            "type": "function_call",
            "id": f"fc_{uuid.uuid4().hex[:24]}",
            "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:24]}"),
            "name": fn.get("name", ""),
            "arguments": fn.get("arguments", "{}"),
            "status": "completed",
        })

    # Map finish_reason to status
    status_map = {
        "stop": "completed",
        "length": "incomplete",
        "tool_calls": "completed",
        "function_call": "completed",
    }
    status = status_map.get(finish_reason, "completed")

    # Usage
    upstream_usage = openai_response.get("usage", {})
    usage = {
        "input_tokens": upstream_usage.get("prompt_tokens", 0),
        "output_tokens": upstream_usage.get("completion_tokens", 0),
        "total_tokens": upstream_usage.get("total_tokens", 0),
    }

    return {
        "id": response_id,
        "object": "response",
        "created_at": openai_response.get("created", 0),
        "status": status,
        "model": original_body.get("model", ""),
        "output": output,
        "usage": usage,
        "metadata": {},
    }


async def stream_responses_from_chat_completions(
    openai_stream,
    body: dict,
    http_request: Request,
    openai_client,
    request_id: str,
):
    """Consume Chat Completions SSE and re-emit as Responses API SSE events."""
    response_id = f"resp_{uuid.uuid4().hex[:24]}"
    msg_id = f"msg_{uuid.uuid4().hex[:24]}"
    model = body.get("model", "")

    # Tracking state
    text_started = False
    text_content_index = 0  # index within the output array
    text_buffer = ""
    tool_calls: dict[int, dict] = {}  # keyed by tc_index from upstream
    tool_output_index = 1  # output array index (0 = message)

    # -- response.created
    yield _sse("response.created", {
        "type": "response.created",
        "response": {
            "id": response_id,
            "object": "response",
            "status": "in_progress",
            "model": model,
            "output": [],
            "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        },
    })

    # -- response.in_progress
    yield _sse("response.in_progress", {
        "type": "response.in_progress",
        "response": {
            "id": response_id,
            "object": "response",
            "status": "in_progress",
            "model": model,
            "output": [],
        },
    })

    final_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    try:
        async for line in openai_stream:
            # Client disconnect check
            if await http_request.is_disconnected():
                logger.info(f"Client disconnected during responses stream {request_id}")
                openai_client.cancel_request(request_id)
                break

            if not line.strip():
                continue
            if line.startswith("data: "):
                chunk_data = line[6:]
                if chunk_data.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(chunk_data)
                except json.JSONDecodeError:
                    continue

                # Extract usage if present
                usage = chunk.get("usage")
                if usage:
                    final_usage = {
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    }

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})

                # --- Text delta ---
                if delta.get("content") is not None:
                    text_chunk = delta["content"]
                    if not text_started:
                        text_started = True
                        # output_item.added for the message
                        yield _sse("response.output_item.added", {
                            "type": "response.output_item.added",
                            "output_index": text_content_index,
                            "item": {
                                "type": "message",
                                "id": msg_id,
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        })
                        # content_part.added
                        yield _sse("response.content_part.added", {
                            "type": "response.content_part.added",
                            "item_id": msg_id,
                            "output_index": text_content_index,
                            "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        })

                    text_buffer += text_chunk
                    yield _sse("response.output_text.delta", {
                        "type": "response.output_text.delta",
                        "item_id": msg_id,
                        "output_index": text_content_index,
                        "content_index": 0,
                        "delta": text_chunk,
                    })

                # --- Tool call deltas ---
                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        tc_index = tc_delta.get("index", 0)
                        if tc_index not in tool_calls:
                            tool_calls[tc_index] = {
                                "id": None,
                                "call_id": None,
                                "name": "",
                                "arguments": "",
                                "output_index": None,
                                "started": False,
                            }
                        tc = tool_calls[tc_index]

                        if tc_delta.get("id"):
                            tc["call_id"] = tc_delta["id"]

                        fn = tc_delta.get("function", {})
                        if fn.get("name"):
                            tc["name"] = fn["name"]

                        # Start the function_call output item once we have id + name
                        if tc["call_id"] and tc["name"] and not tc["started"]:
                            tc["started"] = True
                            tc["id"] = f"fc_{uuid.uuid4().hex[:24]}"
                            # Close text item first if it was open
                            if text_started:
                                yield _sse("response.content_part.done", {
                                    "type": "response.content_part.done",
                                    "item_id": msg_id,
                                    "output_index": text_content_index,
                                    "content_index": 0,
                                    "part": {"type": "output_text", "text": text_buffer, "annotations": []},
                                })
                                yield _sse("response.output_item.done", {
                                    "type": "response.output_item.done",
                                    "output_index": text_content_index,
                                    "item": {
                                        "type": "message",
                                        "id": msg_id,
                                        "role": "assistant",
                                        "status": "completed",
                                        "content": [{"type": "output_text", "text": text_buffer, "annotations": []}],
                                    },
                                })
                                text_started = False  # prevent double-close

                            tc["output_index"] = tool_output_index
                            tool_output_index += 1
                            yield _sse("response.output_item.added", {
                                "type": "response.output_item.added",
                                "output_index": tc["output_index"],
                                "item": {
                                    "type": "function_call",
                                    "id": tc["id"],
                                    "call_id": tc["call_id"],
                                    "name": tc["name"],
                                    "arguments": "",
                                    "status": "in_progress",
                                },
                            })

                        # Arguments delta
                        if fn.get("arguments") and tc["started"]:
                            tc["arguments"] += fn["arguments"]
                            yield _sse("response.function_call_arguments.delta", {
                                "type": "response.function_call_arguments.delta",
                                "item_id": tc["id"],
                                "output_index": tc["output_index"],
                                "delta": fn["arguments"],
                            })

    except HTTPException as e:
        if e.status_code == 499:
            logger.info(f"Responses stream {request_id} cancelled")
        else:
            logger.error(f"HTTP error in responses stream: {e.detail}")
        return
    except Exception as e:
        logger.error(f"Error in responses stream: {e}")
        yield _sse("error", {"type": "error", "message": str(e)})
        return

    # --- Finalize ---
    # Close text output item if still open
    if text_started:
        yield _sse("response.content_part.done", {
            "type": "response.content_part.done",
            "item_id": msg_id,
            "output_index": text_content_index,
            "content_index": 0,
            "part": {"type": "output_text", "text": text_buffer, "annotations": []},
        })
        yield _sse("response.output_item.done", {
            "type": "response.output_item.done",
            "output_index": text_content_index,
            "item": {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text_buffer, "annotations": []}],
            },
        })

    # Close any open tool call items
    for tc in tool_calls.values():
        if tc["started"]:
            yield _sse("response.function_call_arguments.done", {
                "type": "response.function_call_arguments.done",
                "item_id": tc["id"],
                "output_index": tc["output_index"],
                "arguments": tc["arguments"],
            })
            yield _sse("response.output_item.done", {
                "type": "response.output_item.done",
                "output_index": tc["output_index"],
                "item": {
                    "type": "function_call",
                    "id": tc["id"],
                    "call_id": tc["call_id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"],
                    "status": "completed",
                },
            })

    # Build final output array for the completed event
    final_output: list[dict] = []
    if text_buffer:
        final_output.append({
            "type": "message",
            "id": msg_id,
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text_buffer, "annotations": []}],
        })
    for tc in tool_calls.values():
        if tc["started"]:
            final_output.append({
                "type": "function_call",
                "id": tc["id"],
                "call_id": tc["call_id"],
                "name": tc["name"],
                "arguments": tc["arguments"],
                "status": "completed",
            })

    # response.completed
    yield _sse("response.completed", {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "object": "response",
            "status": "completed",
            "model": model,
            "output": final_output,
            "usage": final_usage,
        },
    })


def _sse(event: str, data: dict) -> str:
    """Format a single SSE frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
