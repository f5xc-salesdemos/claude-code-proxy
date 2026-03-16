"""Unit tests for the Responses API → Chat Completions translation layer."""

import pytest

from src.conversion.responses_converter import (
    build_response_object,
    convert_responses_to_chat_completions,
)

# ---------------------------------------------------------------------------
# convert_responses_to_chat_completions
# ---------------------------------------------------------------------------


class TestConvertResponsesToCC:
    """Test conversion of Responses API requests to Chat Completions."""

    def test_string_input(self):
        """String input becomes a single user message."""
        result = convert_responses_to_chat_completions(
            {"model": "gpt-4o", "input": "hello"}
        )
        assert result["messages"] == [{"role": "user", "content": "hello"}]
        assert result["model"] == "gpt-4o"

    def test_instructions_become_system_message(self):
        """Instructions field maps to a system message."""
        result = convert_responses_to_chat_completions(
            {
                "model": "gpt-4o",
                "input": "hi",
                "instructions": "Be helpful",
            }
        )
        assert result["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert result["messages"][1] == {"role": "user", "content": "hi"}

    def test_empty_input_gets_default(self):
        """Missing input creates a fallback user message."""
        result = convert_responses_to_chat_completions({"model": "m"})
        assert result["messages"] == [{"role": "user", "content": ""}]

    def test_max_output_tokens(self):
        """max_output_tokens maps to max_tokens."""
        result = convert_responses_to_chat_completions(
            {"model": "m", "input": "x", "max_output_tokens": 256}
        )
        assert result["max_tokens"] == 256

    def test_temperature_passthrough(self):
        """Temperature is passed through to CC request."""
        result = convert_responses_to_chat_completions(
            {"model": "m", "input": "x", "temperature": 0.5}
        )
        assert result["temperature"] == 0.5

    def test_stream_defaults_false(self):
        """Stream defaults to False."""
        result = convert_responses_to_chat_completions({"model": "m", "input": "x"})
        assert result["stream"] is False

    def test_stream_passthrough(self):
        """Stream value is passed through."""
        result = convert_responses_to_chat_completions(
            {"model": "m", "input": "x", "stream": True}
        )
        assert result["stream"] is True

    def test_function_tools_converted(self):
        """Function tools are converted correctly."""
        result = convert_responses_to_chat_completions(
            {
                "model": "m",
                "input": "x",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Search the web",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            }
        )
        assert len(result["tools"]) == 1
        assert result["tools"][0]["function"]["name"] == "search"

    def test_tool_choice_passthrough(self):
        """Tool choice is passed through unchanged."""
        result = convert_responses_to_chat_completions(
            {"model": "m", "input": "x", "tool_choice": "auto"}
        )
        assert result["tool_choice"] == "auto"

    def test_list_input_with_message_items(self):
        """List input with message items converts correctly."""
        result = convert_responses_to_chat_completions(
            {
                "model": "m",
                "input": [
                    {"type": "message", "role": "user", "content": "hello"},
                    {"type": "message", "role": "assistant", "content": "hi there"},
                ],
            }
        )
        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][1]["role"] == "assistant"

    def test_function_call_input(self):
        """Function call items become assistant messages with tool_calls."""
        result = convert_responses_to_chat_completions(
            {
                "model": "m",
                "input": [
                    {"type": "message", "role": "user", "content": "search for cats"},
                    {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "search",
                        "arguments": '{"q": "cats"}',
                    },
                    {
                        "type": "function_call_output",
                        "call_id": "call_1",
                        "output": "cats are great",
                    },
                ],
            }
        )
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["user", "assistant", "tool"]
        assert result["messages"][1]["tool_calls"][0]["function"]["name"] == "search"
        assert result["messages"][2]["content"] == "cats are great"

    def test_reasoning_items_skipped(self):
        """Reasoning items are silently skipped."""
        result = convert_responses_to_chat_completions(
            {
                "model": "m",
                "input": [
                    {"type": "reasoning", "content": "thinking..."},
                    {"type": "message", "role": "user", "content": "hello"},
                ],
            }
        )
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "hello"


# ---------------------------------------------------------------------------
# build_response_object
# ---------------------------------------------------------------------------


class TestBuildResponseObject:
    """Test conversion of CC responses to Responses API format."""

    def test_text_response(self):
        """Text content creates a message output item."""
        openai_resp = {
            "id": "chatcmpl-1",
            "created": 1234,
            "choices": [{"message": {"content": "Hello"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = build_response_object(openai_resp, {"model": "gpt-4o"})
        assert result["status"] == "completed"
        assert len(result["output"]) == 1
        assert result["output"][0]["type"] == "message"
        assert result["output"][0]["content"][0]["text"] == "Hello"

    def test_tool_call_response(self):
        """Tool calls create function_call output items."""
        openai_resp = {
            "id": "chatcmpl-2",
            "created": 1234,
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "search",
                                    "arguments": '{"q": "cats"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = build_response_object(openai_resp, {"model": "gpt-4o"})
        assert result["status"] == "completed"
        fc_items = [o for o in result["output"] if o["type"] == "function_call"]
        assert len(fc_items) == 1
        assert fc_items[0]["name"] == "search"

    def test_length_finish_reason_maps_to_incomplete(self):
        """finish_reason 'length' maps to status 'incomplete'."""
        openai_resp = {
            "id": "chatcmpl-3",
            "created": 1234,
            "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
        result = build_response_object(openai_resp, {"model": "gpt-4o"})
        assert result["status"] == "incomplete"

    def test_usage_mapping(self):
        """Usage fields are correctly mapped."""
        openai_resp = {
            "id": "chatcmpl-4",
            "created": 1234,
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        result = build_response_object(openai_resp, {"model": "m"})
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15

    def test_no_choices_raises(self):
        """Empty choices raises HTTPException."""
        with pytest.raises(Exception):
            build_response_object(
                {"id": "x", "created": 0, "choices": [], "usage": {}},
                {"model": "m"},
            )
