"""Unit tests for convert_claude_to_openai() in request_converter."""

from unittest.mock import MagicMock

from src.conversion.request_converter import convert_claude_to_openai
from src.models.claude import ClaudeMessagesRequest


def _mock_model_manager(model_name: str = "gpt-4o") -> MagicMock:
    """Build a mock model_manager that returns a fixed model name."""
    mm = MagicMock()
    mm.map_claude_model_to_openai.return_value = model_name
    return mm


def _make_request(**overrides) -> ClaudeMessagesRequest:
    defaults = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(overrides)
    return ClaudeMessagesRequest(**defaults)


class TestConvertClaudeToOpenAI:
    """Tests for the top-level convert_claude_to_openai function."""

    def test_basic_message_with_string_system(self):
        """String system prompt is added as a system message."""
        req = _make_request(system="You are helpful.")
        result, ws = convert_claude_to_openai(req, _mock_model_manager())
        assert ws is None
        msgs = result["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."
        assert msgs[1]["role"] == "user"

    def test_system_prompt_as_list_of_text_blocks(self):
        """System prompt given as list of text blocks is joined with \\n\\n."""
        req = _make_request(
            system=[
                {"type": "text", "text": "First part"},
                {"type": "text", "text": "Second part"},
            ]
        )
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        assert result["messages"][0]["content"] == "First part\n\nSecond part"

    def test_empty_system_prompt_omitted(self):
        """Empty/whitespace-only system prompt does not produce a system message."""
        req = _make_request(system="   ")
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        assert all(m["role"] != "system" for m in result["messages"])

    def test_no_system_prompt(self):
        """When system is None, no system message is added."""
        req = _make_request(system=None)
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        assert all(m["role"] != "system" for m in result["messages"])

    def test_user_and_assistant_ordering(self):
        """User and assistant messages are converted in order."""
        req = _make_request(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ]
        )
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        roles = [m["role"] for m in result["messages"]]
        assert roles == ["user", "assistant", "user"]

    def test_assistant_tool_result_pairing(self):
        """Assistant with tool_use followed by user with tool_result pairs correctly."""
        req = _make_request(
            messages=[
                {"role": "user", "content": "Do something"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "my_tool",
                            "input": {"x": 1},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "result text",
                        }
                    ],
                },
            ]
        )
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        roles = [m["role"] for m in result["messages"]]
        # user, assistant, tool (consumed from next user message)
        assert roles == ["user", "assistant", "tool"]
        assert result["messages"][2]["tool_call_id"] == "call_1"
        assert result["messages"][2]["content"] == "result text"

    def test_tool_conversion(self):
        """ClaudeTool objects are converted to OpenAI function tools."""
        req = _make_request(
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ]
        )
        result, ws = convert_claude_to_openai(req, _mock_model_manager())
        assert ws is None
        assert len(result["tools"]) == 1
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "get_weather"

    def test_web_search_tool_detection(self):
        """web_search tool type returns web_search_config."""
        req = _make_request(
            tools=[
                {"type": "web_search_20250305", "name": "web_search"},
            ]
        )
        result, ws = convert_claude_to_openai(req, _mock_model_manager())
        assert ws is not None
        assert ws["type"] == "web_search_20250305"
        # Should inject synthetic function tool
        assert any(
            t["function"]["name"] == "web_search" for t in result.get("tools", [])
        )

    def test_tool_choice_auto(self):
        """tool_choice type=auto maps to 'auto'."""
        req = _make_request(
            tools=[
                {
                    "name": "t",
                    "description": "d",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "auto"},
        )
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        assert result["tool_choice"] == "auto"

    def test_tool_choice_any(self):
        """tool_choice type=any maps to 'auto'."""
        req = _make_request(
            tools=[
                {
                    "name": "t",
                    "description": "d",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "any"},
        )
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        assert result["tool_choice"] == "auto"

    def test_tool_choice_specific_tool(self):
        """tool_choice type=tool with name maps to function selection."""
        req = _make_request(
            tools=[
                {
                    "name": "my_func",
                    "description": "d",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
            tool_choice={"type": "tool", "name": "my_func"},
        )
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "my_func"

    def test_max_tokens_clamped(self):
        """max_tokens is clamped between min and max limits."""
        req = _make_request(max_tokens=50)
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        # Should be at least min_tokens_limit
        from src.core.config import config

        assert result["max_tokens"] >= config.min_tokens_limit
        assert result["max_tokens"] <= config.max_tokens_limit

    def test_stop_sequences_passthrough(self):
        """stop_sequences are passed through as 'stop'."""
        req = _make_request(stop_sequences=["END", "STOP"])
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        assert result["stop"] == ["END", "STOP"]

    def test_top_p_passthrough(self):
        """top_p is passed through when provided."""
        req = _make_request(top_p=0.9)
        result, _ = convert_claude_to_openai(req, _mock_model_manager())
        assert result["top_p"] == 0.9
