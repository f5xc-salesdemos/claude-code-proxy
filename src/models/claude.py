"""Pydantic models for Claude Messages API request and response schemas."""

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class ClaudeContentBlockText(BaseModel):
    """A plain-text content block."""

    type: Literal["text"]
    text: str


class ClaudeContentBlockImage(BaseModel):
    """An image content block with base64 or URL source."""

    type: Literal["image"]
    source: Dict[str, Any]


class ClaudeContentBlockToolUse(BaseModel):
    """A tool-use content block representing a function call."""

    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ClaudeContentBlockToolResult(BaseModel):
    """A tool-result content block containing the output of a tool call."""

    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any]]


class ClaudeSystemContent(BaseModel):
    """A system-prompt content block."""

    type: Literal["text"]
    text: str


class ClaudeMessage(BaseModel):
    """A single message in a Claude conversation."""

    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ClaudeContentBlockText,
                ClaudeContentBlockImage,
                ClaudeContentBlockToolUse,
                ClaudeContentBlockToolResult,
                Dict[str, Any],
            ]
        ],
        None,
    ] = None


class ClaudeTool(BaseModel):
    """Definition of a tool available to the model."""

    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ClaudeThinkingConfig(BaseModel):
    """Configuration for the model's extended-thinking feature."""

    enabled: bool = True


class ClaudeMessagesRequest(BaseModel):
    """Top-level request body for the Claude Messages API."""

    model: str
    max_tokens: int
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Union[ClaudeTool, Dict[str, Any]]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ClaudeThinkingConfig] = None


class ClaudeTokenCountRequest(BaseModel):
    """Request body for the token-counting endpoint."""

    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[ClaudeSystemContent]]] = None
    tools: Optional[List[Union[ClaudeTool, Dict[str, Any]]]] = None
    thinking: Optional[ClaudeThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None
