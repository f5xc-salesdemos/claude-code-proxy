from fastapi import APIRouter, HTTPException, Request, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
import json
import os
import signal
import uuid
from typing import Optional

import httpx

from src.core.config import config
from src.core.logging import logger
from src.core.client import OpenAIClient
from src.models.claude import ClaudeMessagesRequest, ClaudeTokenCountRequest
from src.conversion.request_converter import convert_claude_to_openai
from src.conversion.response_converter import (
    convert_openai_to_claude_response,
    convert_openai_streaming_to_claude_with_cancellation,
)
from src.conversion.responses_converter import (
    convert_responses_to_chat_completions,
    build_response_object,
    stream_responses_from_chat_completions,
)
from src.core.model_manager import model_manager
from src.services.searxng import SearXNGClient

router = APIRouter()

# Get custom headers from config
custom_headers = config.get_custom_headers()

openai_client = OpenAIClient(
    config.openai_api_key,
    config.openai_base_url,
    config.request_timeout,
    api_version=config.azure_api_version,
    custom_headers=custom_headers,
)

# SearXNG client for WebSearch interception
searxng_client = SearXNGClient(config.searxng_url)

# Shared httpx client for pass-through requests
_httpx_client: Optional[httpx.AsyncClient] = None


def _get_httpx_client() -> httpx.AsyncClient:
    global _httpx_client
    if _httpx_client is None:
        _httpx_client = httpx.AsyncClient(timeout=config.request_timeout)
    return _httpx_client


def _extract_bearer_token(
    x_api_key: Optional[str] = Header(None),
    authorization: Optional[str] = Header(None),
) -> Optional[str]:
    """Extract the API key from request headers."""
    if x_api_key:
        return x_api_key
    if authorization and authorization.startswith("Bearer "):
        return authorization.replace("Bearer ", "")
    return None


async def validate_api_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """Validate the client's API key against ANTHROPIC_API_KEY (Claude Code flow)."""
    client_api_key = _extract_bearer_token(x_api_key, authorization)

    # Skip validation if ANTHROPIC_API_KEY is not set in the environment
    if not config.anthropic_api_key:
        return

    # Validate the client API key
    if not client_api_key or not config.validate_client_api_key(client_api_key):
        logger.warning(f"Invalid API key provided by client")
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Please provide a valid Anthropic API key."
        )


async def validate_openai_api_key(x_api_key: Optional[str] = Header(None), authorization: Optional[str] = Header(None)):
    """Validate the client's API key against either ANTHROPIC_API_KEY or OPENAI_API_KEY.

    Used by Responses API, Chat Completions pass-through, and Models endpoints
    where clients send the OpenAI key rather than the Anthropic key.

    Allows unauthenticated requests — some clients (e.g. Codex CLI) send no
    Authorization header at all.  Since the proxy only listens on localhost
    inside the container, this is safe.
    """
    client_api_key = _extract_bearer_token(x_api_key, authorization)

    # If no validation keys are configured, skip
    if not config.anthropic_api_key and not config.openai_api_key:
        return

    # Allow unauthenticated requests (Codex sends no auth header)
    if not client_api_key:
        return

    # If a key IS provided, it must match one of the configured keys
    if config.anthropic_api_key and client_api_key == config.anthropic_api_key:
        return
    if config.openai_api_key and client_api_key == config.openai_api_key:
        return

    logger.warning("Invalid API key provided by client (openai validation)")
    raise HTTPException(status_code=401, detail="Invalid API key.")

@router.post("/v1/messages")
async def create_message(request: ClaudeMessagesRequest, http_request: Request, _: None = Depends(validate_api_key)):
    try:
        logger.debug(
            f"Processing Claude request: model={request.model}, stream={request.stream}"
        )

        # Generate unique request ID for cancellation tracking
        request_id = str(uuid.uuid4())

        # Convert Claude request to OpenAI format
        openai_request, web_search_config = convert_claude_to_openai(request, model_manager)

        # If web_search was requested, check SearXNG availability
        if web_search_config and not await searxng_client.is_available():
            logger.info("SearXNG not available, stripping web_search tool")
            web_search_config = None
            # Remove the synthetic web_search function tool from the request
            if "tools" in openai_request:
                openai_request["tools"] = [
                    t for t in openai_request["tools"]
                    if not (t.get("type") == "function" and
                            t.get("function", {}).get("name") == "web_search")
                ]
                if not openai_request["tools"]:
                    del openai_request["tools"]

        # Check if client disconnected before processing
        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if request.stream:
            # Streaming response - wrap in error handling
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    openai_request, request_id
                )
                return StreamingResponse(
                    convert_openai_streaming_to_claude_with_cancellation(
                        openai_stream,
                        request,
                        logger,
                        http_request,
                        openai_client,
                        request_id,
                        web_search_config=web_search_config,
                        searxng_client=searxng_client if web_search_config else None,
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                # Convert to proper error response for streaming
                logger.error(f"Streaming error: {e.detail}")
                import traceback

                logger.error(traceback.format_exc())
                error_message = openai_client.classify_openai_error(e.detail)
                error_response = {
                    "type": "error",
                    "error": {"type": "api_error", "message": error_message},
                }
                return JSONResponse(status_code=e.status_code, content=error_response)
        else:
            # Non-streaming response
            openai_response = await openai_client.create_chat_completion(
                openai_request, request_id
            )

            # Handle web_search interception for non-streaming
            if web_search_config:
                tool_calls = openai_response.get("choices", [{}])[0].get("message", {}).get("tool_calls", []) or []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    if func.get("name") == "web_search":
                        try:
                            args = json.loads(func.get("arguments", "{}"))
                            query = args.get("query", "")
                        except json.JSONDecodeError:
                            query = ""
                        if query:
                            search_result = await searxng_client.search(query)
                            # Inject results into the response
                            # Build a modified response with server_tool_use + result
                            return _build_non_streaming_web_search_response(
                                openai_response, request, query, search_result
                            )

            claude_response = convert_openai_to_claude_response(
                openai_response, request
            )
            return claude_response
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        logger.error(f"Unexpected error processing request: {e}")
        logger.error(traceback.format_exc())
        error_message = openai_client.classify_openai_error(str(e))
        raise HTTPException(status_code=500, detail=error_message)


def _build_non_streaming_web_search_response(
    openai_response: dict,
    original_request: ClaudeMessagesRequest,
    query: str,
    search_result: dict,
) -> dict:
    """Build a Claude response with web_search server_tool_use + result for non-streaming."""
    from src.conversion.response_converter import _generate_server_tool_id

    message = openai_response.get("choices", [{}])[0].get("message", {})
    content_blocks = []

    # Add text content if present
    text_content = message.get("content")
    if text_content:
        content_blocks.append({"type": "text", "text": text_content})

    # Add server_tool_use block
    server_tool_id = _generate_server_tool_id()
    content_blocks.append({
        "type": "server_tool_use",
        "id": server_tool_id,
        "name": "web_search",
        "input": {"query": query},
    })

    # Add web_search_tool_result block
    if "error" in search_result:
        result_content = search_result["error"]
    else:
        result_content = search_result.get("results", [])

    content_blocks.append({
        "type": "web_search_tool_result",
        "tool_use_id": server_tool_id,
        "content": result_content,
    })

    usage_data = {
        "input_tokens": openai_response.get("usage", {}).get("prompt_tokens", 0),
        "output_tokens": openai_response.get("usage", {}).get("completion_tokens", 0),
        "server_tool_use": {"web_search_requests": 1},
    }

    return {
        "id": openai_response.get("id", f"msg_{uuid.uuid4()}"),
        "type": "message",
        "role": "assistant",
        "model": original_request.model,
        "content": content_blocks,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": usage_data,
    }


# ============================================================
# Responses API endpoints (Codex CLI)
# ============================================================

@router.post("/responses")
@router.post("/v1/responses")
async def create_response(http_request: Request, _: None = Depends(validate_openai_api_key)):
    """Translate an OpenAI Responses API request to Chat Completions."""
    try:
        body = await http_request.json()
        logger.debug(f"Responses API request: model={body.get('model')}, stream={body.get('stream')}")

        request_id = str(uuid.uuid4())
        cc_request = convert_responses_to_chat_completions(body)

        if await http_request.is_disconnected():
            raise HTTPException(status_code=499, detail="Client disconnected")

        if cc_request.get("stream"):
            try:
                openai_stream = openai_client.create_chat_completion_stream(
                    cc_request, request_id
                )
                return StreamingResponse(
                    stream_responses_from_chat_completions(
                        openai_stream, body, http_request, openai_client, request_id
                    ),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "*",
                    },
                )
            except HTTPException as e:
                logger.error(f"Responses streaming error: {e.detail}")
                return JSONResponse(
                    status_code=e.status_code,
                    content={"error": {"message": str(e.detail), "type": "api_error"}},
                )
        else:
            openai_response = await openai_client.create_chat_completion(
                cc_request, request_id
            )
            return build_response_object(openai_response, body)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Responses API error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# Chat Completions pass-through (OpenCode and other tools)
# ============================================================

@router.post("/chat/completions")
@router.post("/v1/chat/completions")
async def chat_completions_passthrough(http_request: Request, _: None = Depends(validate_openai_api_key)):
    """Forward Chat Completions requests to the upstream server unchanged."""
    try:
        body = await http_request.body()
        body_json = json.loads(body)
        stream = body_json.get("stream", False)

        upstream_url = config.openai_base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.openai_api_key}",
        }
        for k, v in custom_headers.items():
            headers[k] = v

        client = _get_httpx_client()

        if stream:
            upstream_req = client.build_request("POST", upstream_url, headers=headers, content=body)
            upstream_resp = await client.send(upstream_req, stream=True)
            if upstream_resp.status_code != 200:
                resp_body = await upstream_resp.aread()
                return JSONResponse(status_code=upstream_resp.status_code, content=json.loads(resp_body))

            async def _stream():
                try:
                    async for line in upstream_resp.aiter_lines():
                        yield f"{line}\n\n"
                finally:
                    await upstream_resp.aclose()

            return StreamingResponse(
                _stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*",
                },
            )
        else:
            resp = await client.post(upstream_url, headers=headers, content=body)
            return JSONResponse(status_code=resp.status_code, content=resp.json())

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Chat completions pass-through error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=502, detail=str(e))


# ============================================================
# Models pass-through
# ============================================================

@router.get("/models")
@router.get("/v1/models")
async def list_models(_: None = Depends(validate_openai_api_key)):
    """Forward /models request to the upstream server."""
    try:
        upstream_url = config.openai_base_url.rstrip("/") + "/models"
        headers = {
            "Authorization": f"Bearer {config.openai_api_key}",
        }
        for k, v in custom_headers.items():
            headers[k] = v

        client = _get_httpx_client()
        resp = await client.get(upstream_url, headers=headers)
        return JSONResponse(status_code=resp.status_code, content=resp.json())

    except Exception as e:
        logger.error(f"Models pass-through error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/v1/messages/count_tokens")
async def count_tokens(request: ClaudeTokenCountRequest, _: None = Depends(validate_api_key)):
    try:
        # For token counting, we'll use a simple estimation
        # In a real implementation, you might want to use tiktoken or similar

        total_chars = 0

        # Count system message characters
        if request.system:
            if isinstance(request.system, str):
                total_chars += len(request.system)
            elif isinstance(request.system, list):
                for block in request.system:
                    if hasattr(block, "text"):
                        total_chars += len(block.text)

        # Count message characters
        for msg in request.messages:
            if msg.content is None:
                continue
            elif isinstance(msg.content, str):
                total_chars += len(msg.content)
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text") and block.text is not None:
                        total_chars += len(block.text)

        # Rough estimation: 4 characters per token
        estimated_tokens = max(1, total_chars // 4)

        return {"input_tokens": estimated_tokens}

    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "openai_api_configured": bool(config.openai_api_key),
        "api_key_valid": config.validate_api_key(),
        "client_api_key_validation": bool(config.anthropic_api_key),
        "searxng_url": config.searxng_url,
    }


@router.get("/test-connection")
async def test_connection():
    """Test API connectivity to OpenAI"""
    try:
        # Simple test request to verify API connectivity
        test_response = await openai_client.create_chat_completion(
            {
                "model": config.small_model,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 5,
            }
        )

        return {
            "status": "success",
            "message": "Successfully connected to OpenAI API",
            "model_used": config.small_model,
            "timestamp": datetime.now().isoformat(),
            "response_id": test_response.get("id", "unknown"),
        }

    except Exception as e:
        logger.error(f"API connectivity test failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error_type": "API Error",
                "message": str(e),
                "timestamp": datetime.now().isoformat(),
                "suggestions": [
                    "Check your OPENAI_API_KEY is valid",
                    "Verify your API key has the necessary permissions",
                    "Check if you have reached rate limits",
                ],
            },
        )


# ============================================================
# Admin endpoints
# ============================================================

@router.post("/admin/reload")
async def admin_reload(_: None = Depends(validate_api_key)):
    """Trigger a graceful proxy reload via SIGHUP.

    The server finishes in-flight requests, shuts down, and restarts
    on the same port — all inside the same process.  No ECONNREFUSED.
    """
    os.kill(os.getpid(), signal.SIGHUP)
    return JSONResponse(
        status_code=202,
        content={
            "status": "reload_initiated",
            "message": "Server will restart momentarily. Wait 2-3 seconds then check /health.",
            "pid": os.getpid(),
        },
    )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Claude-to-OpenAI API Proxy v1.0.0",
        "status": "running",
        "config": {
            "openai_base_url": config.openai_base_url,
            "max_tokens_limit": config.max_tokens_limit,
            "api_key_configured": bool(config.openai_api_key),
            "client_api_key_validation": bool(config.anthropic_api_key),
            "big_model": config.big_model,
            "small_model": config.small_model,
            "searxng_url": config.searxng_url,
        },
        "endpoints": {
            "messages": "/v1/messages",
            "responses": "/v1/responses",
            "chat_completions": "/v1/chat/completions",
            "models": "/v1/models",
            "count_tokens": "/v1/messages/count_tokens",
            "health": "/health",
            "test_connection": "/test-connection",
            "admin_reload": "/admin/reload",
        },
    }
