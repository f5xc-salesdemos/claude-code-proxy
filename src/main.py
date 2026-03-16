"""FastAPI application and server lifecycle for the Claude-to-OpenAI proxy."""

import os
import signal
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI

from src.api.endpoints import router as api_router
from src.core.client import OpenAIClient
from src.core.config import config
from src.core.model_manager import ModelManager
from src.middleware import CorrelationIdMiddleware
from src.services.search import get_provider

# ---------------------------------------------------------------------------
# Lifespan — create/destroy singletons
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Create shared singletons at startup, tear them down at shutdown."""
    custom_headers = config.get_custom_headers()

    app.state.config = config
    app.state.model_manager = ModelManager(config)
    app.state.openai_client = OpenAIClient(
        config.openai_api_key,
        config.openai_base_url,
        config.request_timeout,
        api_version=config.azure_api_version,
        custom_headers=custom_headers,
    )
    # Initialise the search provider plugin (may be None)
    search_provider = None
    if config.search_provider:
        search_provider = get_provider(config.search_provider)
    app.state.search_provider = search_provider

    # Wire the search provider into the endpoints module global
    # (endpoints.py uses a module-level global for backward compat)
    from src.api import endpoints as _ep  # noqa: PLC0415

    _ep.search_provider = search_provider

    app.state.httpx_client = httpx.AsyncClient(timeout=config.request_timeout)
    app.state.custom_headers = custom_headers

    yield

    # Cleanup
    await app.state.httpx_client.aclose()
    if app.state.search_provider is not None:
        await app.state.search_provider.close()


app = FastAPI(
    title="Claude-to-OpenAI API Proxy",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CorrelationIdMiddleware)
app.include_router(api_router)

# ---------------------------------------------------------------------------
# Graceful reload via SIGHUP
# ---------------------------------------------------------------------------
# Instead of killing the proxy with `pkill` (which drops the listening socket
# and causes ECONNREFUSED for all clients), send SIGHUP to trigger a graceful
# restart.  The server finishes in-flight requests, shuts down, and a new
# server instance is created on the same port — all inside the same process.
#
# Two ways to trigger a reload:
#   1.  kill -HUP <pid>
#   2.  POST /admin/reload   (sends SIGHUP to itself)
# ---------------------------------------------------------------------------

_server: Optional[uvicorn.Server] = None  # pylint: disable=invalid-name
_reload_requested: bool = False  # pylint: disable=invalid-name


def _sighup_handler(signum: int, frame: Any) -> None:  # pylint: disable=unused-argument
    """Handle SIGHUP by requesting a graceful restart."""
    global _reload_requested  # pylint: disable=global-statement
    _reload_requested = True
    if _server is not None:
        _server.should_exit = True


def _print_help_and_exit() -> None:
    """Print CLI help text and exit."""
    print("Claude-to-OpenAI API Proxy v1.0.0")
    print("")
    print("Usage: python src/main.py")
    print("")
    print("Required environment variables:")
    print("  OPENAI_API_KEY - Your OpenAI API key")
    print("")
    print("Optional environment variables:")
    print("  ANTHROPIC_API_KEY - Expected Anthropic API key for client validation")
    print("                      If set, clients must provide this exact API key")
    print(
        "  OPENAI_BASE_URL - OpenAI API base URL (default: https://api.openai.com/v1)"
    )
    print("  BIG_MODEL - Model for opus requests (default: gpt-4o)")
    print("  MIDDLE_MODEL - Model for sonnet requests (default: gpt-4o)")
    print("  SMALL_MODEL - Model for haiku requests (default: gpt-4o-mini)")
    print("  HOST - Server host (default: 0.0.0.0)")
    print("  PORT - Server port (default: 8082)")
    print("  LOG_LEVEL - Logging level (default: WARNING)")
    print("  MAX_TOKENS_LIMIT - Token limit (default: 4096)")
    print("  MIN_TOKENS_LIMIT - Minimum token limit (default: 100)")
    print("  REQUEST_TIMEOUT - Request timeout in seconds (default: 90)")
    print("")
    print("Model mapping:")
    print(f"  Claude haiku models -> {config.small_model}")
    print(f"  Claude sonnet/opus models -> {config.big_model}")
    sys.exit(0)


def main() -> None:
    """Entry point — configure and run the proxy server."""
    global _server, _reload_requested  # pylint: disable=global-statement

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        _print_help_and_exit()

    # Configuration summary
    print("Claude-to-OpenAI API Proxy v1.0.0")
    print("  Configuration loaded successfully")
    print(f"   OpenAI Base URL: {config.openai_base_url}")
    print(f"   Big Model (opus): {config.big_model}")
    print(f"   Middle Model (sonnet): {config.middle_model}")
    print(f"   Small Model (haiku): {config.small_model}")
    print(f"   Max Tokens Limit: {config.max_tokens_limit}")
    print(f"   Request Timeout: {config.request_timeout}s")
    print(f"   Server: {config.host}:{config.port}")
    print(
        f"   Client API Key Validation: "
        f"{'Enabled' if config.anthropic_api_key else 'Disabled'}"
    )
    print(f"   Graceful reload: kill -HUP {os.getpid()}  or  POST /admin/reload")
    print("")

    # Log level is already validated by BaseSettings
    log_level = config.log_level.lower()

    # Register SIGHUP handler for graceful reload
    signal.signal(signal.SIGHUP, _sighup_handler)

    # -----------------------------------------------------------------------
    # Server loop — restarts on SIGHUP, exits cleanly on SIGTERM / SIGINT
    # -----------------------------------------------------------------------
    while True:
        _reload_requested = False

        server_config = uvicorn.Config(
            "src.main:app",
            host=config.host,
            port=config.port,
            log_level=log_level,
            reload=False,
        )
        _server = uvicorn.Server(server_config)
        _server.run()

        if not _reload_requested:
            # Normal shutdown (SIGTERM / SIGINT / ctrl-c) — exit the loop
            break

        print("SIGHUP received — reloading proxy...")


if __name__ == "__main__":
    main()
