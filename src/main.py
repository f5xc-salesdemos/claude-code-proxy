import os
import signal
import sys

import uvicorn
from fastapi import FastAPI

from src.api.endpoints import router as api_router
from src.core.config import config

app = FastAPI(title="Claude-to-OpenAI API Proxy", version="1.0.0")
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

_server: uvicorn.Server | None = None
_reload_requested: bool = False


def _sighup_handler(signum, frame):
    """Handle SIGHUP by requesting a graceful restart."""
    global _reload_requested
    _reload_requested = True
    if _server is not None:
        _server.should_exit = True


def main():
    global _server, _reload_requested

    if len(sys.argv) > 1 and sys.argv[1] == "--help":
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
            f"  OPENAI_BASE_URL - OpenAI API base URL (default: https://api.openai.com/v1)"
        )
        print(f"  BIG_MODEL - Model for opus requests (default: gpt-4o)")
        print(f"  MIDDLE_MODEL - Model for sonnet requests (default: gpt-4o)")
        print(f"  SMALL_MODEL - Model for haiku requests (default: gpt-4o-mini)")
        print(f"  HOST - Server host (default: 0.0.0.0)")
        print(f"  PORT - Server port (default: 8082)")
        print(f"  LOG_LEVEL - Logging level (default: WARNING)")
        print(f"  MAX_TOKENS_LIMIT - Token limit (default: 4096)")
        print(f"  MIN_TOKENS_LIMIT - Minimum token limit (default: 100)")
        print(f"  REQUEST_TIMEOUT - Request timeout in seconds (default: 90)")
        print("")
        print("Model mapping:")
        print(f"  Claude haiku models -> {config.small_model}")
        print(f"  Claude sonnet/opus models -> {config.big_model}")
        sys.exit(0)

    # Configuration summary
    print("Claude-to-OpenAI API Proxy v1.0.0")
    print(f"  Configuration loaded successfully")
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

    # Parse log level - extract just the first word to handle comments
    log_level = config.log_level.split()[0].lower()

    # Validate and set default if invalid
    valid_levels = ["debug", "info", "warning", "error", "critical"]
    if log_level not in valid_levels:
        log_level = "info"

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
