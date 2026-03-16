"""FastAPI dependency functions for dependency injection.

Singletons are stored on ``app.state`` by the lifespan handler in
``main.py``.  Dependency functions retrieve them from the request's
app instance so that endpoint handlers never reference module-level
globals.
"""

from typing import Any, Optional

import httpx
from fastapi import Request

from src.core.client import OpenAIClient
from src.core.config import Config
from src.services.search.base import SearchProvider


def get_config(request: Request) -> Config:
    """Return the Config singleton from app.state."""
    return request.app.state.config  # type: ignore[no-any-return]


def get_openai_client(request: Request) -> OpenAIClient:
    """Return the OpenAI client singleton from app.state."""
    return request.app.state.openai_client  # type: ignore[no-any-return]


def get_model_manager(request: Request) -> Any:
    """Return the ModelManager singleton from app.state."""
    return request.app.state.model_manager


def get_search_provider(request: Request) -> Optional[SearchProvider]:
    """Return the search provider singleton from app.state (may be None)."""
    return getattr(request.app.state, "search_provider", None)


def get_httpx_client(request: Request) -> httpx.AsyncClient:
    """Return the shared httpx client from app.state."""
    return request.app.state.httpx_client  # type: ignore[no-any-return]
