"""Search provider plugin registry.

Providers register a factory function at import time:

    register_provider("tavily", lambda: TavilySearchProvider(...))

The application retrieves an instance via ``get_provider("tavily")``.
"""

from typing import Callable, Dict, List, Optional
from src.services.search.base import SearchProvider

# Internal registry: lowercase name → factory function
_registry: Dict[str, Callable[[], Optional[SearchProvider]]] = {}


def register_provider(
    name: str, factory: Callable[[], Optional[SearchProvider]]
) -> None:
    """Register a provider factory under *name* (case-insensitive)."""
    _registry[name.lower()] = factory


def get_provider(name: str) -> Optional[SearchProvider]:
    """Instantiate and return the provider registered as *name*.

    Returns ``None`` if no provider is registered under that name or
    if the factory itself returns ``None``.
    """
    factory = _registry.get(name.lower())
    if factory is None:
        return None
    return factory()


def available_providers() -> List[str]:
    """Return a sorted list of registered provider names."""
    return sorted(_registry.keys())
