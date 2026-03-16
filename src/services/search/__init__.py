"""Plugin-based web search provider system.

Providers register themselves at import time via
``register_provider(name, factory)``.  The application selects a
provider at startup using ``get_provider(name)``.
"""

from src.services.search.base import SearchProvider
from src.services.search.registry import (
    available_providers,
    get_provider,
    register_provider,
)

# Import provider modules so they auto-register their factories.
# Each module calls ``register_provider(...)`` at module level.
import src.services.search.tavily  # noqa: F401

__all__ = [
    "SearchProvider",
    "register_provider",
    "get_provider",
    "available_providers",
]
