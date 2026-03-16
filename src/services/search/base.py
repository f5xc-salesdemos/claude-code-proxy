"""Abstract base class for web search providers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class SearchProvider(ABC):
    """Interface that all search provider plugins must implement.

    Return contract for ``search()``:
        On success: ``{"results": [<web_search_result>, ...]}``
        On error:   ``{"error": {"type": "web_search_tool_result_error", "error_code": "..."}}``
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier for this provider (e.g. ``"tavily"``)."""
        ...

    @abstractmethod
    async def is_available(self) -> bool:
        """Return whether the provider is currently usable.

        Implementations should cache the result for a reasonable TTL
        to avoid excessive health-check traffic.
        """
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 5,
        *,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute a search and return Anthropic-formatted result content.

        Returns the ``content`` value for a ``web_search_tool_result``
        block — either a list of ``web_search_result`` dicts or an
        error dict.

        ``allowed_domains`` and ``blocked_domains`` correspond to the
        Anthropic ``web_search`` tool configuration fields of the same
        name and should be forwarded to the underlying search API when
        supported.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release resources (HTTP clients, connections, etc.)."""
        ...
