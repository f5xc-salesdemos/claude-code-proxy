"""Tests for the SearchProvider abstract base class."""

# pylint: disable=abstract-class-instantiated,missing-class-docstring

from typing import Any, Dict, List, Optional

import pytest
from src.services.search.base import SearchProvider


class TestSearchProviderABC:
    """Verify the ABC enforces all required methods."""

    def test_cannot_instantiate_directly(self):
        """SearchProvider cannot be instantiated because it is abstract."""
        with pytest.raises(TypeError):
            SearchProvider()  # type: ignore[abstract]

    def test_subclass_missing_methods_raises(self):
        """A subclass that omits abstract methods cannot be instantiated."""

        class IncompleteProvider(SearchProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]

    def test_subclass_missing_provider_name_raises(self):
        """A subclass that omits provider_name cannot be instantiated."""

        class MissingName(SearchProvider):
            async def is_available(self) -> bool:
                return True

            async def search(
                self,
                query,
                max_results=5,
                *,
                allowed_domains=None,
                blocked_domains=None,
            ):
                return {"results": []}

            async def close(self):
                pass

        with pytest.raises(TypeError):
            MissingName()  # type: ignore[abstract]

    def test_subclass_missing_is_available_raises(self):
        """A subclass that omits is_available cannot be instantiated."""

        class MissingAvail(SearchProvider):
            @property
            def provider_name(self) -> str:
                return "test"

            async def search(
                self,
                query,
                max_results=5,
                *,
                allowed_domains=None,
                blocked_domains=None,
            ):
                return {"results": []}

            async def close(self):
                pass

        with pytest.raises(TypeError):
            MissingAvail()  # type: ignore[abstract]

    def test_subclass_missing_search_raises(self):
        """A subclass that omits search cannot be instantiated."""

        class MissingSearch(SearchProvider):
            @property
            def provider_name(self) -> str:
                return "test"

            async def is_available(self) -> bool:
                return True

            async def close(self):
                pass

        with pytest.raises(TypeError):
            MissingSearch()  # type: ignore[abstract]

    def test_subclass_missing_close_raises(self):
        """A subclass that omits close cannot be instantiated."""

        class MissingClose(SearchProvider):
            @property
            def provider_name(self) -> str:
                return "test"

            async def is_available(self) -> bool:
                return True

            async def search(
                self,
                query,
                max_results=5,
                *,
                allowed_domains=None,
                blocked_domains=None,
            ):
                return {"results": []}

        with pytest.raises(TypeError):
            MissingClose()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self):
        """A subclass implementing all methods can be instantiated."""

        class CompleteProvider(SearchProvider):
            @property
            def provider_name(self) -> str:
                return "test"

            async def is_available(self) -> bool:
                return True

            async def search(
                self,
                query,
                max_results=5,
                *,
                allowed_domains=None,
                blocked_domains=None,
            ):
                return {"results": []}

            async def close(self):
                pass

        provider = CompleteProvider()
        assert provider.provider_name == "test"

    @pytest.mark.asyncio
    async def test_complete_subclass_methods_work(self):
        """A complete subclass's methods return expected values."""

        class WorkingProvider(SearchProvider):
            @property
            def provider_name(self) -> str:
                return "working"

            async def is_available(self) -> bool:
                return True

            async def search(
                self,
                query: str,
                max_results: int = 5,
                *,
                allowed_domains: Optional[List[str]] = None,
                blocked_domains: Optional[List[str]] = None,
            ) -> Dict[str, Any]:
                return {
                    "results": [
                        {"type": "web_search_result", "url": "http://example.com"}
                    ]
                }

            async def close(self):
                pass

        provider = WorkingProvider()
        assert provider.provider_name == "working"
        assert await provider.is_available() is True
        result = await provider.search("test")
        assert "results" in result
        assert len(result["results"]) == 1
        await provider.close()  # Should not raise

    @pytest.mark.asyncio
    async def test_search_accepts_domain_filters(self):
        """The search() method accepts allowed_domains and blocked_domains kwargs."""

        class DomainProvider(SearchProvider):
            def __init__(self):
                self.last_allowed = None
                self.last_blocked = None

            @property
            def provider_name(self) -> str:
                return "domain-test"

            async def is_available(self) -> bool:
                return True

            async def search(
                self,
                query: str,
                max_results: int = 5,
                *,
                allowed_domains: Optional[List[str]] = None,
                blocked_domains: Optional[List[str]] = None,
            ) -> Dict[str, Any]:
                self.last_allowed = allowed_domains
                self.last_blocked = blocked_domains
                return {"results": []}

            async def close(self):
                pass

        provider = DomainProvider()
        await provider.search(
            "test",
            allowed_domains=["example.com"],
            blocked_domains=["bad.com"],
        )
        assert provider.last_allowed == ["example.com"]
        assert provider.last_blocked == ["bad.com"]
