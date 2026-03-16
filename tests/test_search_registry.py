"""Tests for the search provider registry."""

from typing import Any, Dict, List, Optional

import pytest
from src.services.search.base import SearchProvider
from src.services.search.registry import (
    _registry,
    available_providers,
    get_provider,
    register_provider,
)


class _DummyProvider(SearchProvider):
    """Minimal concrete provider for testing the registry."""

    def __init__(self, name: str = "dummy"):
        self._name = name

    @property
    def provider_name(self) -> str:
        return self._name

    async def is_available(self) -> bool:
        return True

    async def search(  # type: ignore[override]
        self,
        query: str,
        max_results: int = 5,
        *,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return {"results": []}

    async def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _clear_registry():
    """Ensure each test starts with a clean registry."""
    saved = dict(_registry)
    _registry.clear()
    yield
    _registry.clear()
    _registry.update(saved)


class TestRegisterProvider:
    """Test provider registration."""

    def test_register_and_retrieve(self):
        """Registered provider can be retrieved by name."""
        register_provider("dummy", lambda: _DummyProvider("dummy"))
        provider = get_provider("dummy")
        assert provider is not None
        assert provider.provider_name == "dummy"

    def test_case_insensitive_lookup(self):
        """Registry lookup is case-insensitive."""
        register_provider("Tavily", lambda: _DummyProvider("tavily"))
        assert get_provider("tavily") is not None
        assert get_provider("TAVILY") is not None
        assert get_provider("Tavily") is not None

    def test_case_insensitive_registration(self):
        """Registration normalises the name to lowercase."""
        register_provider("UPPER", lambda: _DummyProvider("upper"))
        assert "upper" in _registry
        assert get_provider("upper") is not None

    def test_unknown_provider_returns_none(self):
        """Requesting an unregistered name returns None."""
        assert get_provider("nonexistent") is None

    def test_register_overwrites_existing(self):
        """Re-registering the same name overwrites the factory."""
        register_provider("dup", lambda: _DummyProvider("first"))
        register_provider("dup", lambda: _DummyProvider("second"))
        provider = get_provider("dup")
        assert provider is not None
        assert provider.provider_name == "second"


class TestAvailableProviders:
    """Test listing registered providers."""

    def test_empty_registry(self):
        """No providers registered returns empty list."""
        assert available_providers() == []

    def test_lists_registered_names(self):
        """All registered names are listed."""
        register_provider("alpha", lambda: _DummyProvider("alpha"))
        register_provider("beta", lambda: _DummyProvider("beta"))
        names = available_providers()
        assert sorted(names) == ["alpha", "beta"]

    def test_names_are_lowercase(self):
        """Listed names are always lowercase."""
        register_provider("MixedCase", lambda: _DummyProvider("mixed"))
        names = available_providers()
        assert names == ["mixedcase"]


class TestGetProvider:
    """Test provider instantiation via factory."""

    def test_factory_called_each_time(self):
        """Each get_provider call invokes the factory (no caching)."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return _DummyProvider("counter")

        register_provider("counter", factory)
        get_provider("counter")
        get_provider("counter")
        assert call_count == 2

    def test_factory_returning_none_propagates(self):
        """A factory that returns None results in None from get_provider."""
        register_provider("bad", lambda: None)  # type: ignore[arg-type]
        assert get_provider("bad") is None
