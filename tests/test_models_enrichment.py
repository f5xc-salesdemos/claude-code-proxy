"""Unit tests for /v1/models enrichment with Claude aliases and context_window."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from src.core.model_registry import ModelLimits


def _upstream_models_response(*model_ids):
    """Build a minimal upstream /models JSON body."""
    return {
        "object": "list",
        "data": [
            {"id": mid, "object": "model", "created": 0, "owned_by": "openai"}
            for mid in model_ids
        ],
    }


class TestModelsEnrichment:
    """Verify /v1/models enrichment with Claude aliases."""

    @pytest.mark.asyncio
    async def test_claude_aliases_injected(self, monkeypatch):
        """Claude aliases are injected into the /v1/models response."""
        import src.api.endpoints as ep

        # Mock registry that knows about the big/middle/small models
        mock_registry = MagicMock()
        mock_registry.get_limits.return_value = ModelLimits(1_000_000, 128_000)
        monkeypatch.setattr(ep, "model_registry", mock_registry)

        # Mock the httpx client to return a canned upstream response
        upstream_body = _upstream_models_response("gpt-4o", "gpt-4o-mini")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = upstream_body

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        monkeypatch.setattr(ep, "_get_httpx_client", lambda: mock_client)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()["data"]
        ids = [m["id"] for m in data]
        assert "claude-opus" in ids
        assert "claude-sonnet" in ids
        assert "claude-haiku" in ids
        # Originals preserved
        assert "gpt-4o" in ids
        assert "gpt-4o-mini" in ids

        # Check owned_by for Claude aliases
        for m in data:
            if m["id"].startswith("claude-"):
                assert m["owned_by"] == "anthropic"

    @pytest.mark.asyncio
    async def test_context_window_field(self, monkeypatch):
        """Claude alias entries have context_window as positive integer."""
        import src.api.endpoints as ep

        mock_registry = MagicMock()
        mock_registry.get_limits.return_value = ModelLimits(1_000_000, 128_000)
        monkeypatch.setattr(ep, "model_registry", mock_registry)

        upstream_body = _upstream_models_response("gpt-4o")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = upstream_body

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        monkeypatch.setattr(ep, "_get_httpx_client", lambda: mock_client)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")
        data = response.json()["data"]

        opus_entry = next(m for m in data if m["id"] == "claude-opus")
        assert "context_window" in opus_entry
        assert isinstance(opus_entry["context_window"], int)
        assert opus_entry["context_window"] == 1_000_000

    @pytest.mark.asyncio
    async def test_upstream_preserved(self, monkeypatch):
        """Original upstream models are preserved in the response."""
        import src.api.endpoints as ep

        mock_registry = MagicMock()
        mock_registry.get_limits.return_value = None  # No limits known
        monkeypatch.setattr(ep, "model_registry", mock_registry)

        upstream_body = _upstream_models_response("my-custom-model", "another-model")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = upstream_body

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        monkeypatch.setattr(ep, "_get_httpx_client", lambda: mock_client)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")
        data = response.json()["data"]
        ids = [m["id"] for m in data]

        assert "my-custom-model" in ids
        assert "another-model" in ids

    @pytest.mark.asyncio
    async def test_no_duplicates(self, monkeypatch):
        """If upstream already has 'claude-opus', don't inject a duplicate."""
        import src.api.endpoints as ep

        mock_registry = MagicMock()
        mock_registry.get_limits.return_value = ModelLimits(1_000_000, 128_000)
        monkeypatch.setattr(ep, "model_registry", mock_registry)

        # Upstream already has claude-opus
        upstream_body = {
            "object": "list",
            "data": [
                {
                    "id": "claude-opus",
                    "object": "model",
                    "created": 0,
                    "owned_by": "upstream",
                },
                {"id": "gpt-4o", "object": "model", "created": 0, "owned_by": "openai"},
            ],
        }
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = upstream_body

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        monkeypatch.setattr(ep, "_get_httpx_client", lambda: mock_client)

        from fastapi.testclient import TestClient
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/v1/models")
        data = response.json()["data"]

        # Count how many claude-opus entries
        opus_count = sum(1 for m in data if m["id"] == "claude-opus")
        assert opus_count == 1, f"Expected 1 claude-opus, got {opus_count}"
