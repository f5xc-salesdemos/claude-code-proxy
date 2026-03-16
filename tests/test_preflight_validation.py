"""Unit tests for pre-flight context window validation in endpoints.py."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.core.model_registry import ModelLimits


def _make_registry(limits=None):
    """Build a mock ModelRegistry."""
    registry = MagicMock()
    registry.get_limits.return_value = limits
    return registry


class TestPreflightValidation:
    """Unit tests for the pre-flight context window check."""

    def test_oversized_request_rejected(self, monkeypatch):
        """Oversized request returns 400 Claude-format error."""
        import src.api.endpoints as ep

        limits = ModelLimits(max_input_tokens=1000, max_output_tokens=100)
        mock_registry = _make_registry(limits=limits)

        monkeypatch.setattr(ep, "model_registry", mock_registry)
        monkeypatch.setattr(ep, "estimate_tokens", lambda *a, **kw: 950)
        monkeypatch.setattr(ep.config, "model_registry_enabled", True)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 0.95)

        # 950 + 200 = 1150 > 1000 * 0.95 = 950 → reject
        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 200,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "invalid_request_error"
        assert (
            "token" in data["error"]["message"].lower()
            or "context" in data["error"]["message"].lower()
        )

    def test_unknown_model_passes(self, monkeypatch):
        """Unknown model (registry returns None) is not rejected by pre-flight."""
        import src.api.endpoints as ep

        mock_registry = _make_registry(limits=None)
        monkeypatch.setattr(ep, "model_registry", mock_registry)
        monkeypatch.setattr(ep.config, "model_registry_enabled", True)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 0.95)

        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "some-unknown-model",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        # Should NOT get a 400 from pre-flight; any other status is fine
        # (502/500 from upstream mock being absent is expected)
        assert response.status_code != 400

    def test_disabled_skips(self, monkeypatch):
        """model_registry_enabled=False skips pre-flight entirely."""
        import src.api.endpoints as ep

        # Even with tiny limits that would normally reject, disabling skips check
        limits = ModelLimits(max_input_tokens=1, max_output_tokens=1)
        mock_registry = _make_registry(limits=limits)
        monkeypatch.setattr(ep, "model_registry", mock_registry)
        monkeypatch.setattr(ep.config, "model_registry_enabled", False)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 0.95)
        monkeypatch.setattr(ep, "estimate_tokens", lambda *a, **kw: 999_999)

        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 999_999,
                "messages": [{"role": "user", "content": "x" * 10000}],
            },
        )
        # Pre-flight is disabled — must NOT get a 400 rejection
        assert response.status_code != 400

    def test_underlimit_passes(self, monkeypatch):
        """Request within limits is not rejected by pre-flight."""
        import src.api.endpoints as ep

        limits = ModelLimits(max_input_tokens=1_000_000, max_output_tokens=128_000)
        mock_registry = _make_registry(limits=limits)
        monkeypatch.setattr(ep, "model_registry", mock_registry)
        monkeypatch.setattr(ep, "estimate_tokens", lambda *a, **kw: 10)
        monkeypatch.setattr(ep.config, "model_registry_enabled", True)
        monkeypatch.setattr(ep.config, "model_registry_safety_margin", 0.95)

        from src.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-opus-4-6",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        # 10 + 100 = 110 << 950000 → no pre-flight rejection
        assert response.status_code != 400
