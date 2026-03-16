"""Unit tests for logging configuration and correlation ID middleware."""

import logging
import uuid

from starlette.testclient import TestClient


class TestLoggingConfiguration:
    """Verify that the logging module configures handlers correctly."""

    def test_root_logger_has_handler(self):
        """Root logger should have at least one handler after import."""
        # Import triggers _configure_logging()
        import src.core.logging  # noqa: F401  # pylint: disable=unused-import

        root = logging.getLogger()
        assert len(root.handlers) > 0

    def test_logger_level_matches_config(self):
        """Logger level should match the configured log level."""
        from src.core.config import config

        root = logging.getLogger()
        expected = getattr(logging, config.log_level)
        assert root.level == expected

    def test_uvicorn_loggers_suppressed(self):
        """Uvicorn loggers should be set to WARNING or higher."""
        import src.core.logging  # noqa: F401  # pylint: disable=unused-import

        for name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
            assert logging.getLogger(name).level >= logging.WARNING


class TestCorrelationIdMiddleware:
    """Test correlation ID propagation."""

    def _make_app(self):
        """Create a minimal FastAPI app with the middleware."""
        from fastapi import FastAPI, Request

        from src.middleware import CorrelationIdMiddleware

        app = FastAPI()
        app.add_middleware(CorrelationIdMiddleware)

        @app.get("/test")
        async def test_endpoint(request: Request):
            """Return the correlation ID from request state."""
            return {"correlation_id": request.state.correlation_id}

        return app

    def test_generates_correlation_id(self):
        """Middleware generates a UUID when no header is provided."""
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        cid = response.headers.get("X-Correlation-ID")
        assert cid is not None
        # Verify it's a valid UUID
        uuid.UUID(cid)

    def test_reuses_provided_correlation_id(self):
        """Middleware reuses the correlation ID from the request header."""
        app = self._make_app()
        client = TestClient(app)
        custom_id = "my-custom-id-123"
        response = client.get("/test", headers={"X-Correlation-ID": custom_id})
        assert response.status_code == 200
        assert response.headers["X-Correlation-ID"] == custom_id
        body = response.json()
        assert body["correlation_id"] == custom_id

    def test_correlation_id_available_in_state(self):
        """Correlation ID is stored in request.state."""
        app = self._make_app()
        client = TestClient(app)
        response = client.get("/test")
        body = response.json()
        cid = body["correlation_id"]
        # Should match the response header
        assert cid == response.headers["X-Correlation-ID"]
