"""FastAPI middleware for request observability."""

import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique ``X-Correlation-ID`` to every request/response.

    The ID is generated per-request (or reused from the incoming
    ``X-Correlation-ID`` header if present) and stored in
    ``request.state.correlation_id`` for use by endpoint handlers
    and logging.

    The ID is also returned to the client in the response header.
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process a request, attaching correlation ID."""
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
