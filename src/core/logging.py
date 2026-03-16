"""Centralised logging configuration for the proxy.

When ``LOG_FORMAT=json`` (default in production), log records are emitted as
single-line JSON objects suitable for log aggregation (ELK, CloudWatch, etc.).
When ``LOG_FORMAT=text``, a human-readable format is used instead.

A ``correlation_id`` field is attached to every log record by the
``CorrelationIdMiddleware`` (see ``src/middleware.py``).  In JSON mode
it appears as a top-level key; in text mode it is interpolated into the
format string.
"""

import logging
import os

from pythonjsonlogger.json import JsonFormatter
from src.core.config import config

_LOG_FORMAT_ENV = os.environ.get("LOG_FORMAT", "text").lower()


def _configure_logging() -> None:
    """Set up the root logger with the configured format and level."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, config.log_level))

    # Remove any existing handlers (e.g. from basicConfig)
    root.handlers.clear()

    handler = logging.StreamHandler()

    if _LOG_FORMAT_ENV == "json":
        formatter = JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            rename_fields={"asctime": "timestamp", "levelname": "level"},
        )
    else:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Configure uvicorn to be quieter
    for uvicorn_logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error"]:
        logging.getLogger(uvicorn_logger_name).setLevel(logging.WARNING)


_configure_logging()

logger = logging.getLogger(__name__)
