"""JSON structured logging. Every event includes correlation_id."""

from __future__ import annotations

import logging
import os
import uuid
from contextvars import ContextVar
from typing import cast

import structlog
from structlog.typing import EventDict, FilteringBoundLogger, WrappedLogger

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")
_configured = False


def get_correlation_id() -> str:
    """Return the correlation id bound to the current context."""
    return _correlation_id.get()


def bind_correlation_id(correlation_id: str | None = None) -> str:
    """Bind a correlation id to the current context and return it."""
    bound = correlation_id or str(uuid.uuid4())
    _correlation_id.set(bound)
    structlog.contextvars.bind_contextvars(correlation_id=bound)
    return bound


def _add_correlation_id(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    event_dict.setdefault("correlation_id", get_correlation_id())
    return event_dict


def configure_logging(log_level: str | None = None) -> None:
    """Configure structlog for JSON output on stdout."""
    global _configured
    level_name = (log_level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    structlog.contextvars.clear_contextvars()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_correlation_id,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> FilteringBoundLogger:
    """Return a bound structlog logger. Configures logging on first use."""
    if not _configured:
        configure_logging()
    return cast(FilteringBoundLogger, structlog.get_logger(name))
