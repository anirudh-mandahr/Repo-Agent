"""Timeout and bounded retry for async MCP calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from core.exceptions import AgentUnavailableError
from core.logging import get_logger

log = get_logger(__name__)

TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
    AgentUnavailableError,
)


async def await_with_timeout_retry[T](
    factory: Callable[[], Awaitable[T]],
    *,
    timeout_s: float,
    retry_count: int,
    transient: tuple[type[BaseException], ...] = TRANSIENT_ERRORS,
) -> T:
    """Await ``factory()`` with a timeout and bounded retry on transient failure.

    Args:
        factory: Zero-argument async callable.
        timeout_s: Per-attempt timeout in seconds.
        retry_count: Extra attempts after the first.
        transient: Exception types that trigger a retry.

    Returns:
        The factory result.

    Raises:
        The last transient error, or a non-transient error immediately.
    """
    attempts = max(0, retry_count) + 1
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(factory(), timeout=timeout_s)
        except transient as exc:
            last_exc = exc
            if attempt + 1 >= attempts:
                raise
            log.warning(
                "mcp.retry",
                attempt=attempt + 1,
                retry_count=retry_count,
                error=str(exc),
            )
    assert last_exc is not None
    raise last_exc
