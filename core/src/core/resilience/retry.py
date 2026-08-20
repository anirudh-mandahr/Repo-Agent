"""Timeout and bounded retry for async MCP calls."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from core.exceptions import AgentUnavailableError
from core.logging import get_logger

log = get_logger(__name__)

TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
    AgentUnavailableError,
)

_CANCEL_GRACE_S = 0.25


async def await_with_timeout_retry[T](
    factory: Callable[[], Awaitable[T]],
    *,
    timeout_s: float,
    retry_count: int,
    transient: tuple[type[BaseException], ...] = TRANSIENT_ERRORS,
) -> T:
    """Await ``factory()`` with a timeout and bounded retry on transient failure.

    ``asyncio.wait_for`` waits for cancelled tasks to finish. MCP streamable-HTTP
    reads can ignore cancellation (anyio cancel scopes), which would hang the
    gateway past ``GATEWAY_CHAT_TIMEOUT_S``. This helper abandons a stuck task
    after a short grace period so the caller can return an error.

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

        async def _run() -> T:
            return await factory()

        task: asyncio.Task[T] = asyncio.create_task(_run())
        done, _pending = await asyncio.wait({task}, timeout=timeout_s)
        if task not in done:
            await _abandon(task)
            last_exc = TimeoutError(f"timed out after {timeout_s}s")
            if attempt + 1 >= attempts:
                raise last_exc
            log.warning(
                "mcp.retry",
                attempt=attempt + 1,
                retry_count=retry_count,
                error=str(last_exc),
            )
            continue
        try:
            return task.result()
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise  # the caller itself is being cancelled
            # The child task was cancelled from inside the MCP client (anyio
            # cancel-scope teardown), not by us -- we only cancel on timeout,
            # and then we never read the result. Re-raising would cancel the
            # request task without a response; treat it as a transient
            # transport failure instead.
            last_exc = ConnectionError("mcp call cancelled by transport teardown")
            if attempt + 1 >= attempts:
                raise last_exc from None
            log.warning(
                "mcp.retry",
                attempt=attempt + 1,
                retry_count=retry_count,
                error=str(last_exc),
            )
            continue
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


async def _abandon(task: asyncio.Task[Any]) -> None:
    """Cancel ``task`` without waiting forever for the cancellation to finish."""
    task.cancel()
    await asyncio.wait({task}, timeout=_CANCEL_GRACE_S)
