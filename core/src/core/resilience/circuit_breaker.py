"""Per-agent circuit breaker for MCP calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Literal

from core.exceptions import AgentUnavailableError, CircuitBreakerOpenError
from core.health import CircuitBreakerSnapshot
from core.logging import get_logger

log = get_logger(__name__)

BreakerState = Literal["closed", "open", "half_open"]
Clock = Callable[[], float]

_FAILURE_TYPES: tuple[type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
    OSError,
    AgentUnavailableError,
)


class CircuitBreaker:
    """Fail fast after consecutive failures, then probe after a cooldown."""

    def __init__(
        self,
        agent: str,
        *,
        failure_threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Clock,
    ) -> None:
        """Create a breaker for ``agent``.

        Args:
            agent: Upstream agent name.
            failure_threshold: Consecutive failures that trip the breaker.
            cooldown_s: Seconds to stay open before a half-open probe.
            clock: Monotonic clock used for cooldown math.
        """
        self.agent = agent
        self._failure_threshold = max(1, failure_threshold)
        self._cooldown_s = max(0.0, cooldown_s)
        self._clock = clock
        self._state: BreakerState = "closed"
        self._consecutive_failures = 0
        self._opened_at: float | None = None

    @property
    def state(self) -> BreakerState:
        """Current breaker state, advancing open → half_open when cooldown elapses.

        Returns:
            The breaker state after applying cooldown.
        """
        self._maybe_half_open()
        return self._state

    def snapshot(self) -> CircuitBreakerSnapshot:
        """Return a serializable snapshot of breaker state.

        Returns:
            Health/metadata payload for this breaker.
        """
        state = self.state
        remaining: float | None = None
        if state == "open" and self._opened_at is not None:
            remaining = max(0.0, self._cooldown_s - (self._clock() - self._opened_at))
        return CircuitBreakerSnapshot(
            state=state,
            consecutive_failures=self._consecutive_failures,
            cooldown_remaining_s=remaining,
        )

    def guard(self) -> None:
        """Raise if the breaker is open and cooldown has not elapsed.

        Raises:
            CircuitBreakerOpenError: The breaker is open.
        """
        if self.state == "open":
            raise CircuitBreakerOpenError(
                agent=self.agent,
                message=f"circuit breaker open for {self.agent}",
                data=self.snapshot().model_dump(mode="json"),
            )

    def record_success(self) -> None:
        """Reset consecutive failures and close the breaker."""
        was_open = self._state != "closed"
        self._state = "closed"
        self._consecutive_failures = 0
        self._opened_at = None
        if was_open:
            log.info("circuit_breaker.closed", agent=self.agent)

    def record_failure(self) -> None:
        """Count a failure and trip the breaker when the threshold is reached."""
        if self._state == "half_open":
            self._trip()
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._trip()

    async def run[T](self, factory: Callable[[], Awaitable[T]]) -> T:
        """Run ``factory`` under this breaker.

        Args:
            factory: Zero-argument async callable that performs the MCP call.

        Returns:
            The factory result.

        Raises:
            CircuitBreakerOpenError: The breaker is open.
        """
        self.guard()
        try:
            result = await factory()
        except _FAILURE_TYPES:
            self.record_failure()
            raise
        except Exception:
            raise
        else:
            self.record_success()
            return result

    def _maybe_half_open(self) -> None:
        if self._state != "open" or self._opened_at is None:
            return
        if self._clock() - self._opened_at >= self._cooldown_s:
            self._state = "half_open"
            log.info("circuit_breaker.half_open", agent=self.agent)

    def _trip(self) -> None:
        self._state = "open"
        self._opened_at = self._clock()
        log.warning(
            "circuit_breaker.open",
            agent=self.agent,
            consecutive_failures=self._consecutive_failures,
            cooldown_s=self._cooldown_s,
        )


class CircuitBreakerRegistry:
    """Named circuit breakers with per-agent thresholds."""

    def __init__(
        self,
        *,
        default_failure_threshold: int = 3,
        default_cooldown_s: float = 30.0,
        per_agent: Mapping[str, tuple[int, float]] | None = None,
        clock: Clock,
    ) -> None:
        """Create a registry.

        Args:
            default_failure_threshold: Threshold used when an agent has no override.
            default_cooldown_s: Cooldown used when an agent has no override.
            per_agent: Optional ``agent -> (threshold, cooldown_s)`` overrides.
            clock: Shared monotonic clock.
        """
        self._default_threshold = default_failure_threshold
        self._default_cooldown_s = default_cooldown_s
        self._per_agent = dict(per_agent or {})
        self._clock = clock
        self._breakers: dict[str, CircuitBreaker] = {}

    @classmethod
    def from_orchestrator_settings(
        cls,
        settings: object,
        *,
        clock: Clock,
    ) -> CircuitBreakerRegistry:
        """Build a registry from :class:`~core.settings.OrchestratorSettings`.

        Args:
            settings: Orchestrator settings object with ``breaker_for``.
            clock: Monotonic clock.

        Returns:
            Registry configured from settings.
        """
        default_threshold = int(getattr(settings, "breaker_failure_threshold", 3))
        default_cooldown = float(getattr(settings, "breaker_cooldown_s", 30.0))
        per_agent: dict[str, tuple[int, float]] = {}
        breaker_for = getattr(settings, "breaker_for", None)
        if callable(breaker_for):
            for agent in ("graph_query", "code_analyst", "indexer", "memory"):
                per_agent[agent] = breaker_for(agent)
        return cls(
            default_failure_threshold=default_threshold,
            default_cooldown_s=default_cooldown,
            per_agent=per_agent,
            clock=clock,
        )

    def get(self, agent: str) -> CircuitBreaker:
        """Return the breaker for ``agent``, creating it on first use.

        Args:
            agent: Upstream agent name.

        Returns:
            The breaker for that agent.
        """
        existing = self._breakers.get(agent)
        if existing is not None:
            return existing
        threshold, cooldown = self._per_agent.get(
            agent, (self._default_threshold, self._default_cooldown_s)
        )
        breaker = CircuitBreaker(
            agent,
            failure_threshold=threshold,
            cooldown_s=cooldown,
            clock=self._clock,
        )
        self._breakers[agent] = breaker
        return breaker

    def snapshot(self) -> dict[str, CircuitBreakerSnapshot]:
        """Snapshot every breaker that has been used.

        Returns:
            Mapping of agent name to breaker snapshot.
        """
        return {name: breaker.snapshot() for name, breaker in self._breakers.items()}
