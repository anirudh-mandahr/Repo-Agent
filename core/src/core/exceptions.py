"""Project-wide exception hierarchy shared across agents.

These exceptions carry both:
- the agent name that produced the error
- the current `correlation_id` for request-level tracing
"""

from __future__ import annotations

from typing import Any

from core.logging import get_correlation_id


class AgentError(Exception):
    """Base class for all agent failures surfaced through the orchestrator."""

    def __init__(
        self,
        *,
        agent: str,
        correlation_id: str | None = None,
        message: str,
        data: Any = None,
    ) -> None:
        """Record the failing agent, message, and correlation id.

        Args:
            agent: Agent that produced the error.
            correlation_id: Request id, or the bound logging id when omitted.
            message: Human-readable error text.
            data: Optional structured payload.
        """
        super().__init__(message)
        self.agent = agent
        self.correlation_id = correlation_id or get_correlation_id()
        self.data = data


class RoutingError(AgentError):
    """Raised when the orchestrator cannot decide which agents to call."""


class AgentUnavailableError(AgentError):
    """Raised when an agent is unavailable or times out."""


class CircuitBreakerOpenError(AgentUnavailableError):
    """Raised when a circuit breaker is open and the call is fail-fast."""


class SynthesisError(AgentError):
    """Raised when the orchestrator cannot synthesize a final answer."""


class SchemaValidationError(AgentError, ValueError):
    """Raised when structured-output parsing/validation fails."""


class GraphLookupError(AgentError):
    """Raised when the Code Analyst cannot query the Graph Query agent."""


class ConfigurationError(Exception):
    """Raised when required runtime configuration is missing or invalid."""


class UnsafeCloneUrlError(AgentError):
    """Raised when a clone URL fails the scheme and host allowlist."""

