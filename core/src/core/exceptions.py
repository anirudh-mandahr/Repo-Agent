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
        super().__init__(message)
        self.agent = agent
        self.correlation_id = correlation_id or get_correlation_id()
        self.data = data


class RoutingError(AgentError):
    """Raised when the orchestrator cannot decide which agents to call."""


class AgentUnavailableError(AgentError):
    """Raised when an agent is unavailable or times out."""


class SynthesisError(AgentError):
    """Raised when the orchestrator cannot synthesize a final answer."""


class SchemaValidationError(AgentError, ValueError):
    """Raised when structured-output parsing/validation fails."""

