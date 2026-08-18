"""Health check models and helpers shared by agents and the gateway."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthStatus(BaseModel):
    """Health payload returned by each agent MCP health tool."""

    status: Literal["ok", "error"]
    agent: str
    detail: str | None = None


class AggregateHealth(BaseModel):
    """Gateway aggregate of all five agent health tools."""

    status: Literal["ok", "degraded"]
    agents: dict[str, HealthStatus] = Field(default_factory=dict)


def agent_health(agent: str) -> HealthStatus:
    """Return a successful health status for the named agent."""
    return HealthStatus(status="ok", agent=agent)
