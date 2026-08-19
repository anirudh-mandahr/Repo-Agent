"""Tests for health helpers."""

from __future__ import annotations

from core.health import agent_health


def test_agent_health_ok() -> None:
    status = agent_health("orchestrator")
    assert status.status == "ok"
    assert status.agent == "orchestrator"
    assert status.model_dump(exclude_none=True) == {"status": "ok", "agent": "orchestrator"}
