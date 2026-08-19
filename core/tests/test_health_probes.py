"""Tests for real health probes and the MCP healthcheck CLI."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from core.health import (
    HealthProbeError,
    HealthStatus,
    check_code_analyst_health,
    check_graph_query_health,
    check_memory_health,
    check_orchestrator_health,
    main,
    probe_neo4j,
    probe_repo_mount,
    probe_sqlite,
)


def test_probe_neo4j_fails_when_dependency_down(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Boom:
        def ping(self) -> None:
            raise OSError("connection refused")

        def close(self) -> None:
            return None

    monkeypatch.setattr("core.graph.client.GraphClient", lambda: _Boom())
    with pytest.raises(HealthProbeError, match="neo4j unreachable"):
        probe_neo4j()
    status = check_graph_query_health()
    assert status.status == "error"
    assert status.agent == "graph_query"
    assert "neo4j" in (status.detail or "").lower() or "refused" in (status.detail or "").lower()


def test_probe_sqlite_fails_when_parent_is_a_file(tmp_path: Path) -> None:
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("x", encoding="utf-8")
    db_path = str(blocked / "memory.db")
    with pytest.raises(HealthProbeError, match="sqlite"):
        probe_sqlite(db_path)
    status = check_memory_health(db_path=db_path)
    assert status.status == "error"
    assert status.agent == "memory"


def test_probe_sqlite_ok(tmp_path: Path) -> None:
    db_path = str(tmp_path / "memory.db")
    probe_sqlite(db_path)
    status = check_memory_health(db_path=db_path)
    assert status.status == "ok"


def test_probe_repo_mount_fails_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "nope"
    with pytest.raises(HealthProbeError, match="repo mount"):
        probe_repo_mount(str(missing))


def test_code_analyst_health_fails_when_repo_missing(tmp_path: Path) -> None:
    status = asyncio.run(
        check_code_analyst_health(
            repo_root=str(tmp_path / "missing"),
            graph_query_url="http://127.0.0.1:9/mcp",
        )
    )
    assert status.status == "error"
    assert status.agent == "code_analyst"


def test_code_analyst_health_fails_when_graph_query_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    async def _fail(url: str, timeout_s: float = 2.0) -> None:
        _ = url, timeout_s
        raise HealthProbeError("graph_query down")

    monkeypatch.setattr("core.health.probe_graph_query_mcp", _fail)
    status = asyncio.run(
        check_code_analyst_health(repo_root=str(repo), graph_query_url="http://gq/mcp")
    )
    assert status.status == "error"
    assert "graph_query" in (status.detail or "")


def test_code_analyst_health_ok_when_probes_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    async def _ok(url: str, timeout_s: float = 2.0) -> None:
        _ = url, timeout_s

    monkeypatch.setattr("core.health.probe_graph_query_mcp", _ok)
    status = asyncio.run(
        check_code_analyst_health(repo_root=str(repo), graph_query_url="http://gq/mcp")
    )
    assert status.status == "ok"


def test_orchestrator_health_fails_when_downstream_down() -> None:
    status = asyncio.run(
        check_orchestrator_health(
            downstream={"memory": HealthStatus(status="error", agent="memory", detail="down")}
        )
    )
    assert status.status == "error"
    assert status.agent == "orchestrator"
    assert "memory" in (status.detail or "")


def test_orchestrator_health_ok_when_downstream_healthy() -> None:
    status = asyncio.run(
        check_orchestrator_health(
            downstream={"memory": HealthStatus(status="ok", agent="memory")}
        )
    )
    assert status.status == "ok"


def test_healthcheck_cli_exits_nonzero_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fail(url: str, *, timeout_s: float = 5.0) -> HealthStatus:
        _ = url, timeout_s
        return HealthStatus(status="error", agent="healthcheck", detail="down")

    monkeypatch.setattr("core.health.mcp_health_ok", _fail)
    assert main(["--url", "http://127.0.0.1:9/mcp"]) == 1


def test_graph_client_ping_is_single_attempt() -> None:
    from core.graph.client import GraphClient, GraphSettings

    class _Driver:
        def __init__(self) -> None:
            self.calls = 0

        def verify_connectivity(self) -> None:
            self.calls += 1
            raise OSError("down")

        def close(self) -> None:
            return None

    client = GraphClient(GraphSettings(uri="bolt://localhost:7687", user="neo4j", password="x"))
    driver = _Driver()
    client._driver = driver  # type: ignore[assignment]
    with pytest.raises(OSError, match="down"):
        client.ping()
    assert driver.calls == 1


def test_collect_downstream_health_maps_errors() -> None:
    from core.health import collect_downstream_health

    class _Pool:
        async def call(self, agent: str, tool: str, arguments: Any, **kwargs: Any) -> Any:
            _ = tool, arguments, kwargs
            if agent == "memory":
                raise ConnectionError("down")
            return {"status": "ok", "agent": agent}

    results = asyncio.run(collect_downstream_health(_Pool(), timeout_s=1.0))
    assert results["memory"].status == "error"
    assert results["indexer"].status == "ok"
