"""Unit tests for GraphClient batching, reads, and connectivity retry. No database required."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import pytest

from core.graph.client import WRITE_BATCH_SIZE, GraphClient, GraphSettings, unwind_write


class _Record:
    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def data(self) -> dict[str, Any]:
        return self._data


class _FakeResult:
    def __init__(self, rows: Sequence[Mapping[str, Any]] | None = None) -> None:
        self._rows = [_Record(dict(row)) for row in (rows or [])]

    def __iter__(self) -> Iterator[_Record]:
        return iter(self._rows)


class _FakeTx:
    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    def run(
        self,
        query: str,
        rows: Sequence[Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> _FakeResult:
        if self._store.get("fail_read"):
            raise RuntimeError("boom")
        payload: Any = rows if rows is not None else kwargs
        self._store.setdefault("runs", []).append((query, payload))
        return _FakeResult(self._store.get("read_rows", []))

    def commit(self) -> None:
        self._store["committed"] = True

    def rollback(self) -> None:
        self._store["rolled_back"] = True


class _FakeSession:
    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store

    def __enter__(self) -> _FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute_write(self, work: Any, *args: Any) -> Any:
        return work(_FakeTx(self._store), *args)

    def begin_transaction(self, timeout: float | None = None) -> _FakeTx:
        self._store["timeout_s"] = timeout
        return _FakeTx(self._store)


class _FakeDriver:
    def __init__(self, store: dict[str, Any] | None = None) -> None:
        self.store = store if store is not None else {}
        self.closed = False
        self.verify_calls = 0
        self.verify_failures = 0

    def session(self, default_access_mode: str | None = None) -> _FakeSession:
        self.store["access_mode"] = default_access_mode
        return _FakeSession(self.store)

    def verify_connectivity(self) -> None:
        self.verify_calls += 1
        if self.verify_calls <= self.verify_failures:
            raise OSError("not ready")

    def close(self) -> None:
        self.closed = True


def _client_with_driver(driver: _FakeDriver) -> GraphClient:
    client = GraphClient(GraphSettings(uri="bolt://localhost:7687", user="neo4j", password="test"))
    client._driver = driver  # type: ignore[assignment]
    return client


def test_write_batch_size_is_500() -> None:
    assert WRITE_BATCH_SIZE == 500


def test_run_write_batch_chunks_rows() -> None:
    driver = _FakeDriver()
    client = _client_with_driver(driver)
    rows = [{"i": i} for i in range(WRITE_BATCH_SIZE + 1)]
    client.run_write_batch("UNWIND $rows AS row RETURN row", rows)
    runs = driver.store["runs"]
    assert len(runs) == 2
    assert len(runs[0][1]) == WRITE_BATCH_SIZE
    assert len(runs[1][1]) == 1


def test_run_write_batch_rejects_non_unwind() -> None:
    client = _client_with_driver(_FakeDriver())
    with pytest.raises(ValueError, match="UNWIND"):
        client.run_write_batch("CREATE (n:Node)", [{"x": 1}])


def test_run_write_batch_skips_empty_rows() -> None:
    driver = _FakeDriver()
    client = _client_with_driver(driver)
    client.run_write_batch("UNWIND $rows AS row RETURN row", [])
    assert "runs" not in driver.store


def test_run_read_uses_explicit_read_only_transaction() -> None:
    from neo4j import READ_ACCESS

    driver = _FakeDriver({"read_rows": [{"n": 1}]})
    client = _client_with_driver(driver)
    records = client.run_read("RETURN 1 AS n", {"k": "v"}, timeout_s=7)
    assert records == [{"n": 1}]
    assert driver.store["access_mode"] == READ_ACCESS
    assert driver.store["timeout_s"] == 7
    assert driver.store["committed"] is True
    query, params = driver.store["runs"][0]
    assert query == "RETURN 1 AS n"
    assert params == {"k": "v"}


def test_verify_connectivity_retries_with_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("core.graph.client.time.sleep", sleeps.append)
    driver = _FakeDriver()
    driver.verify_failures = 2
    client = _client_with_driver(driver)
    client.verify_connectivity()
    assert driver.verify_calls == 3
    assert sleeps == [0.5, 1.0]


def test_graph_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_URI", "bolt://example:7687")
    monkeypatch.setenv("NEO4J_USER", "alice")
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    settings = GraphSettings.from_env()
    assert settings.uri == "bolt://example:7687"
    assert settings.user == "alice"
    assert settings.password == "secret"
    client = GraphClient()
    assert client._settings.uri == "bolt://example:7687"


def test_run_read_rolls_back_on_error() -> None:
    driver = _FakeDriver({"fail_read": True})
    client = _client_with_driver(driver)
    with pytest.raises(RuntimeError, match="boom"):
        client.run_read("RETURN 1")
    assert driver.store.get("rolled_back") is True
    assert driver.store.get("committed") is not True


def test_verify_connectivity_raises_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("core.graph.client.time.sleep", lambda _delay: None)
    driver = _FakeDriver()
    driver.verify_failures = 99
    client = _client_with_driver(driver)
    with pytest.raises(OSError, match="not ready"):
        client.verify_connectivity()
    assert driver.verify_calls == 8


def test_context_manager_and_aliases() -> None:
    from neo4j import WRITE_ACCESS

    driver = _FakeDriver({"read_rows": [{"ok": True}]})
    client = _client_with_driver(driver)
    with client as entered:
        assert entered is client
        assert client.read("RETURN 1") == [{"ok": True}]
        client.write_unwind("UNWIND $rows AS row RETURN row", [{"i": 1}])
    assert driver.closed is True
    assert driver.store["access_mode"] == WRITE_ACCESS


def test_ensure_schema_runs_each_statement() -> None:
    from core.graph.schema import SCHEMA_STATEMENTS, ensure_schema

    driver = _FakeDriver()
    client = _client_with_driver(driver)
    ensure_schema(client)
    queries = [query for query, _payload in driver.store["runs"]]
    assert queries == list(SCHEMA_STATEMENTS)


def test_unwind_write_rejects_non_unwind() -> None:
    class _Tx:
        def run(self, query: str, **kwargs: object) -> None:
            raise AssertionError("should not run")

    with pytest.raises(ValueError, match="UNWIND"):
        unwind_write(_Tx(), "CREATE (n:Node)", [])  # type: ignore[arg-type]
