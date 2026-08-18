"""Neo4j driver helpers. Writes are batched with UNWIND; user-facing Cypher is read-only."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Any, Self

from neo4j import READ_ACCESS, WRITE_ACCESS, Driver, GraphDatabase, ManagedTransaction, Session
from pydantic import BaseModel, Field, SecretStr

from core.logging import get_logger
from core.settings import Neo4jSettings

log = get_logger(__name__)

WRITE_BATCH_SIZE = 500
_VERIFY_ATTEMPTS = 8
_VERIFY_INITIAL_DELAY_S = 0.5
_VERIFY_MAX_DELAY_S = 8.0


class GraphSettings(BaseModel):
    """Bolt connection settings loaded from ``NEO4J_*`` environment variables."""

    uri: str = Field(default="bolt://neo4j:7687")
    user: str = Field(default="neo4j")
    password: str = Field(default="changeme123")

    @classmethod
    def from_env(cls) -> GraphSettings:
        """Load URI, user, and password from the process environment."""
        env = Neo4jSettings.from_env()
        return cls(uri=env.uri, user=env.user, password=env.password.get_secret_value())


def unwind_write(
    tx: ManagedTransaction,
    query: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Execute a batched write. `query` must use UNWIND $rows AS row."""
    _require_unwind(query)
    tx.run(query, rows=list(rows))


def _require_unwind(query: str) -> None:
    if "UNWIND" not in query.upper():
        raise ValueError("Write queries must batch rows with UNWIND")


class GraphClient:
    """Thin driver wrapper. Callers decide read vs write; user-facing queries use read-only txs."""

    def __init__(self, settings: GraphSettings | Neo4jSettings | None = None) -> None:
        if settings is None:
            self._settings: GraphSettings | Neo4jSettings = GraphSettings.from_env()
        else:
            self._settings = settings
        self._driver: Driver | None = None

    def connect(self) -> Driver:
        """Open a driver if needed and return it."""
        if self._driver is None:
            password = self._settings.password
            if isinstance(password, SecretStr):
                password_value = password.get_secret_value()
            else:
                password_value = password
            self._driver = GraphDatabase.driver(
                self._settings.uri,
                auth=(self._settings.user, password_value),
            )
            log.info("neo4j.connected", uri=self._settings.uri)
        return self._driver

    def session(self, *, write: bool = False) -> Session:
        """Open a Neo4j session from the shared driver."""
        access = WRITE_ACCESS if write else READ_ACCESS
        return self.connect().session(default_access_mode=access)

    def verify_connectivity(self) -> None:
        """Ping Neo4j, retrying with exponential backoff so callers can wait out startup."""
        delay = _VERIFY_INITIAL_DELAY_S
        last_error: BaseException | None = None
        for attempt in range(1, _VERIFY_ATTEMPTS + 1):
            try:
                self.connect().verify_connectivity()
                log.info("neo4j.verified", attempt=attempt)
                return
            except Exception as exc:
                last_error = exc
                log.warning(
                    "neo4j.verify_failed",
                    attempt=attempt,
                    attempts=_VERIFY_ATTEMPTS,
                    error=str(exc),
                    delay_s=delay,
                )
                if attempt == _VERIFY_ATTEMPTS:
                    break
                time.sleep(delay)
                delay = min(delay * 2, _VERIFY_MAX_DELAY_S)
        assert last_error is not None
        raise last_error

    def run_write_batch(self, query: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Run an UNWIND write, splitting `rows` into batches of WRITE_BATCH_SIZE."""
        _require_unwind(query)
        if not rows:
            return
        driver = self.connect()
        total = len(rows)
        with driver.session(default_access_mode=WRITE_ACCESS) as session:
            for start in range(0, total, WRITE_BATCH_SIZE):
                batch = list(rows[start : start + WRITE_BATCH_SIZE])
                log.info(
                    "neo4j.write_batch",
                    offset=start,
                    row_count=len(batch),
                    total=total,
                )
                session.execute_write(_run_unwind, query, batch)

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]:
        """Run Cypher in an explicit read-only transaction and return records as dicts."""
        parameters = dict(params or {})
        with self.connect().session(default_access_mode=READ_ACCESS) as session:
            tx = session.begin_transaction(timeout=timeout_s)
            try:
                result = tx.run(query, parameters)
                records = [record.data() for record in result]
                tx.commit()
                return records
            except Exception:
                tx.rollback()
                raise

    def read(self, query: str, parameters: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Run a read-only Cypher transaction and return records as dicts."""
        return self.run_read(query, parameters)

    def write_unwind(self, query: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Run a batched UNWIND write in a write transaction."""
        self.run_write_batch(query, rows)

    def close(self) -> None:
        """Close the driver if open."""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _run_unwind(
    tx: ManagedTransaction,
    query: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    unwind_write(tx, query, rows)
