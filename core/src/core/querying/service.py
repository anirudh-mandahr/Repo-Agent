"""Read-only graph query service. All user-facing Cypher goes through guard_readonly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from core.graph.client import GraphClient
from core.graph.schema import FIND_ENTITY_LABELS, NODE_LABELS, RELATIONSHIP_TYPES
from core.logging import get_logger
from core.querying.safety import guard_readonly, has_limit
from core.querying.templates import (
    DEFAULT_TRACE_DEPTH,
    FIND_ENTITY,
    FIND_RELATED,
    GET_DEPENDENCIES,
    GET_DEPENDENTS,
    TRACE_IMPORTS,
)

log = get_logger(__name__)

DEFAULT_RESULT_LIMIT = 200
READ_TIMEOUT_S = 10.0
Direction = Literal["outgoing", "incoming"]
STATISTICS_TIMEOUT_S = 10.0

LABEL_COUNTS_QUERY = "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS count"
RELATIONSHIP_COUNTS_QUERY = (
    "MATCH ()-[r]->() RETURN type(r) AS relationship_type, count(*) AS count"
)
INDEX_META_QUERY = (
    "MATCH (m:Meta {key: 'index_version'}) "
    "RETURN m.value AS index_version, m.updated_at AS last_indexed_at"
)


class QueryGraphClient(Protocol):
    """Read surface used by GraphQueryService. Tests may supply a fake."""

    def run_read(
        self,
        query: str,
        params: Mapping[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]: ...


class EntityHit(BaseModel):
    """A graph entity match."""

    entity_type: str
    name: str = ""
    qualified_name: str = ""
    file_path: str = ""
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None


class EntityQueryResult(BaseModel):
    """Result of ``find_entity``. Invalid ``entity_type`` sets ``error``."""

    matches: list[EntityHit] = Field(default_factory=list)
    result_count: int = 0
    truncated: bool = False
    error: str | None = None
    valid_types: list[str] = Field(default_factory=lambda: list(FIND_ENTITY_LABELS))


class NeighborHit(BaseModel):
    """A neighboring entity reached by IMPORTS, DEPENDS_ON, or CALLS."""

    entity_type: str
    name: str = ""
    qualified_name: str = ""
    relationship_type: str
    direction: Direction
    file_path: str = ""
    path: str | None = None


class NeighborQueryResult(BaseModel):
    """Outgoing or incoming IMPORTS / DEPENDS_ON / CALLS neighbors."""

    name: str
    neighbors: list[NeighborHit] = Field(default_factory=list)
    result_count: int = 0
    truncated: bool = False


class ImportTraceResult(BaseModel):
    """IMPORTS / DEPENDS_ON paths from a module, depth-capped."""

    module: str
    paths: list[list[str]] = Field(default_factory=list)
    depth: int = DEFAULT_TRACE_DEPTH
    result_count: int = 0
    truncated: bool = False


class RelatedHit(BaseModel):
    """A neighbor reached by a validated spec relationship type."""

    entity_type: str
    name: str = ""
    qualified_name: str = ""
    relationship_type: str
    direction: Direction
    file_path: str = ""
    path: str | None = None


class RelatedQueryResult(BaseModel):
    """Result of ``find_related``. Invalid relationship types set ``error``."""

    name: str
    relationship_type: str
    neighbors: list[RelatedHit] = Field(default_factory=list)
    result_count: int = 0
    truncated: bool = False
    error: str | None = None
    valid_types: list[str] = Field(default_factory=lambda: list(RELATIONSHIP_TYPES))


class QueryResult(BaseModel):
    """Raw read-only Cypher result."""
    cypher_executed: str
    params: dict[str, Any] = Field(default_factory=dict)

    rows: list[dict[str, Any]] = Field(default_factory=list)
    result_count: int = 0
    truncated: bool = False


class GraphStatistics(BaseModel):
    """Label counts, relationship counts, and current index metadata."""

    node_counts: dict[str, int] = Field(default_factory=dict)
    relationship_counts: dict[str, int] = Field(default_factory=dict)
    index_version: str | None = None
    last_indexed_at: str | None = None


class GraphQueryService:
    """One method per graph-query tool, plus guarded ``execute_query``."""

    def __init__(self, client: QueryGraphClient | None = None) -> None:
        self._client: QueryGraphClient = client if client is not None else GraphClient()

    def find_entity(self, name: str, entity_type: str | None = None) -> EntityQueryResult:
        """Match Module/Class/Function/Method by name, or File by path."""
        if entity_type is not None and entity_type not in FIND_ENTITY_LABELS:
            valid = list(FIND_ENTITY_LABELS)
            error = (
                f"Invalid entity_type {entity_type!r}. "
                f"Valid labels: {', '.join(valid)}"
            )
            log.warning("query.invalid_entity_type", entity_type=entity_type, valid=valid)
            return EntityQueryResult(
                error=error,
                valid_types=valid,
                result_count=0,
                truncated=False,
            )
        result = self.execute_query(
            FIND_ENTITY,
            {"name": name, "entity_type": entity_type},
        )
        matches = [_entity_hit(row) for row in result.rows]
        return EntityQueryResult(
            matches=matches,
            result_count=result.result_count,
            truncated=result.truncated,
        )

    def get_dependencies(self, name: str) -> NeighborQueryResult:
        """Return outgoing IMPORTS / DEPENDS_ON / CALLS neighbors."""
        result = self.execute_query(GET_DEPENDENCIES, {"name": name})
        neighbors = [_neighbor_hit(row) for row in result.rows]
        return NeighborQueryResult(
            name=name,
            neighbors=neighbors,
            result_count=result.result_count,
            truncated=result.truncated,
        )

    def get_dependents(self, name: str) -> NeighborQueryResult:
        """Return incoming IMPORTS / DEPENDS_ON / CALLS neighbors."""
        result = self.execute_query(GET_DEPENDENTS, {"name": name})
        neighbors = [_neighbor_hit(row) for row in result.rows]
        return NeighborQueryResult(
            name=name,
            neighbors=neighbors,
            result_count=result.result_count,
            truncated=result.truncated,
        )

    def trace_imports(self, module: str, depth: int = DEFAULT_TRACE_DEPTH) -> ImportTraceResult:
        """Follow IMPORTS / DEPENDS_ON chains from ``module``, capped at ``depth`` (max 5)."""
        capped = max(1, min(int(depth), DEFAULT_TRACE_DEPTH))
        result = self.execute_query(TRACE_IMPORTS, {"module": module, "depth": capped})
        paths = [_str_list(row.get("nodes")) for row in result.rows]
        return ImportTraceResult(
            module=module,
            paths=paths,
            depth=capped,
            result_count=result.result_count,
            truncated=result.truncated,
        )

    def find_related(self, name: str, relationship_type: str) -> RelatedQueryResult:
        """Return neighbors along ``relationship_type`` with direction."""
        if relationship_type not in RELATIONSHIP_TYPES:
            valid = list(RELATIONSHIP_TYPES)
            error = (
                f"Invalid relationship_type {relationship_type!r}. "
                f"Valid types: {', '.join(valid)}"
            )
            log.warning(
                "query.invalid_relationship_type",
                relationship_type=relationship_type,
                valid=valid,
            )
            return RelatedQueryResult(
                name=name,
                relationship_type=relationship_type,
                error=error,
                valid_types=valid,
                result_count=0,
                truncated=False,
            )
        result = self.execute_query(
            FIND_RELATED,
            {"name": name, "relationship_type": relationship_type},
        )
        neighbors = [_related_hit(row) for row in result.rows]
        return RelatedQueryResult(
            name=name,
            relationship_type=relationship_type,
            neighbors=neighbors,
            result_count=result.result_count,
            truncated=result.truncated,
        )

    def execute_query(
        self,
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        """Guard, cap with LIMIT 200, and run a read-only query (timeout 10s)."""
        guarded = guard_readonly(cypher)
        limited, applied_default_limit = _ensure_limit(guarded)
        parameters = dict(params or {})
        log.info(
            "query.execute",
            applied_default_limit=applied_default_limit,
            param_keys=sorted(parameters),
        )
        rows = [
            _jsonable_row(row)
            for row in self._client.run_read(
                limited,
                parameters,
                timeout_s=READ_TIMEOUT_S,
            )
        ]
        truncated = applied_default_limit and len(rows) >= DEFAULT_RESULT_LIMIT
        return QueryResult(
            cypher_executed=limited,
            params=parameters,
            rows=rows,
            result_count=len(rows),
            truncated=truncated,
        )

    def get_statistics(self) -> GraphStatistics:
        """Return graph label counts, relationship counts, and index metadata."""
        label_rows = self._client.run_read(LABEL_COUNTS_QUERY, timeout_s=STATISTICS_TIMEOUT_S)
        relationship_rows = self._client.run_read(
            RELATIONSHIP_COUNTS_QUERY,
            timeout_s=STATISTICS_TIMEOUT_S,
        )
        meta_rows = self._client.run_read(INDEX_META_QUERY, timeout_s=STATISTICS_TIMEOUT_S)
        node_counts = {label: 0 for label in NODE_LABELS}
        for row in label_rows:
            label = row.get("label")
            if isinstance(label, str):
                node_counts[label] = int(row.get("count") or 0)
        relationship_counts = {rel_type: 0 for rel_type in RELATIONSHIP_TYPES}
        for row in relationship_rows:
            rel_type = row.get("relationship_type")
            if isinstance(rel_type, str):
                relationship_counts[rel_type] = int(row.get("count") or 0)
        meta = meta_rows[0] if meta_rows else {}
        return GraphStatistics(
            node_counts=node_counts,
            relationship_counts=relationship_counts,
            index_version=_opt_str(meta.get("index_version")),
            last_indexed_at=_opt_str(meta.get("last_indexed_at")),
        )


def _ensure_limit(cypher: str) -> tuple[str, bool]:
    if has_limit(cypher):
        return cypher, False
    trimmed = cypher.rstrip()
    if trimmed.endswith(";"):
        trimmed = trimmed[:-1].rstrip()
    return f"{trimmed} LIMIT {DEFAULT_RESULT_LIMIT}", True


def _entity_hit(row: Mapping[str, Any]) -> EntityHit:
    labels = _str_list(row.get("labels"))
    qualified_name = str(row.get("qualified_name") or "")
    path = _opt_str(row.get("path"))
    name = str(row.get("name") or path or qualified_name)
    return EntityHit(
        entity_type=_primary_label(labels),
        name=name,
        qualified_name=qualified_name,
        file_path=str(row.get("file_path") or path or ""),
        path=path,
        line_start=_opt_int(row.get("line_start")),
        line_end=_opt_int(row.get("line_end")),
    )


def _neighbor_hit(row: Mapping[str, Any]) -> NeighborHit:
    labels = _str_list(row.get("labels"))
    qualified_name = str(row.get("qualified_name") or "")
    path = _opt_str(row.get("path"))
    module = _opt_str(row.get("module"))
    name = str(row.get("name") or module or path or qualified_name)
    direction = row.get("direction")
    resolved_direction: Direction = "incoming" if direction == "incoming" else "outgoing"
    return NeighborHit(
        entity_type=_primary_label(labels),
        name=name,
        qualified_name=qualified_name,
        relationship_type=str(row.get("relationship_type") or ""),
        direction=resolved_direction,
        file_path=str(row.get("file_path") or path or ""),
        path=path,
    )


def _related_hit(row: Mapping[str, Any]) -> RelatedHit:
    neighbor = _neighbor_hit(row)
    return RelatedHit(
        entity_type=neighbor.entity_type,
        name=neighbor.name,
        qualified_name=neighbor.qualified_name,
        relationship_type=neighbor.relationship_type,
        direction=neighbor.direction,
        file_path=neighbor.file_path,
        path=neighbor.path,
    )


def _primary_label(labels: Sequence[str]) -> str:
    for label in NODE_LABELS:
        if label in labels:
            return label
    return labels[0] if labels else ""


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, Sequence):
        return [str(item) for item in value if item is not None and str(item)]
    return [str(value)]


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _jsonable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _jsonable(val) for key, val in row.items()}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)


__all__ = [
    "DEFAULT_RESULT_LIMIT",
    "DEFAULT_TRACE_DEPTH",
    "READ_TIMEOUT_S",
    "EntityHit",
    "EntityQueryResult",
    "GraphQueryService",
    "GraphStatistics",
    "ImportTraceResult",
    "NeighborHit",
    "NeighborQueryResult",
    "QueryResult",
    "RelatedHit",
    "RelatedQueryResult",
]
