"""Read-only graph query service. All user-facing Cypher goes through guard_readonly."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel, Field

from core.graph.client import GraphClient
from core.graph.schema import (
    FIND_ENTITY_LABELS,
    FULLTEXT_INDEX_NAME,
    NODE_LABELS,
    RELATIONSHIP_TYPES,
    VECTOR_INDEX_NAMES,
)
from core.logging import get_logger
from core.querying.embeddings import EmbeddingProvider
from core.querying.safety import guard_readonly, has_limit
from core.querying.templates import (
    DEFAULT_NEIGHBOR_BRANCH_LIMIT,
    DEFAULT_TRACE_DEPTH,
    FIND_ENTITY,
    FIND_FULLTEXT,
    FIND_RELATED,
    GET_DEPENDENCIES,
    GET_DEPENDENCIES_COUNT,
    GET_DEPENDENTS,
    GET_DEPENDENTS_COUNT,
    TRACE_IMPORTS,
    VECTOR_SEARCH,
)
from core.settings import GraphQuerySettings

log = get_logger(__name__)

DEFAULT_RESULT_LIMIT = 200
DEFAULT_RETRIEVAL_TOP_K = 10
DEFAULT_EMBEDDING_MIN_SCORE = 0.6
READ_TIMEOUT_S = 10.0
Direction = Literal["outgoing", "incoming"]
RetrievalTier = Literal["exact", "fulltext", "lexical"]
STATISTICS_TIMEOUT_S = 10.0

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_PROPER_TOKEN_RE = re.compile(
    r"^(?:[A-Z][A-Za-z0-9_]+|[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z0-9_.]+)$"
)
_LUCENE_SPECIAL = re.compile(r'([+\-!(){}\[\]^"~*?:\\/]|&&|\|\|)')
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_LUCENE_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "codebase",
        "does",
        "example",
        "examples",
        "for",
        "from",
        "how",
        "i",
        "in",
        "implementation",
        "implemented",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "repository",
        "show",
        "snippet",
        "snippets",
        "source",
        "the",
        "this",
        "to",
        "we",
        "what",
        "where",
        "who",
        "why",
        "with",
        "work",
        "works",
        "you",
    }
)
_SENTENCE_STARTERS = frozenset(
    {"compare", "describe", "explain", "find", "list", "reindex", "show"}
)
_TIER_RANK = {"exact": 0, "fulltext": 1, "lexical": 2}
_SOURCE_PRIORITY_PACKAGE = 0
_SOURCE_PRIORITY_DOCS = 1
_SOURCE_PRIORITY_TESTS = 2


def source_priority(file_path: str | None) -> int:
    """Rank a repo path so package source outranks docs and tests.

    Args:
        file_path: Repo-relative path from a graph hit.

    Returns:
        Lower is better: package source, then ``docs_src/``, then ``tests/``.
    """
    normalized = (file_path or "").replace("\\", "/").lstrip("./")
    if normalized.startswith("tests/") or "/tests/" in normalized:
        return _SOURCE_PRIORITY_TESTS
    if normalized.startswith("docs_src/") or "/docs_src/" in normalized:
        return _SOURCE_PRIORITY_DOCS
    return _SOURCE_PRIORITY_PACKAGE


def retrieval_sort_key(
    tier: str | None,
    file_path: str | None,
    score: float,
) -> tuple[int, int, float]:
    """Sort key: retrieval tier, then source path, then descending score.

    Args:
        tier: Cascade tier name.
        file_path: Repo-relative path.
        score: Hit score (higher is better).

    Returns:
        Tuple suitable for ``list.sort``.
    """
    return (_TIER_RANK.get(tier or "", 9), source_priority(file_path), -float(score))


def proper_noun_tokens(text: str) -> list[str]:
    """Return identifier-like tokens treated as required names in Lucene queries.

    Args:
        text: User query or lookup phrase.

    Returns:
        Deduplicated proper-noun / dotted tokens, excluding sentence starters.
    """
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _TOKEN_RE.findall(text):
        if len(token) <= 1:
            continue
        lowered = token.lower()
        if lowered in _LUCENE_STOPWORDS or lowered in _SENTENCE_STARTERS:
            continue
        if not _PROPER_TOKEN_RE.fullmatch(token):
            continue
        if lowered in seen:
            continue
        seen.add(lowered)
        tokens.append(token)
    return tokens

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
    ) -> list[dict[str, Any]]:
        """Run a read-only Cypher query.

        Args:
            query: Cypher text.
            params: Query parameters.
            timeout_s: Per-query timeout in seconds.

        Returns:
            Result rows as dictionaries.
        """
        ...


class EntityHit(BaseModel):
    """A graph entity match, optionally tagged with the retrieval tier that produced it."""

    entity_type: str
    name: str = ""
    qualified_name: str = ""
    file_path: str = ""
    path: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    tier: RetrievalTier = "exact"
    score: float = 1.0


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
    hop: str | None = None


class NeighborQueryResult(BaseModel):
    """Outgoing or incoming IMPORTS / DEPENDS_ON / CALLS neighbors."""

    name: str
    neighbors: list[NeighborHit] = Field(default_factory=list)
    result_count: int = 0
    truncated: bool = False
    total_count: int = 0
    hops: list[str] = Field(default_factory=list)


class ImportTraceResult(BaseModel):
    """IMPORTS / DEPENDS_ON paths from a module, depth-capped."""

    module: str
    paths: list[list[str]] = Field(default_factory=list)
    depth: int = DEFAULT_TRACE_DEPTH
    result_count: int = 0
    truncated: bool = False
    hop: str | None = None


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

    def __init__(
        self,
        client: QueryGraphClient | None = None,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        embeddings_enabled: bool | None = None,
    ) -> None:
        """Create the query service.

        Args:
            client: Read-only graph client. A live ``GraphClient`` is used when omitted.
            embedding_provider: Optional vector backend for the hash-based
                lexical fallback. Injected even when the flag is off so the
                seam stays wired; the flag decides whether it runs.
            embeddings_enabled: Override for ``GQ_EMBEDDINGS_ENABLED``. When
                omitted, the settings flag is authoritative (default off).
                Passing a provider does not enable the tier by itself.
        """
        self._client: QueryGraphClient = client if client is not None else GraphClient()
        if embeddings_enabled is None:
            embeddings_enabled = GraphQuerySettings.from_env().embeddings_enabled
        self._embeddings_enabled = embeddings_enabled
        self._embedding_provider = embedding_provider

    def find_entity(self, name: str, entity_type: str | None = None) -> EntityQueryResult:
        """Match by exact name, then full-text, then optional lexical hash search.

        Args:
            name: str.
            entity_type: str | None.

        Returns:
            EntityQueryResult.
        """
        return self.retrieve(name, entity_type=entity_type)

    def retrieve(
        self,
        query: str,
        entity_type: str | None = None,
        *,
        top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    ) -> EntityQueryResult:
        """Cascade: exact name, full-text, optional hash-based lexical fallback.

        Args:
            query: str.
            entity_type: str | None.
            top_k: int.

        Returns:
            EntityQueryResult.
        """
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

        seen: set[str] = set()
        ranked: list[EntityHit] = []
        truncated = False

        exact = self.execute_query(
            FIND_ENTITY,
            {"name": query, "entity_type": entity_type},
        )
        truncated = truncated or exact.truncated
        _merge_hits(ranked, seen, exact.rows, tier="exact")

        lucene = lucene_query(query)
        if lucene:
            try:
                fulltext = self.execute_query(
                    FIND_FULLTEXT,
                    {
                        "index_name": FULLTEXT_INDEX_NAME,
                        "lucene_query": lucene,
                        "entity_type": entity_type,
                        "top_k": max(1, int(top_k)),
                    },
                )
                truncated = truncated or fulltext.truncated or fulltext.result_count >= top_k
                _merge_hits(ranked, seen, fulltext.rows, tier="fulltext")
            except Exception as exc:
                log.warning("query.fulltext_failed", error=str(exc))

        if self._embeddings_enabled and self._embedding_provider is not None:
            try:
                embedded = self._embedding_search(query, entity_type=entity_type, top_k=top_k)
                _merge_hits(
                    ranked,
                    seen,
                    [hit.model_dump() for hit in embedded],
                    tier="lexical",
                )
            except Exception as exc:
                log.warning("query.lexical_failed", error=str(exc))

        ranked.sort(key=lambda hit: retrieval_sort_key(hit.tier, hit.file_path, hit.score))
        log.info(
            "query.retrieve",
            query=query,
            result_count=len(ranked),
            tiers=[hit.tier for hit in ranked[:top_k]],
        )
        return EntityQueryResult(
            matches=ranked,
            result_count=len(ranked),
            truncated=truncated,
        )

    def get_dependencies(self, name: str) -> NeighborQueryResult:
        """Return outgoing IMPORTS / DEPENDS_ON / CALLS neighbors.
        
        Args:
            name: str.

        Returns:
            NeighborQueryResult.
        """
        result = self.execute_query(
            GET_DEPENDENCIES,
            {"name": name, "branch_limit": DEFAULT_NEIGHBOR_BRANCH_LIMIT},
        )
        neighbors = [_neighbor_hit(row) for row in result.rows]
        hops = _unique_hops(neighbors)
        total_count = _neighbor_total(
            self.execute_query(GET_DEPENDENCIES_COUNT, {"name": name}),
            fallback=len(neighbors),
        )
        truncated = total_count > len(neighbors) or result.truncated
        return NeighborQueryResult(
            name=name,
            neighbors=neighbors,
            result_count=len(neighbors),
            truncated=truncated,
            total_count=total_count,
            hops=hops,
        )

    def get_dependents(self, name: str) -> NeighborQueryResult:
        """Return incoming IMPORTS / DEPENDS_ON / CALLS neighbors.
        
        Args:
            name: str.

        Returns:
            NeighborQueryResult.
        """
        result = self.execute_query(
            GET_DEPENDENTS,
            {"name": name, "branch_limit": DEFAULT_NEIGHBOR_BRANCH_LIMIT},
        )
        neighbors = [_neighbor_hit(row) for row in result.rows]
        hops = _unique_hops(neighbors)
        total_count = _neighbor_total(
            self.execute_query(GET_DEPENDENTS_COUNT, {"name": name}),
            fallback=len(neighbors),
        )
        truncated = total_count > len(neighbors) or result.truncated
        return NeighborQueryResult(
            name=name,
            neighbors=neighbors,
            result_count=len(neighbors),
            truncated=truncated,
            total_count=total_count,
            hops=hops,
        )

    def trace_imports(self, module: str, depth: int = DEFAULT_TRACE_DEPTH) -> ImportTraceResult:
        """Follow IMPORTS / DEPENDS_ON chains from ``module``, capped at ``depth`` (max 5).
        
        Args:
            module: str.
            depth: int.

        Returns:
            ImportTraceResult.
        """
        capped = max(1, min(int(depth), DEFAULT_TRACE_DEPTH))
        result = self.execute_query(TRACE_IMPORTS, {"module": module, "depth": capped})
        paths = [_str_list(row.get("nodes")) for row in result.rows]
        hop = _opt_str(result.rows[0].get("hop")) if result.rows else None
        return ImportTraceResult(
            module=module,
            paths=paths,
            depth=capped,
            result_count=result.result_count,
            truncated=result.truncated,
            hop=hop,
        )

    def find_related(self, name: str, relationship_type: str) -> RelatedQueryResult:
        """Return neighbors along ``relationship_type`` with direction.
        
        Args:
            name: str.
            relationship_type: str.

        Returns:
            RelatedQueryResult.
        """
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

    def _embedding_search(
        self,
        query: str,
        *,
        entity_type: str | None,
        top_k: int,
    ) -> list[EntityHit]:
        """Score stored hash vectors via the Neo4j vector index (lexical overlap)."""
        provider = self._embedding_provider
        if provider is None:
            return []
        query_vectors = list(provider.embed([query]))
        if not query_vectors or not any(query_vectors[0]):
            return []
        query_vec = [float(value) for value in query_vectors[0]]
        labels: list[str]
        if entity_type in VECTOR_INDEX_NAMES:
            labels = [entity_type]
        else:
            labels = list(VECTOR_INDEX_NAMES)
        hits: list[EntityHit] = []
        seen: set[str] = set()
        for label in labels:
            index_name = VECTOR_INDEX_NAMES[label]
            try:
                result = self.execute_query(
                    VECTOR_SEARCH,
                    {
                        "index_name": index_name,
                        "top_k": max(1, int(top_k)),
                        "query_vector": query_vec,
                        "entity_type": entity_type,
                        "min_score": DEFAULT_EMBEDDING_MIN_SCORE,
                    },
                )
            except Exception as exc:
                log.warning(
                    "query.vector_search_failed",
                    index_name=index_name,
                    error=str(exc),
                )
                continue
            for row in result.rows:
                hit = _entity_hit(row, tier="lexical")
                key = _hit_key(hit)
                if not key or key in seen:
                    continue
                seen.add(key)
                hits.append(hit)
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[: max(1, int(top_k))]

    def execute_query(
        self,
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> QueryResult:
        """Guard, cap with LIMIT 200, and run a read-only query (timeout 10s).
        
        Args:
            cypher: str.
            params: Mapping[str, Any] | None.

        Returns:
            QueryResult.
        """
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
        """Return graph label counts, relationship counts, and index metadata.
        
        Returns:
            GraphStatistics.
        """
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


def _entity_hit(
    row: Mapping[str, Any],
    *,
    tier: RetrievalTier = "exact",
    score: float | None = None,
) -> EntityHit:
    labels = _str_list(row.get("labels"))
    qualified_name = str(row.get("qualified_name") or "")
    path = _opt_str(row.get("path"))
    name = str(row.get("name") or path or qualified_name)
    resolved_score = score
    if resolved_score is None:
        raw_score = row.get("score")
        resolved_score = float(raw_score) if raw_score is not None and raw_score != "" else 1.0
    raw_tier = row.get("tier")
    resolved_tier: RetrievalTier = tier
    if isinstance(raw_tier, str) and raw_tier in _TIER_RANK:
        resolved_tier = cast(RetrievalTier, raw_tier)
    return EntityHit(
        entity_type=_primary_label(labels) or str(row.get("entity_type") or ""),
        name=name,
        qualified_name=qualified_name,
        file_path=str(row.get("file_path") or path or ""),
        path=path,
        line_start=_opt_int(row.get("line_start")),
        line_end=_opt_int(row.get("line_end")),
        tier=resolved_tier,
        score=float(resolved_score),
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
        hop=_opt_str(row.get("hop")),
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


def _unique_hops(neighbors: Sequence[NeighborHit]) -> list[str]:
    seen: set[str] = set()
    hops: list[str] = []
    for neighbor in neighbors:
        hop = neighbor.hop
        if not hop or hop in seen:
            continue
        seen.add(hop)
        hops.append(hop)
    return hops


def _neighbor_total(result: QueryResult, *, fallback: int) -> int:
    if not result.rows:
        return fallback
    raw = result.rows[0].get("total")
    if raw is None or raw == "":
        return fallback
    return max(fallback, int(raw))


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


def _merge_hits(
    ranked: list[EntityHit],
    seen: set[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    tier: RetrievalTier,
) -> None:
    for row in rows:
        hit = _entity_hit(row, tier=tier)
        key = _hit_key(hit)
        if not key or key in seen:
            continue
        seen.add(key)
        ranked.append(hit)


def _hit_key(hit: EntityHit) -> str:
    return hit.qualified_name or hit.file_path or hit.path or hit.name


def lucene_query(text: str) -> str:
    """Build a Lucene string: names-only for identifiers, should/must for NL questions.

    Proper-noun and dotted tokens are required (AND). Remaining content tokens
    are optional and OR-combined so Neo4j can rank by relevance instead of
    demanding every word appear on the same node.
    
    Args:
        text: str.

    Returns:
        str.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    if _IDENTIFIER_RE.fullmatch(stripped):
        escaped = _escape_lucene(stripped)
        quoted = f'"{escaped}"'
        return f"name:{quoted} OR qualified_name:{quoted}"
    tokens = [_escape_lucene(token) for token in _TOKEN_RE.findall(stripped) if len(token) > 1]
    unique: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        unique.append(token)
    if not unique:
        return ""
    content = [token for token in unique if token.lower() not in _LUCENE_STOPWORDS]
    if not content:
        return ""
    must = [
        token
        for token in content
        if _PROPER_TOKEN_RE.fullmatch(token) and token.lower() not in _SENTENCE_STARTERS
    ]
    should = [token for token in content if token not in must]
    clauses: list[str] = []
    if must:
        required = [_proper_noun_clause(token) for token in must]
        clauses.append("(" + " AND ".join(required) + ")" if len(required) > 1 else required[0])
    if should:
        clauses.append("(" + " OR ".join(should) + ")" if len(should) > 1 else should[0])
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return " AND ".join(clauses)


def _proper_noun_clause(token: str) -> str:
    quoted = f'"{token}"'
    return f"(name:{quoted} OR qualified_name:{quoted} OR {token})"


def _escape_lucene(token: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", token)


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
    "DEFAULT_EMBEDDING_MIN_SCORE",
    "DEFAULT_NEIGHBOR_BRANCH_LIMIT",
    "DEFAULT_RESULT_LIMIT",
    "DEFAULT_RETRIEVAL_TOP_K",
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
    "RetrievalTier",
    "lucene_query",
    "proper_noun_tokens",
    "retrieval_sort_key",
    "source_priority",
]
