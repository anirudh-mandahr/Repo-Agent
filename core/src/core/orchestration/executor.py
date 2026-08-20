"""Execute an :class:`~core.orchestration.models.ExecutionPlan`."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from core.exceptions import AgentUnavailableError
from core.graph.schema import REL_CALLS, REL_DEPENDS_ON, REL_IMPORTS, REL_INHERITS_FROM
from core.logging import get_logger
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger, record_payload_usage
from core.querying.patterns import SUPPORTED_PATTERNS, pattern_path_prefix
from core.querying.service import retrieval_sort_key
from core.querying.templates import DEFAULT_TRACE_DEPTH
from core.resilience.retry import await_with_timeout_retry
from core.settings import OrchestratorSettings

from .budget import RequestBudget
from .models import AgentName, AgentOutput, ExecutionPlan, QueryIntent, QueryIntentIntent
from .router import retrieval_search_terms
from .scope import is_out_of_scope

log = get_logger(__name__)

DEFAULT_ANALYSIS_CANDIDATES = 3
_REFINEMENT_CANDIDATE_CAP = 2


class _BudgetSkip(Exception):
    """Internal: stop issuing further specialist calls for this agent."""

# Specialist tools to invoke per intent. `first` runs (gathered) before `parallel`.
INTENT_TOOL_MAP: dict[QueryIntentIntent, dict[AgentName, dict[str, tuple[str, ...]]]] = {
    "lookup": {
        "graph_query": {"first": ("find_entity",), "parallel": ()},
        "code_analyst": {"first": (), "parallel": ("get_code_snippet", "analyze_class")},
    },
    "relationship": {
        "graph_query": {
            "first": ("find_entity",),
            "parallel": (
                "get_dependencies",
                "get_dependents",
                "trace_imports",
                "find_related",
            ),
        },
    },
    "explanation": {
        "graph_query": {"first": ("find_entity",), "parallel": ()},
        "code_analyst": {
            "first": (),
            "parallel": (
                "explain_implementation",
                "analyze_function",
                "analyze_class",
                "get_code_snippet",
            ),
        },
    },
    "comparison": {
        "graph_query": {"first": ("find_entity",), "parallel": ()},
        "code_analyst": {"first": (), "parallel": ("compare_implementations",)},
    },
    "pattern": {
        "code_analyst": {"first": (), "parallel": ("find_patterns",)},
    },
    "indexing": {
        "indexer": {"first": (), "parallel": ("index_repository",)},
    },
    "mixed": {
        "graph_query": {
            "first": ("find_entity",),
            "parallel": (
                "get_dependencies",
                "get_dependents",
                "trace_imports",
                "find_related",
            ),
        },
        "code_analyst": {
            "first": (),
            "parallel": (
                "get_code_snippet",
                "explain_implementation",
                "analyze_function",
                "analyze_class",
            ),
        },
    },
}


class GraphQueryClient(Protocol):
    """Graph Query agent client used by the executor."""

    async def get_statistics(self) -> Any:
        """Return graph statistics including ``index_version``.

        Returns:
            Statistics payload or model.
        """
        ...

    async def find_entity(self, name: str, entity_type: str | None = None) -> Any:
        """Look up an entity by name.

        Args:
            name: Entity name or qualified name.
            entity_type: Optional label filter.

        Returns:
            Match payload or ``None``.
        """
        ...

    async def get_dependencies(self, name: str) -> Any:
        """Return outgoing neighbors for ``name``.

        Args:
            name: Entity name.

        Returns:
            Neighbor payload.
        """
        ...

    async def get_dependents(self, name: str) -> Any:
        """Return incoming neighbors for ``name``.

        Args:
            name: Entity name.

        Returns:
            Neighbor payload.
        """
        ...

    async def find_related(self, name: str, relationship_type: str) -> Any:
        """Return neighbors along ``relationship_type``.

        Args:
            name: Entity name.
            relationship_type: Typed graph relationship.

        Returns:
            Related-entity payload.
        """
        ...

    async def trace_imports(self, module: str, depth: int = DEFAULT_TRACE_DEPTH) -> Any:
        """Trace import/depends-on paths from ``module``.

        Args:
            module: Module name.
            depth: Maximum hop count.

        Returns:
            Import-trace payload.
        """
        ...


class CodeAnalystClient(Protocol):
    """Code Analyst agent client used by the executor."""

    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> Any:
        """Fetch a numbered source snippet.

        Args:
            qualified_name: Optional graph coordinate.
            file_path: Optional repo-relative path.
            line_start: Inclusive start line.
            line_end: Inclusive end line.

        Returns:
            Snippet payload.
        """
        ...

    async def explain_implementation(self, qualified_name: str) -> Any:
        """Explain a function or class implementation.

        Args:
            qualified_name: Graph qualified name.

        Returns:
            Explanation payload.
        """
        ...

    async def analyze_function(self, qualified_name: str) -> Any:
        """Return structured analysis for a function or method.

        Args:
            qualified_name: Graph qualified name.

        Returns:
            Analysis payload.
        """
        ...

    async def analyze_class(self, qualified_name: str) -> Any:
        """Return structured analysis for a class.

        Args:
            qualified_name: Graph qualified name.

        Returns:
            Analysis payload.
        """
        ...

    async def compare_implementations(self, name_a: str, name_b: str) -> Any:
        """Compare two implementations.

        Args:
            name_a: First qualified name.
            name_b: Second qualified name.

        Returns:
            Comparison payload.
        """
        ...

    async def find_patterns(
        self,
        pattern: str,
        path_prefix: str | None = None,
    ) -> Any:
        """Find instances of a named structural pattern.

        Args:
            pattern: One of the supported pattern names.
            path_prefix: Optional module or file-path prefix for decorator scoping.

        Returns:
            Pattern payload.
        """
        ...


class IndexerClient(Protocol):
    """Indexer agent client used by the executor."""

    async def index_repository(self, repo_url: str | None = None) -> Any:
        """Trigger a repository index.

        Args:
            repo_url: Optional clone URL override.

        Returns:
            Index report payload.
        """
        ...


class AgentClients(Protocol):
    """Specialist clients required to execute an orchestration plan."""

    graph_query: GraphQueryClient
    code_analyst: CodeAnalystClient
    indexer: IndexerClient


def _default_snippet_range() -> tuple[int, int]:
    # Keep this deterministic and small for degraded mode.
    return (1, 120)


def code_analyst_can_start_now(intent: QueryIntent, query: str = "") -> bool:
    """True when the analyst can run without waiting for graph_query.

    Pattern search needs no graph coordinates. Every other analyst tool
    requires a resolved qualified name, so the executor waits for
    ``graph_done`` rather than invoking tools with a bare entity name.

    Args:
        intent: Routed query intent.
        query: User query used to specialize mixed/comparison tool plans.

    Returns:
        Whether code_analyst work can overlap graph_query.
    """
    tools = _tool_plan(intent.intent, "code_analyst", query)
    parallel = tools.get("parallel", ())
    if not parallel:
        return False
    entity_tools = {
        "get_code_snippet",
        "explain_implementation",
        "analyze_function",
        "analyze_class",
        "compare_implementations",
    }
    if entity_tools.intersection(parallel):
        return False
    return "find_patterns" in parallel


def _jsonish(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
    elif isinstance(value, Mapping):
        dumped = dict(value)
    else:
        return value
    if isinstance(dumped, dict):
        dumped.pop("usage", None)
    return dumped


def _entity_hits(result: Any) -> list[dict[str, Any]]:
    payload = _jsonish(result)
    if not isinstance(payload, Mapping):
        return []
    matches = payload.get("matches")
    if isinstance(matches, list) and matches:
        hits: list[dict[str, Any]] = []
        for match in matches:
            item = _jsonish(match)
            if isinstance(item, Mapping):
                hits.append(dict(item))
        return hits
    if any(payload.get(key) for key in ("file_path", "filePath", "qualified_name", "name")):
        return [dict(payload)]
    return []


def _primary_hit(result: Any) -> dict[str, Any]:
    hits = _entity_hits(result)
    return hits[0] if hits else {}


def _hit_identity(hit: Mapping[str, Any]) -> str:
    return str(
        hit.get("qualified_name")
        or hit.get("file_path")
        or hit.get("filePath")
        or hit.get("name")
        or hit.get("path")
        or ""
    )


def _candidate_limit(iteration: int) -> int:
    """Tighten analysis fan-out after the first retrieval round."""
    if iteration > 1:
        return min(_REFINEMENT_CANDIDATE_CAP, DEFAULT_ANALYSIS_CANDIDATES)
    return DEFAULT_ANALYSIS_CANDIDATES


def _select_analysis_candidates(
    hits: Sequence[Mapping[str, Any]],
    *,
    limit: int = DEFAULT_ANALYSIS_CANDIDATES,
    prefer_exact: bool = True,
) -> list[dict[str, Any]]:
    """Prefer exact-tier hits, then package source over docs and tests."""
    exact = [dict(hit) for hit in hits if str(hit.get("tier") or "") == "exact"]
    pool: list[dict[str, Any]] = (
        exact if (prefer_exact and exact) else [dict(hit) for hit in hits]
    )
    pool.sort(
        key=lambda hit: retrieval_sort_key(
            str(hit.get("tier") or ""),
            str(hit.get("file_path") or hit.get("filePath") or ""),
            float(hit.get("score") or 0.0),
            name=str(hit.get("name") or ""),
            qualified_name=str(hit.get("qualified_name") or ""),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in pool:
        key = _hit_identity(hit)
        if not key or key in seen:
            continue
        seen.add(key)
        selected.append(dict(hit))
        if len(selected) >= limit:
            break
    return selected


def _resolved_name(entity: str, hit: Mapping[str, Any]) -> str:
    return str(hit.get("qualified_name") or hit.get("name") or entity)


def _hit_is_class(hit: Mapping[str, Any]) -> bool:
    entity_type = str(hit.get("entity_type") or hit.get("type") or "")
    if entity_type == "Class":
        return True
    labels = hit.get("labels")
    return isinstance(labels, list) and any(str(label) == "Class" for label in labels)


def _pattern_name(query: str, entities: Sequence[str]) -> str:
    lowered = query.lower().replace("-", "_")
    for name in SUPPORTED_PATTERNS:
        needle = name.replace("_", " ")
        if name in lowered or needle in lowered:
            return name
    for entity in entities:
        candidate = entity.lower().replace("-", "_").replace(" ", "_")
        if candidate in SUPPORTED_PATTERNS:
            return candidate
    return "decorator"


def _tool_plan(
    intent: QueryIntentIntent,
    agent: AgentName,
    query: str = "",
) -> dict[str, tuple[str, ...]]:
    if _wants_comparison(query) and intent in {"comparison", "mixed"}:
        if agent == "graph_query":
            return {
                "first": ("find_entity",),
                "parallel": _relationship_tools_for_query(query),
            }
        if agent == "code_analyst":
            return {"first": (), "parallel": ("compare_implementations",)}
    return INTENT_TOOL_MAP.get(intent, {}).get(agent, {"first": (), "parallel": ()})


def _wants_comparison(query: str) -> bool:
    return "compare" in query.lower()


def _relationship_tools_for_query(query: str) -> tuple[str, ...]:
    """Relationship tools implied by the query, not the full mixed fan-out."""
    lowered = query.lower()
    tools: list[str] = []
    if re.search(r"who depends|\bdependents?\b|\bdepends on them\b", lowered):
        tools.append("get_dependents")
    elif re.search(r"\bdepends on\b|\buses\b", lowered):
        tools.append("get_dependencies")
    if re.search(r"\bimports?\b", lowered):
        tools.append("trace_imports")
    if re.search(r"inherit|extends|subclass|who calls|\bcalled by\b", lowered):
        tools.append("find_related")
    return tuple(tools)


def related_relationship_type(query: str) -> str | None:
    """Map relationship wording onto a typed graph edge, if one is implied.
    
    Args:
        query: str.

    Returns:
        str | None.
    """
    lowered = query.lower()
    if re.search(r"inherit|extends|subclass", lowered):
        return REL_INHERITS_FROM
    if re.search(r"who calls|\bcalled by\b", lowered):
        return REL_CALLS
    if re.search(r"\bimports?\b", lowered):
        return REL_IMPORTS
    if re.search(r"\bdepends on\b|\bdependents?\b|\buses\b", lowered):
        return REL_DEPENDS_ON
    return None


async def run_plan(
    plan: ExecutionPlan,
    *,
    query: str,
    context: ConversationContext | None,
    clients: AgentClients,
    settings: OrchestratorSettings,
    correlation_id: str,
    budget: RequestBudget | None = None,
    token_ledger: TokenLedger | None = None,
) -> Mapping[AgentName, AgentOutput]:
    """Execute plan while degrading gracefully.
    
    The executor never raises; it returns best-effort `AgentOutput` objects.
    When ``budget`` is exhausted, in-flight calls finish but no new specialist
    calls are issued.
    
    Args:
        plan: ExecutionPlan.
        query: str.
        context: ConversationContext | None.
        clients: AgentClients.
        settings: OrchestratorSettings.
        correlation_id: str.
        budget: Optional outer request budget.
        token_ledger: Ledger used to read spend for budget checks.

    Returns:
        Mapping[AgentName, AgentOutput].
    """
    _ = context  # entities are resolved before routing; plan.intent carries them

    graph_entities: list[dict[str, Any]] = []
    graph_hits: list[dict[str, Any]] = []
    analysis_candidates: list[dict[str, Any]] = []
    graph_available = True
    code_available = True
    tools_invoked: list[str] = []
    graph_extra: dict[str, Any] = {}
    planned_agents = set(plan.agents)
    graph_done = asyncio.Event()
    graph_coords_ready = asyncio.Event()
    if "graph_query" not in planned_agents:
        graph_done.set()
        graph_coords_ready.set()

    agent_status: dict[AgentName, AgentOutput] = {
        agent: AgentOutput(agent=agent, ok=True) for agent in plan.agents
    }

    def _record(agent: AgentName, tool: str) -> None:
        tools_invoked.append(f"{agent}.{tool}")

    def _bind(
        method: Callable[..., Awaitable[Any]],
        /,
        *args: Any,
        **kwargs: Any,
    ) -> Callable[[], Awaitable[Any]]:
        async def _run() -> Any:
            return await method(*args, **kwargs)

        return _run

    async def _call(
        agent: AgentName,
        tool: str,
        factory: Callable[[], Awaitable[Any]],
        timeout_s: float,
    ) -> Any:
        if budget is not None and not budget.allow_new_call(token_ledger, correlation_id):
            raise _BudgetSkip()
        _record(agent, tool)
        remaining = budget.specialist_remaining_s() if budget is not None else timeout_s
        # Clip to one per-agent attempt. Multiplying by retry_count let the
        # analyst hold the plan for 30s and starve the synthesis reserve.
        timeout = min(timeout_s, remaining) if budget is not None else timeout_s
        if timeout <= 0:
            raise _BudgetSkip()
        # MCP streamable-HTTP reads can ignore cancellation (anyio cancel
        # scopes), and asyncio.wait_for waits for the cancelled task to
        # finish -- a dead specialist would hang the whole request past the
        # gateway ceiling. This helper abandons a stuck task instead.
        result = await await_with_timeout_retry(
            factory,
            timeout_s=timeout,
            retry_count=0,
        )
        record_payload_usage(token_ledger, correlation_id, result)
        return result

    async def _run_graph(intent: QueryIntent) -> None:
        nonlocal graph_entities, graph_available, graph_hits, graph_extra, analysis_candidates
        try:
            entities = list(intent.entities or [])
            tools = _tool_plan(intent.intent, "graph_query", query)
            if not tools.get("first") and not tools.get("parallel"):
                log.info(
                    "orchestrator.empty_tool_plan",
                    agent="graph_query",
                    intent=intent.intent,
                    correlation_id=correlation_id,
                )
                agent_status["graph_query"] = AgentOutput(
                    agent="graph_query",
                    ok=False,
                    degraded_note="no tool plan for intent; specialist was not invoked",
                    error="empty_tool_plan",
                )
                return
            search_terms = (
                list(plan.search_terms)
                if plan.search_terms
                else retrieval_search_terms(query, entities)
            )
            candidate_limit = _candidate_limit(plan.iteration)
            if "find_entity" in tools.get("first", ()) and not search_terms:
                agent_status["graph_query"] = AgentOutput(
                    agent="graph_query",
                    ok=True,
                    output={
                        "entities": [],
                        "queried_entities": [],
                        "candidates": [],
                        "tools_invoked": [],
                    },
                    tools_invoked=[],
                )
                return

            find_results: list[Any] = []
            candidates: list[dict[str, Any]] = []
            if "find_entity" in tools.get("first", ()) and search_terms:
                find_results = await asyncio.gather(
                    *[
                        _call(
                            "graph_query",
                            "find_entity",
                            _bind(clients.graph_query.find_entity, name=entity),
                            settings.graph_query_timeout_s,
                        )
                        for entity in search_terms
                    ]
                )
                graph_entities = [_jsonish(result) for result in find_results if result is not None]
                graph_hits = [_primary_hit(result) for result in find_results]
                per_term: list[dict[str, Any]] = []
                for result in find_results:
                    hits = _entity_hits(result)
                    candidates.extend(hits)
                    per_term.extend(
                        _select_analysis_candidates(hits, limit=candidate_limit)
                    )
                analysis_candidates = _select_analysis_candidates(
                    per_term, prefer_exact=False, limit=candidate_limit
                )
                if plan.iteration > 1:
                    log.info(
                        "orchestrator.refinement_candidate_cap",
                        limit=candidate_limit,
                        selected=len(analysis_candidates),
                        iteration=plan.iteration,
                        correlation_id=correlation_id,
                    )

            if is_out_of_scope(query):
                analysis_candidates = []
                graph_hits = [{} for _ in graph_hits]
            graph_coords_ready.set()

            parallel = tools.get("parallel", ())
            expand_relationships = bool(parallel and search_terms) and not is_out_of_scope(
                query
            )
            if expand_relationships:
                rel_tasks: list[asyncio.Task[Any]] = []
                rel_keys: list[tuple[str, str]] = []
                rel_names: list[str] = []
                if analysis_candidates:
                    for hit in analysis_candidates:
                        name = _resolved_name("", hit)
                        if name:
                            rel_names.append(name)
                else:
                    for index, entity in enumerate(search_terms):
                        hit = graph_hits[index] if index < len(graph_hits) else {}
                        if not _hit_identity(hit):
                            continue
                        rel_names.append(_resolved_name(entity, hit))
                seen_rel: set[str] = set()
                for name in rel_names:
                    if not name or name in seen_rel:
                        continue
                    seen_rel.add(name)
                    if "get_dependencies" in parallel:
                        rel_keys.append(("dependencies", name))
                        rel_tasks.append(
                            asyncio.create_task(
                                _call(
                                    "graph_query",
                                    "get_dependencies",
                                    _bind(clients.graph_query.get_dependencies, name),
                                    settings.graph_query_timeout_s,
                                )
                            )
                        )
                    if "get_dependents" in parallel:
                        rel_keys.append(("dependents", name))
                        rel_tasks.append(
                            asyncio.create_task(
                                _call(
                                    "graph_query",
                                    "get_dependents",
                                    _bind(clients.graph_query.get_dependents, name),
                                    settings.graph_query_timeout_s,
                                )
                            )
                        )
                    if "trace_imports" in parallel:
                        rel_keys.append(("import_traces", name))
                        rel_tasks.append(
                            asyncio.create_task(
                                _call(
                                    "graph_query",
                                    "trace_imports",
                                    _bind(clients.graph_query.trace_imports, name),
                                    settings.graph_query_timeout_s,
                                )
                            )
                        )
                    if "find_related" in parallel:
                        relationship_type = related_relationship_type(query)
                        if relationship_type:
                            rel_keys.append(("related", name))
                            rel_tasks.append(
                                asyncio.create_task(
                                    _call(
                                        "graph_query",
                                        "find_related",
                                        _bind(
                                            clients.graph_query.find_related,
                                            name,
                                            relationship_type,
                                        ),
                                        settings.graph_query_timeout_s,
                                    )
                                )
                            )
                if rel_tasks:
                    rel_results = await asyncio.gather(*rel_tasks, return_exceptions=True)
                    grouped: dict[str, list[Any]] = {
                        "dependencies": [],
                        "dependents": [],
                        "import_traces": [],
                        "related": [],
                    }
                    for (bucket, _name), result in zip(rel_keys, rel_results, strict=True):
                        if isinstance(result, BaseException):
                            if not isinstance(result, Exception):
                                raise result
                            continue
                        grouped[bucket].append(_jsonish(result))
                    graph_extra = grouped

            agent_status["graph_query"] = AgentOutput(
                agent="graph_query",
                ok=True,
                output={
                    "entities": graph_entities,
                    "queried_entities": search_terms,
                    "candidates": candidates,
                    "analysis_candidates": analysis_candidates,
                    **graph_extra,
                },
            )
        except _BudgetSkip:
            existing = agent_status.get("graph_query")
            if graph_entities or (existing is not None and existing.output):
                payload = (
                    dict(existing.output)
                    if existing is not None and isinstance(existing.output, dict)
                    else {}
                )
                agent_status["graph_query"] = AgentOutput(
                    agent="graph_query",
                    ok=True,
                    output={
                        "entities": graph_entities,
                        "queried_entities": payload.get(
                            "queried_entities", list(intent.entities or [])
                        ),
                        "candidates": payload.get("candidates", []),
                        "analysis_candidates": analysis_candidates,
                        **graph_extra,
                    },
                )
            else:
                log.info(
                    "orchestrator.budget_skip",
                    agent="graph_query",
                    correlation_id=correlation_id,
                )
                agent_status["graph_query"] = AgentOutput(
                    agent="graph_query",
                    ok=False,
                    degraded_note="budget reserved remaining time for synthesis",
                    error="budget_skip",
                )
        except (TimeoutError, AgentUnavailableError):
            graph_available = False
            agent_status["graph_query"] = AgentOutput(
                agent="graph_query",
                ok=False,
                degraded_note="graph_query unavailable/timeout; using raw-file fallback",
                error="graph_query timeout",
            )
        except Exception as exc:  # pragma: no cover - defensive
            graph_available = False
            agent_status["graph_query"] = AgentOutput(
                agent="graph_query",
                ok=False,
                degraded_note="graph_query unavailable; using raw-file fallback",
                error=str(exc),
            )

    async def _run_code(intent: QueryIntent) -> None:
        nonlocal code_available
        output: dict[str, Any] = {}
        try:
            entities = list(intent.entities or [])
            candidate_hits = list(analysis_candidates)
            if is_out_of_scope(query):
                candidate_hits = []
            elif not candidate_hits:
                candidate_hits = [
                    hit
                    for hit in graph_hits
                    if hit and _hit_identity(hit)
                ]
            if not entities:
                entities = [
                    _resolved_name("", hit)
                    for hit in candidate_hits
                ]
            tools = _tool_plan(intent.intent, "code_analyst", query)
            parallel = tools.get("parallel", ())
            if plan.iteration > 1 and plan.retry_tools is not None:
                retry = set(plan.retry_tools)
                parallel = tuple(tool for tool in parallel if tool in retry)
                log.info(
                    "orchestrator.refinement_analyst_tools",
                    tools=list(parallel),
                    iteration=plan.iteration,
                    candidate_limit=_candidate_limit(plan.iteration),
                    correlation_id=correlation_id,
                )
            if plan.iteration > 1:
                cap = _candidate_limit(plan.iteration)
                if len(candidate_hits) > cap:
                    candidate_hits = candidate_hits[:cap]
            if not parallel:
                if plan.iteration > 1:
                    agent_status["code_analyst"] = AgentOutput(
                        agent="code_analyst",
                        ok=True,
                        output={},
                    )
                    return
                log.info(
                    "orchestrator.empty_tool_plan",
                    agent="code_analyst",
                    intent=intent.intent,
                    correlation_id=correlation_id,
                )
                agent_status["code_analyst"] = AgentOutput(
                    agent="code_analyst",
                    ok=False,
                    degraded_note="no tool plan for intent; specialist was not invoked",
                    error="empty_tool_plan",
                )
                return

            start, end = _default_snippet_range()

            if "find_patterns" in parallel:
                pattern = _pattern_name(query, entities)
                path_prefix = pattern_path_prefix(query)
                result = await _call(
                    "code_analyst",
                    "find_patterns",
                    _bind(
                        clients.code_analyst.find_patterns,
                        pattern,
                        path_prefix=path_prefix,
                    ),
                    settings.code_analyst_timeout_s,
                )
                output["patterns"] = _jsonish(result)
                agent_status["code_analyst"] = AgentOutput(
                    agent="code_analyst",
                    ok=True,
                    output=output,
                )
                return

            if "compare_implementations" in parallel:
                if len(entities) < 2:
                    agent_status["code_analyst"] = AgentOutput(
                        agent="code_analyst",
                        ok=True,
                        output={"comparison": None, "note": "need two entities to compare"},
                    )
                    return
                name_a = _resolved_name(
                    entities[0], graph_hits[0] if graph_hits else {}
                )
                name_b = _resolved_name(
                    entities[1], graph_hits[1] if len(graph_hits) > 1 else {}
                )
                result = await _call(
                    "code_analyst",
                    "compare_implementations",
                    _bind(clients.code_analyst.compare_implementations, name_a, name_b),
                    settings.code_analyst_timeout_s,
                )
                output["comparison"] = _jsonish(result)

            analysis_tasks: list[asyncio.Task[Any]] = []
            analysis_keys: list[str] = []
            if candidate_hits:
                named_hits: list[tuple[str, dict[str, Any]]] = [
                    (_resolved_name("", hit), hit) for hit in candidate_hits
                ]
            else:
                named_hits = [
                    (
                        _resolved_name(
                            entity, graph_hits[i] if i < len(graph_hits) else {}
                        ),
                        graph_hits[i] if i < len(graph_hits) else {},
                    )
                    for i, entity in enumerate(entities)
                ]
            names = [name for name, _hit in named_hits if name]

            if "explain_implementation" in parallel:
                for name, _hit in named_hits:
                    if not name:
                        continue
                    analysis_keys.append("explanations")
                    analysis_tasks.append(
                        asyncio.create_task(
                            _call(
                                "code_analyst",
                                "explain_implementation",
                                _bind(clients.code_analyst.explain_implementation, name),
                                settings.code_analyst_timeout_s,
                            )
                        )
                    )
            if "analyze_class" in parallel:
                for name, hit in named_hits:
                    if not name or not _hit_is_class(hit):
                        continue
                    analysis_keys.append("class_analyses")
                    analysis_tasks.append(
                        asyncio.create_task(
                            _call(
                                "code_analyst",
                                "analyze_class",
                                _bind(clients.code_analyst.analyze_class, name),
                                settings.code_analyst_timeout_s,
                            )
                        )
                    )
            if "analyze_function" in parallel:
                for name, hit in named_hits:
                    if not name or _hit_is_class(hit):
                        continue
                    analysis_keys.append("function_analyses")
                    analysis_tasks.append(
                        asyncio.create_task(
                            _call(
                                "code_analyst",
                                "analyze_function",
                                _bind(clients.code_analyst.analyze_function, name),
                                settings.code_analyst_timeout_s,
                            )
                        )
                    )
            if "get_code_snippet" in parallel:

                def _snippet_factory(
                    *,
                    file_path: str | None,
                    line_start: int,
                    line_end: int,
                    qualified_name: str | None,
                ) -> Callable[[], Awaitable[Any]]:
                    async def _run() -> Any:
                        return await clients.code_analyst.get_code_snippet(
                            qualified_name=qualified_name,
                            file_path=file_path,
                            line_start=line_start,
                            line_end=line_end,
                        )

                    return _run

                snippet_hits: list[dict[str, Any]]
                if candidate_hits:
                    snippet_hits = candidate_hits
                elif graph_available and graph_hits:
                    snippet_hits = []
                    snippet_entities = entities or names
                    for i, entity in enumerate(snippet_entities):
                        graph_hit = graph_hits[i] if i < len(graph_hits) else {}
                        file_path = graph_hit.get("file_path") or graph_hit.get("filePath")
                        qualified = graph_hit.get("qualified_name") or entity
                        snippet_hits.append(
                            {
                                "file_path": file_path,
                                "line_start": graph_hit.get("line_start")
                                or graph_hit.get("lineStart")
                                or start,
                                "line_end": graph_hit.get("line_end")
                                or graph_hit.get("lineEnd")
                                or end,
                                "qualified_name": qualified,
                            }
                        )
                else:
                    snippet_hits = [
                        {"qualified_name": entity, "line_start": start, "line_end": end}
                        for entity in (entities or names)
                    ]

                for snippet_hit in snippet_hits:
                    file_path = snippet_hit.get("file_path") or snippet_hit.get("filePath")
                    line_start = (
                        snippet_hit.get("line_start")
                        or snippet_hit.get("lineStart")
                        or start
                    )
                    line_end = (
                        snippet_hit.get("line_end") or snippet_hit.get("lineEnd") or end
                    )
                    qualified = snippet_hit.get("qualified_name") or snippet_hit.get("name")
                    if not file_path and not qualified:
                        continue
                    analysis_keys.append("snippets")
                    analysis_tasks.append(
                        asyncio.create_task(
                            _call(
                                "code_analyst",
                                "get_code_snippet",
                                _snippet_factory(
                                    file_path=str(file_path) if file_path else None,
                                    line_start=int(line_start) if line_start else start,
                                    line_end=int(line_end) if line_end else end,
                                    qualified_name=str(qualified) if qualified else None,
                                ),
                                settings.code_analyst_timeout_s,
                            )
                        )
                    )

            if analysis_tasks:
                results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
                grouped: dict[str, list[Any]] = {}
                for key, result in zip(analysis_keys, results, strict=True):
                    if isinstance(result, BaseException):
                        if not isinstance(result, Exception):
                            raise result
                        log.info(
                            "orchestrator.analyst_task_failed",
                            bucket=key,
                            error=type(result).__name__,
                            correlation_id=correlation_id,
                        )
                        continue
                    grouped.setdefault(key, []).append(_jsonish(result))
                output.update(grouped)
                if not grouped and not output:
                    raise TimeoutError("code_analyst timeout")

            agent_status["code_analyst"] = AgentOutput(
                agent="code_analyst",
                ok=True,
                output=output or None,
            )
        except _BudgetSkip:
            existing = agent_status.get("code_analyst")
            if output or (existing is not None and existing.output):
                agent_status["code_analyst"] = AgentOutput(
                    agent="code_analyst",
                    ok=True,
                    output=output or (existing.output if existing is not None else None),
                )
            else:
                log.info(
                    "orchestrator.budget_skip",
                    agent="code_analyst",
                    correlation_id=correlation_id,
                )
                agent_status["code_analyst"] = AgentOutput(
                    agent="code_analyst",
                    ok=False,
                    degraded_note="budget reserved remaining time for synthesis",
                    error="budget_skip",
                )
        except (TimeoutError, AgentUnavailableError):
            code_available = False
            agent_status["code_analyst"] = AgentOutput(
                agent="code_analyst",
                ok=False,
                degraded_note="code_analyst unavailable/timeout; returning graph facts only",
                error="code_analyst timeout",
            )
        except Exception as exc:  # pragma: no cover - defensive
            code_available = False
            agent_status["code_analyst"] = AgentOutput(
                agent="code_analyst",
                ok=False,
                degraded_note="code_analyst unavailable; returning graph facts only",
                error=str(exc),
            )

    async def _run_index(intent: QueryIntent) -> None:
        try:
            repo_url: str | None = None
            report = await _call(
                "indexer",
                "index_repository",
                _bind(clients.indexer.index_repository, repo_url),
                settings.indexer_timeout_s,
            )
            agent_status["indexer"] = AgentOutput(
                agent="indexer",
                ok=True,
                output=report,
            )
        except _BudgetSkip:
            log.info(
                "orchestrator.budget_skip",
                agent="indexer",
                correlation_id=correlation_id,
            )
            agent_status["indexer"] = AgentOutput(
                agent="indexer",
                ok=False,
                degraded_note="budget reserved remaining time for synthesis",
                error="budget_skip",
            )
        except (TimeoutError, AgentUnavailableError):
            agent_status["indexer"] = AgentOutput(
                agent="indexer",
                ok=False,
                degraded_note="indexer unavailable/timeout",
                error="indexer timeout",
            )
        except Exception as exc:  # pragma: no cover - defensive
            agent_status["indexer"] = AgentOutput(
                agent="indexer",
                ok=False,
                degraded_note="indexer unavailable",
                error=str(exc),
            )

    async def _run_graph_and_signal(intent: QueryIntent) -> None:
        try:
            await _run_graph(intent)
        finally:
            graph_coords_ready.set()
            graph_done.set()

    async def _run_code_maybe_wait(intent: QueryIntent) -> None:
        if "graph_query" in planned_agents and not code_analyst_can_start_now(intent, query):
            await graph_coords_ready.wait()
        await _run_code(intent)

    # Concurrent specialists. Graph and analyst overlap when analyst inputs
    # are already known; otherwise the analyst waits for graph coordinates.
    tasks: list[asyncio.Task[None]] = []
    for agent in plan.agents:
        if agent == "graph_query":
            tasks.append(asyncio.create_task(_run_graph_and_signal(plan.intent)))
        elif agent == "code_analyst":
            tasks.append(asyncio.create_task(_run_code_maybe_wait(plan.intent)))
        elif agent == "indexer":
            tasks.append(asyncio.create_task(_run_index(plan.intent)))
    if tasks:
        await asyncio.gather(*tasks)

    if graph_available and "graph_query" in planned_agents:
        existing = agent_status.get("graph_query")
        if existing is not None and isinstance(existing.output, dict):
            output = existing.output
        else:
            output = {}
        agent_status["graph_query"] = AgentOutput(
            agent="graph_query",
            ok=True if existing is None else existing.ok,
            output={
                "entities": graph_entities,
                "queried_entities": output.get("queried_entities", plan.intent.entities),
                **{k: v for k, v in output.items() if k not in {"entities", "queried_entities"}},
                **graph_extra,
            },
            degraded_note=None if existing is None else existing.degraded_note,
            error=None if existing is None else existing.error,
        )

    for agent_name, agent_output in agent_status.items():
        agent_tools = [item for item in tools_invoked if item.startswith(f"{agent_name}.")]
        agent_status[agent_name] = agent_output.model_copy(update={"tools_invoked": agent_tools})

    _ = code_available, correlation_id
    return agent_status
