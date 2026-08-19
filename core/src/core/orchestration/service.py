"""Orchestrator loop: route, execute specialists, synthesize, cache."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from core.exceptions import RoutingError
from core.health import CircuitBreakerSnapshot
from core.llm.provider import LLMProvider
from core.logging import get_logger
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger, empty_token_totals
from core.orchestration.budget import RequestBudget
from core.orchestration.loop import run_refinement_loop
from core.orchestration.models import AgentName, ExecutionPlan, SynthesisResult
from core.orchestration.router import (
    _rule_based_entities,
    analyze_query,
    apply_conversation_entities,
    intent_from_rule_result,
    normalize_grounded_intent,
    resolve_entities,
    route_to_agents,
    rule_based_route,
)
from core.orchestration.synthesis import synthesize_response
from core.resilience.circuit_breaker import CircuitBreakerRegistry
from core.settings import OrchestratorSettings

log = get_logger(__name__)

OnQueryEvent = Callable[[str, dict[str, Any]], Awaitable[None]]


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip()).lower()


def _entity_fingerprint(entities: Sequence[str]) -> str:
    unique = {item.strip() for item in entities if item.strip()}
    return ",".join(sorted(unique))


def _cache_key(
    *,
    session_id: str,
    normalized_query: str,
    index_version: str | None,
    entities: Sequence[str] = (),
) -> str:
    version = index_version or "unknown"
    fingerprint = _entity_fingerprint(entities)
    return f"orchestrator:v1:{version}:{session_id}:{normalized_query}:{fingerprint}"


def _skipped_agents(
    plan: ExecutionPlan,
    agent_outputs: Mapping[AgentName, Any],
) -> list[dict[str, str]]:
    skipped: list[dict[str, str]] = []
    for agent in plan.agents:
        output = agent_outputs.get(agent)
        tools = list(getattr(output, "tools_invoked", None) or []) if output is not None else []
        if tools:
            continue
        reason = "not_invoked"
        if output is not None:
            reason = str(output.error or output.degraded_note or "not_invoked")
        skipped.append({"agent": agent, "reason": reason})
    return skipped


def _tools_invoked(
    plan: ExecutionPlan, agent_outputs: Mapping[AgentName, Any]
) -> list[str]:
    tools: list[str] = []
    for agent in plan.agents:
        output = agent_outputs.get(agent)
        invoked = getattr(output, "tools_invoked", None) if output is not None else None
        if invoked:
            tools.extend(list(invoked))
    return tools


class MemoryClient(Protocol):
    """Memory agent client used by the orchestrator loop."""

    async def get_context(
        self,
        session_id: str,
        token_budget: int = 3000,
    ) -> ConversationContext:
        """Load conversation summary and recent turns.

        Args:
            session_id: Conversation id.
            token_budget: Maximum tokens of context to return.

        Returns:
            Conversation context for this session.
        """
        ...

    async def get_cached_response(self, cache_key: str) -> Any | None:
        """Return a cached orchestrator payload when present.

        Args:
            cache_key: Caller-computed cache key.

        Returns:
            Cached payload or ``None``.
        """
        ...

    async def cache_response(self, cache_key: str, response_json: Any) -> None:
        """Store an orchestrator payload in Memory.

        Args:
            cache_key: Caller-computed cache key.
            response_json: Opaque JSON-serializable payload.
        """
        ...

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Append one conversation turn.

        Args:
            session_id: Conversation id.
            role: ``user`` or ``assistant``.
            content: Turn text.
        """
        ...


class GraphQueryClient(Protocol):
    """Graph Query agent client used by the orchestrator loop."""

    async def get_statistics(self) -> Any:
        """Return graph statistics including ``index_version``.

        Returns:
            Statistics payload.
        """
        ...

    async def find_entity(self, name: str, entity_type: str | None = None) -> Any:
        """Look up an entity by name.

        Args:
            name: Entity name or phrase.
            entity_type: Optional label filter.

        Returns:
            Match payload.
        """
        ...

    async def get_dependencies(self, name: str) -> Any:
        """Return outgoing neighbors for ``name``.

        Args:
            name: Qualified name.

        Returns:
            Neighbor payload.
        """
        ...

    async def get_dependents(self, name: str) -> Any:
        """Return incoming neighbors for ``name``.

        Args:
            name: Qualified name.

        Returns:
            Neighbor payload.
        """
        ...

    async def find_related(self, name: str, relationship_type: str) -> Any:
        """Return neighbors along one relationship type.

        Args:
            name: Qualified name.
            relationship_type: Graph relationship type.

        Returns:
            Related-entity payload.
        """
        ...

    async def trace_imports(self, module: str, depth: int = 5) -> Any:
        """Follow import chains from ``module``.

        Args:
            module: Module name.
            depth: Traversal cap.

        Returns:
            Import-trace payload.
        """
        ...


class CodeAnalystClient(Protocol):
    """Code Analyst agent client used by the orchestrator loop."""

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
        """Explain how a function or class is implemented.

        Args:
            qualified_name: Graph coordinate.

        Returns:
            Explanation payload.
        """
        ...

    async def analyze_function(self, qualified_name: str) -> Any:
        """Analyze a function or method.

        Args:
            qualified_name: Graph coordinate.

        Returns:
            Analysis payload.
        """
        ...

    async def analyze_class(self, qualified_name: str) -> Any:
        """Analyze a class.

        Args:
            qualified_name: Graph coordinate.

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
        """Find supported code patterns.

        Args:
            pattern: Pattern name.
            path_prefix: Optional module or file-path prefix for decorator scoping.

        Returns:
            Pattern payload.
        """
        ...


class IndexerClient(Protocol):
    """Indexer agent client used by the orchestrator loop."""

    async def index_repository(self, repo_url: str | None = None) -> Any:
        """Trigger a repository index.

        Args:
            repo_url: Optional clone URL override.

        Returns:
            Index report payload.
        """
        ...


class OrchestratorClients(Protocol):
    """Specialist clients required by :meth:`OrchestratorService.handle_query`."""

    graph_query: GraphQueryClient
    code_analyst: CodeAnalystClient
    indexer: IndexerClient
    memory: MemoryClient


def _serialize_agent_outputs(agent_outputs: Mapping[AgentName, Any]) -> dict[str, Any]:
    """JSON-ify specialist payloads for eval groundedness checks.

    Args:
        agent_outputs: Per-agent results from the executed plan.

    Returns:
        Mapping of agent name to JSON-serializable output.
    """
    serialized: dict[str, Any] = {}
    for name, output in agent_outputs.items():
        serialized[str(name)] = (
            output.model_dump(mode="json") if hasattr(output, "model_dump") else output
        )
    return serialized


@dataclass(frozen=True)
class HandleQueryResult:
    """Answer plus orchestrator metadata for one ``handle_query`` turn."""

    answer: str
    metadata: dict[str, Any]
    agent_outputs: dict[str, Any] = field(default_factory=dict)


class OrchestratorService:
    """Single-process orchestrator loop (tool-agnostic for tests)."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        settings: OrchestratorSettings,
        token_ledger: TokenLedger | None = None,
        breakers: CircuitBreakerRegistry | None = None,
    ) -> None:
        """Create the orchestrator loop.

        Args:
            llm_provider: Routing and synthesis backend.
            settings: Timeouts, routing strategy, and prompt budget.
            token_ledger: Optional per-request token accumulator.
            breakers: Optional upstream circuit-breaker registry.
        """
        self._llm_provider = llm_provider
        self._settings = settings
        self._token_ledger = token_ledger or TokenLedger()
        self._breakers = breakers

    def _breaker_snapshots(self) -> dict[str, CircuitBreakerSnapshot]:
        if self._breakers is None:
            return {}
        return self._breakers.snapshot()

    async def handle_query(
        self,
        query: str,
        session_id: str,
        *,
        clients: OrchestratorClients,
        token_budget: int = 3000,
        correlation_id: str = "-",
        on_token: Callable[[str], Awaitable[None]] | None = None,
        on_event: OnQueryEvent | None = None,
    ) -> HandleQueryResult:
        """Full orchestrator loop with graceful degradation and response caching.

        Synthesis LLM timeouts and errors keep any tokens already streamed
        (``metadata.partial=true``). With no emitted tokens they return an
        evidence-only answer with ``degraded=true`` rather than discarding
        retrieved specialist output.

        Args:
            query: User question.
            session_id: Conversation id.
            clients: Specialist MCP (or test) clients.
            token_budget: Memory context token budget.
            correlation_id: Request id for logs and the token ledger.
            on_token: Optional callback for streamed synthesis chunks.
            on_event: Optional callback for live ``routing`` / ``agent_result`` events.

        Returns:
            Final answer and metadata.

        Raises:
            RoutingError: When intent classification cannot produce a plan.
        """
        self._token_ledger.open(correlation_id)
        budget = RequestBudget.from_settings(self._settings)

        memory_context: ConversationContext
        memory_available = True
        try:
            memory_context = await clients.memory.get_context(
                session_id, token_budget=token_budget
            )
        except Exception as exc:
            memory_available = False
            memory_context = ConversationContext()
            log.warning(
                "orchestrator.memory_context_failed",
                correlation_id=correlation_id,
                error=str(exc),
                exception_type=type(exc).__name__,
            )

        graph_statistics_available = True
        index_version: str | None = None
        try:
            stats = await clients.graph_query.get_statistics()
            index_version = getattr(stats, "index_version", None)
        except Exception as exc:
            graph_statistics_available = False
            index_version = None
            log.warning(
                "orchestrator.graph_statistics_failed",
                correlation_id=correlation_id,
                error=str(exc),
                exception_type=type(exc).__name__,
            )

        resolved_entities = resolve_entities(
            query,
            _rule_based_entities(query),
            memory_context,
        )
        cache_key = _cache_key(
            session_id=session_id,
            normalized_query=_normalize_query(query),
            index_version=index_version,
            entities=resolved_entities,
        )

        if memory_available:
            try:
                cached = await clients.memory.get_cached_response(cache_key)
                if cached is not None:
                    self._token_ledger.close(correlation_id)
                    from core.observability.metrics import record_cache_lookup

                    record_cache_lookup(hit=True)
                    payload: Any
                    if hasattr(cached, "response_json"):
                        payload = cached.response_json
                    elif isinstance(cached, dict) and "response_json" in cached:
                        payload = cached["response_json"]
                    else:
                        payload = cached
                    cached_tools: list[str] = []
                    if isinstance(payload, dict):
                        cached_meta = payload.get("metadata")
                        if isinstance(cached_meta, dict):
                            maybe_tools = cached_meta.get("tools_invoked")
                            if isinstance(maybe_tools, list):
                                cached_tools = [str(item) for item in maybe_tools]
                    return HandleQueryResult(
                        answer=str(payload.get("answer", "")) if isinstance(payload, dict) else "",
                        metadata={
                            "cached": True,
                            "cache_key": cache_key,
                            "index_version": index_version,
                            "routing_mode": "cache_hit",
                            "tools_invoked": cached_tools,
                            "tokens": empty_token_totals(),
                            "memory_available": memory_available,
                            "graph_statistics_available": graph_statistics_available,
                        },
                        agent_outputs={},
                    )
            except Exception as exc:
                log.warning(
                    "orchestrator.cache_lookup_failed",
                    correlation_id=correlation_id,
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )

        from core.observability.metrics import record_cache_lookup

        record_cache_lookup(hit=False)

        try:
            if self._settings.routing_strategy == "llm_first":
                intent = await analyze_query(
                    query,
                    memory_context,
                    llm_provider=self._llm_provider,
                    token_ledger=self._token_ledger,
                    correlation_id=correlation_id,
                )
            else:
                rule_result = rule_based_route(query, settings=self._settings)
                if rule_result.ambiguous:
                    intent = await analyze_query(
                        query,
                        memory_context,
                        llm_provider=self._llm_provider,
                        token_ledger=self._token_ledger,
                        correlation_id=correlation_id,
                    )
                else:
                    intent = intent_from_rule_result(
                        query, rule_result, routing_mode="rules", context=memory_context
                    )
        except RoutingError:
            raise
        except Exception as exc:
            raise RoutingError(
                agent="orchestrator",
                correlation_id=correlation_id,
                message=str(exc),
            ) from exc
        intent = normalize_grounded_intent(
            apply_conversation_entities(intent, query, memory_context),
            query,
        )
        plan: ExecutionPlan = route_to_agents(intent)
        log.info(
            "orchestrator.routing_decided",
            routing_mode=intent.routing_mode,
            target_agents=intent.target_agents,
        )

        agent_outputs, plan_iterations = await run_refinement_loop(
            plan,
            query=query,
            context=memory_context,
            clients=clients,
            settings=self._settings,
            correlation_id=correlation_id,
            deadline_monotonic=budget.deadline_monotonic,
            budget=budget,
            token_ledger=self._token_ledger,
        )
        tools_invoked: list[str] = []
        for iteration in plan_iterations:
            for tool in iteration.tools_invoked:
                if tool not in tools_invoked:
                    tools_invoked.append(tool)
        if not tools_invoked:
            tools_invoked = _tools_invoked(plan, agent_outputs)

        if on_event is not None:
            for item in plan_iterations:
                await on_event("routing", item.model_dump(mode="json"))
            await on_event(
                "agent_result",
                {"agent": "orchestrator", "ok": True, "degraded": False},
            )

        synthesis: SynthesisResult = await synthesize_response(
            query,
            agent_outputs,
            memory_context,
            llm_provider=self._llm_provider,
            settings=self._settings,
            token_ledger=self._token_ledger,
            correlation_id=correlation_id,
            budget=budget,
            on_token=on_token,
        )
        answer = synthesis.answer

        tokens = self._token_ledger.close(correlation_id)
        breaker_snapshots = self._breaker_snapshots()
        agent_degraded = bool(
            any(
                hasattr(output, "degraded_note") and output.degraded_note
                for output in agent_outputs.values()
            )
            or any(item.state == "open" for item in breaker_snapshots.values())
        )
        degraded = agent_degraded or synthesis.evidence_only
        metadata: dict[str, Any] = {
            "cached": False,
            "cache_key": cache_key,
            "index_version": index_version,
            "routing_mode": intent.routing_mode,
            "memory_available": memory_available,
            "graph_statistics_available": graph_statistics_available,
            "tokens": tokens,
            "degraded": degraded,
            "tools_invoked": tools_invoked,
            "plan_iterations": [item.model_dump(mode="json") for item in plan_iterations],
            "plan_iteration_count": len(plan_iterations),
        }
        skipped_agents = _skipped_agents(plan, agent_outputs)
        if skipped_agents:
            metadata["skipped_agents"] = skipped_agents
        if synthesis.evidence_only:
            metadata["evidence_only"] = True
            metadata["degraded_reason"] = synthesis.degraded_reason
        if synthesis.partial:
            metadata["partial"] = True
            if synthesis.degraded_reason is not None:
                metadata["degraded_reason"] = synthesis.degraded_reason
        if budget.exhausted:
            metadata["budget_exhausted"] = budget.exhausted
            from core.observability.metrics import record_budget_exhausted

            record_budget_exhausted(budget.exhausted)
        if synthesis.prompt_truncated is not None:
            metadata["prompt_truncated"] = synthesis.prompt_truncated.model_dump(mode="json")
        if breaker_snapshots:
            metadata["circuit_breakers"] = {
                name: snapshot.model_dump(mode="json")
                for name, snapshot in breaker_snapshots.items()
            }

        if not memory_available:
            metadata["note"] = "memory unavailable; proceeding statelessly"

        if memory_available:
            try:
                await clients.memory.append_turn(session_id, "user", query)
                await clients.memory.append_turn(session_id, "assistant", answer)
            except Exception as exc:
                log.warning(
                    "orchestrator.append_turn_failed",
                    correlation_id=correlation_id,
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )
            try:
                await clients.memory.cache_response(
                    cache_key,
                    {"answer": answer, "metadata": metadata},
                )
            except Exception as exc:
                log.warning(
                    "orchestrator.cache_store_failed",
                    correlation_id=correlation_id,
                    error=str(exc),
                    exception_type=type(exc).__name__,
                )

        return HandleQueryResult(
            answer=answer,
            metadata=metadata,
            agent_outputs=_serialize_agent_outputs(agent_outputs),
        )


def build_default_llm_provider() -> LLMProvider:
    """Choose OpenRouter if a key exists; otherwise use the deterministic offline stub.

    Returns:
        Production or offline LLM provider. Tests should inject StubProvider.
    """
    from core.llm.factory import build_llm_provider

    return build_llm_provider()
