"""Orchestrator loop implementation used by the FastMCP adapter and tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.provider import LLMProvider
from core.logging import get_logger
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger
from core.orchestration.executor import run_plan
from core.orchestration.models import ExecutionPlan, QueryIntent
from core.orchestration.router import (
    analyze_query,
    intent_from_rule_result,
    route_to_agents,
    rule_based_route,
)
from core.orchestration.synthesis import synthesize_response
from core.settings import LLMSettings, OrchestratorSettings

log = get_logger(__name__)


def _normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip()).lower()


def _cache_key(*, normalized_query: str, index_version: str | None) -> str:
    version = index_version or "unknown"
    return f"orchestrator:v1:{version}:{normalized_query}"


class MemoryClient(Protocol):
    async def get_context(
        self,
        session_id: str,
        token_budget: int = 3000,
    ) -> ConversationContext: ...

    async def get_cached_response(self, cache_key: str) -> Any | None: ...

    async def cache_response(self, cache_key: str, response_json: Any) -> None: ...

    async def append_turn(self, session_id: str, role: str, content: str) -> None: ...


class GraphQueryClient(Protocol):
    async def get_statistics(self) -> Any: ...

    async def find_entity(self, name: str, entity_type: str | None = None) -> Any: ...


class CodeAnalystClient(Protocol):
    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> Any: ...


class IndexerClient(Protocol):
    async def index_repository(self, repo_url: str | None = None) -> Any: ...


class OrchestratorClients(Protocol):
    graph_query: GraphQueryClient
    code_analyst: CodeAnalystClient
    indexer: IndexerClient
    memory: MemoryClient


@dataclass(frozen=True)
class HandleQueryResult:
    answer: str
    metadata: dict[str, Any]


class OrchestratorService:
    """Single-process orchestrator loop (tool-agnostic for tests)."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        settings: OrchestratorSettings,
        token_ledger: TokenLedger | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._settings = settings
        self._token_ledger = token_ledger or TokenLedger()

    async def handle_query(
        self,
        query: str,
        session_id: str,
        *,
        clients: OrchestratorClients,
        token_budget: int = 3000,
        correlation_id: str = "-",
    ) -> HandleQueryResult:
        """Full orchestrator loop with graceful degradation and response caching."""

        self._token_ledger.open(correlation_id)

        # 1) Conversation context.
        memory_context: ConversationContext
        memory_available = True
        try:
            memory_context = await clients.memory.get_context(
                session_id, token_budget=token_budget
            )
        except Exception:
            memory_available = False
            memory_context = ConversationContext()

        # 2) Cache key.
        index_version: str | None = None
        cache_key: str | None = None
        try:
            stats = await clients.graph_query.get_statistics()
            index_version = getattr(stats, "index_version", None)
        except Exception:
            index_version = None

        cache_key = _cache_key(
            normalized_query=_normalize_query(query),
            index_version=index_version,
        )

        # 3) Cache lookup.
        if memory_available:
            try:
                cached = await clients.memory.get_cached_response(cache_key)
                if cached is not None:
                    self._token_ledger.close(correlation_id)
                    # `response_json` is stored as an opaque JSON blob.
                    #
                    # Depending on whether the memory MCP client returns a
                    # Pydantic model or a plain dict, `cached` may take different
                    # shapes:
                    # - model: cached.response_json
                    # - dict:  cached["response_json"]
                    payload: Any
                    if hasattr(cached, "response_json"):
                        payload = cached.response_json
                    elif isinstance(cached, dict) and "response_json" in cached:
                        payload = cached["response_json"]
                    else:
                        payload = cached
                    return HandleQueryResult(
                        answer=str(payload.get("answer", "")) if isinstance(payload, dict) else "",
                        metadata={
                            "cached": True,
                            "cache_key": cache_key,
                            "index_version": index_version,
                            "routing_mode": "cache_hit",
                            "tokens": {
                                "total": 0,
                                "prompt": 0,
                                "completion": 0,
                                "llm_calls": 0,
                                "by_purpose": {},
                            },
                        },
                    )
            except Exception:
                # Cache failure should not break the query.
                pass

        # 4) Analyze -> route -> execute.
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
                    intent = intent_from_rule_result(query, rule_result, routing_mode="rules")
        except Exception:
            # Never 500: if routing fails for any reason, fall back to rules.
            fallback = rule_based_route(query, settings=self._settings)
            if fallback.ambiguous:
                intent = QueryIntent(
                    routing_mode="rules_fallback",
                    intent="mixed",
                    entities=[],
                    target_agents=["graph_query", "code_analyst"],
                    reasoning="rules_fallback: routing failure defaulted to mixed",
                )
            else:
                intent = intent_from_rule_result(query, fallback, routing_mode="rules_fallback")
        plan: ExecutionPlan = route_to_agents(intent)
        log.info(
            "orchestrator.routing_decided",
            routing_mode=intent.routing_mode,
            target_agents=intent.target_agents,
        )

        agent_outputs = await run_plan(
            plan,
            query=query,
            context=memory_context,
            clients=clients,
            settings=self._settings,
            correlation_id=correlation_id,
        )

        # 5) Synthesize (one LLM call).
        try:
            answer = await synthesize_response(
                query,
                agent_outputs,
                memory_context,
                llm_provider=self._llm_provider,
                settings=self._settings,
                token_ledger=self._token_ledger,
                correlation_id=correlation_id,
            )
        except Exception:
            answer = "Degraded response: failed to synthesize final answer."
            tokens = self._token_ledger.close(correlation_id)
            metadata: dict[str, Any] = {
                "cached": False,
                "cache_key": cache_key,
                "index_version": index_version,
                "routing_mode": intent.routing_mode,
                "memory_available": memory_available,
                "degraded": True,
                "note": "synthesis unavailable; returning degraded message",
                "tokens": tokens,
            }
            return HandleQueryResult(answer=answer, metadata=metadata)

        # 6) Append turns and cache put.
        tokens = self._token_ledger.close(correlation_id)
        metadata = {
            "cached": False,
            "cache_key": cache_key,
            "index_version": index_version,
            "routing_mode": intent.routing_mode,
            "memory_available": memory_available,
            "tokens": tokens,
            "degraded": bool(
                any(
                    hasattr(output, "degraded_note") and output.degraded_note
                    for output in agent_outputs.values()
                )
            ),
        }

        if not memory_available:
            metadata["note"] = "memory unavailable; proceeding statelessly"

        if memory_available:
            try:
                await clients.memory.append_turn(session_id, "user", query)
                await clients.memory.append_turn(session_id, "assistant", answer)
            except Exception:
                pass
            try:
                await clients.memory.cache_response(
                    cache_key,
                    {"answer": answer, "metadata": metadata},
                )
            except Exception:
                pass

        return HandleQueryResult(answer=answer, metadata=metadata)


def build_default_llm_provider() -> LLMProvider:
    """Choose OpenRouter if key exists; otherwise use deterministic offline stub."""

    llm_settings = LLMSettings.from_env()
    if llm_settings.api_key:
        return OpenRouterProvider.from_env()
    return OfflineProvider()

