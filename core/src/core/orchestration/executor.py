"""Execute an :class:`~core.orchestration.models.ExecutionPlan`."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Protocol

from core.exceptions import AgentUnavailableError
from core.memory import ConversationContext
from core.settings import OrchestratorSettings

from .models import AgentName, AgentOutput, ExecutionPlan, QueryIntent


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


class AgentClients(Protocol):
    graph_query: GraphQueryClient
    code_analyst: CodeAnalystClient
    indexer: IndexerClient


def _default_snippet_range() -> tuple[int, int]:
    # Keep this deterministic and small for degraded mode.
    return (1, 120)


async def run_plan(
    plan: ExecutionPlan,
    *,
    query: str,
    context: ConversationContext | None,
    clients: AgentClients,
    settings: OrchestratorSettings,
    correlation_id: str,
) -> Mapping[AgentName, AgentOutput]:
    """Execute plan while degrading gracefully.

    The executor never raises; it returns best-effort `AgentOutput` objects.
    """

    _ = context  # reserved for future intent/context coupling

    graph_entities: list[dict[str, Any]] = []
    graph_available = True
    code_available = True

    agent_status: dict[AgentName, AgentOutput] = {
        agent: AgentOutput(agent=agent, ok=True) for phase in plan.phases for agent in phase
    }

    async def _run_graph(intent: QueryIntent) -> None:
        nonlocal graph_entities, graph_available
        try:
            entities = intent.entities or []
            if not entities:
                return
            tasks = [
                clients.graph_query.find_entity(name=entity)
                for entity in entities
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=settings.graph_query_timeout_s,
            )
            graph_entities = [dict(r) for r in results if r is not None]
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
        try:
            entities = intent.entities or []
            if not entities:
                return

            start, end = _default_snippet_range()
            snippet_tasks: list[asyncio.Task[Any]] = []
            if graph_available and graph_entities:
                # Use locations from graph hits when possible.
                for i, entity in enumerate(entities):
                    hit = graph_entities[i] if i < len(graph_entities) else {}
                    file_path = (
                        hit.get("file_path")
                        or hit.get("filePath")
                        or (entity if isinstance(entity, str) else None)
                    )
                    line_start = hit.get("line_start") or hit.get("lineStart") or start
                    line_end = hit.get("line_end") or hit.get("lineEnd") or end
                    snippet_tasks.append(
                        asyncio.create_task(
                            clients.code_analyst.get_code_snippet(
                                file_path=str(file_path) if file_path else str(entity),
                                line_start=int(line_start) if line_start else start,
                                line_end=int(line_end) if line_end else end,
                            )
                        )
                    )
            else:
                # Degraded mode: read raw file directly.
                for entity in entities:
                    snippet_tasks.append(
                        asyncio.create_task(
                            clients.code_analyst.get_code_snippet(
                                file_path=entity,
                                line_start=start,
                                line_end=end,
                            )
                        )
                    )

            results = await asyncio.wait_for(
                asyncio.gather(*snippet_tasks),
                timeout=settings.code_analyst_timeout_s,
            )
            agent_status["code_analyst"] = AgentOutput(
                agent="code_analyst",
                ok=True,
                output={"snippets": results},
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
            agent_status["indexer"] = AgentOutput(agent="indexer", ok=True)
            if hasattr(clients.indexer, "index_repository"):
                # Executor always tries repo indexing; callers can pre-filter.
                report = await asyncio.wait_for(
                    clients.indexer.index_repository(repo_url=repo_url),
                    timeout=settings.indexer_timeout_s,
                )
                agent_status["indexer"] = AgentOutput(
                    agent="indexer",
                    ok=True,
                    output=report,
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

    # Phase-parallel execution.
    for phase in plan.phases:
        tasks: list[asyncio.Task[None]] = []
        for agent in phase:
            if agent == "graph_query":
                tasks.append(asyncio.create_task(_run_graph(plan.intent)))
            elif agent == "code_analyst":
                tasks.append(asyncio.create_task(_run_code(plan.intent)))
            elif agent == "indexer":
                tasks.append(asyncio.create_task(_run_index(plan.intent)))
        if tasks:
            await asyncio.gather(*tasks)

    # Attach graph_query output if available and not already set as degraded.
    if graph_available:
        agent_status.setdefault("graph_query", AgentOutput(agent="graph_query", ok=True))
        # Keep it JSON-serializable for LLM prompt embedding.
        agent_status["graph_query"] = AgentOutput(
            agent="graph_query",
            ok=True,
            output={"entities": graph_entities, "queried_entities": plan.intent.entities},
        )

    _ = code_available, query, correlation_id
    return agent_status

