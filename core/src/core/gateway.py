"""Gateway-facing chat and index orchestration models/services."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from core.exceptions import AgentUnavailableError
from core.memory import ConversationContext
from core.orchestration.models import AgentName, AgentOutput, ExecutionPlan, QueryIntent
from core.settings import GatewaySettings, OrchestratorSettings


class ChatRequest(BaseModel):
    """HTTP request payload for the gateway chat endpoint."""

    message: str
    session_id: str | None = None
    stream: bool = False


class IndexRequest(BaseModel):
    """Kick off a full or incremental indexing run."""

    mode: Literal["full", "incremental"] = "incremental"


class ChatEvent(BaseModel):
    """Transport-neutral event used for SSE and WebSocket streams."""

    type: Literal["routing", "agent_result", "answer", "done"]
    correlation_id: str
    data: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """Non-streaming response mirrors the streaming event protocol."""

    session_id: str
    correlation_id: str
    routing: dict[str, Any]
    agent_results: list[dict[str, Any]] = Field(default_factory=list)
    answer: str
    done: dict[str, Any]


class IndexJobAccepted(BaseModel):
    """Response body when an index job is created."""

    job_id: str


class IndexJobStatus(BaseModel):
    """Current lifecycle state for an in-process index job."""

    job_id: str
    status: Literal["pending", "running", "done", "failed"]
    mode: Literal["full", "incremental"]
    correlation_id: str
    report: dict[str, Any] | None = None
    error: str | None = None


@dataclass(slots=True)
class IndexJobRecord:
    """Mutable in-process registry entry for one index job."""

    job_id: str
    mode: Literal["full", "incremental"]
    correlation_id: str
    status: Literal["pending", "running", "done", "failed"] = "pending"
    report: dict[str, Any] | None = None
    error: str | None = None

    def to_status(self) -> IndexJobStatus:
        return IndexJobStatus(
            job_id=self.job_id,
            status=self.status,
            mode=self.mode,
            correlation_id=self.correlation_id,
            report=self.report,
            error=self.error,
        )


class OrchestratorGatewayClient(Protocol):
    async def get_conversation_context(
        self,
        session_id: str,
        token_budget: int,
        *,
        correlation_id: str,
    ) -> ConversationContext: ...

    async def handle_query(
        self,
        query: str,
        session_id: str,
        *,
        correlation_id: str,
    ) -> dict[str, Any]: ...

    async def analyze_query(
        self,
        query: str,
        context: ConversationContext,
        *,
        correlation_id: str,
    ) -> QueryIntent: ...

    async def route_to_agents(
        self,
        intent: QueryIntent,
        *,
        correlation_id: str,
    ) -> ExecutionPlan: ...

    async def synthesize_response(
        self,
        query: str,
        agent_outputs: dict[str, Any],
        context: ConversationContext,
        *,
        correlation_id: str,
    ) -> str: ...


class GraphQueryGatewayClient(Protocol):
    async def find_entity(
        self,
        name: str,
        entity_type: str | None = None,
        *,
        correlation_id: str,
    ) -> Any: ...


class CodeAnalystGatewayClient(Protocol):
    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        correlation_id: str,
    ) -> Any: ...


class IndexerGatewayClient(Protocol):
    async def index_repository(
        self,
        repo_url: str | None = None,
        *,
        correlation_id: str,
    ) -> Any: ...


class GatewaySpecialistClients(Protocol):
    graph_query: GraphQueryGatewayClient
    code_analyst: CodeAnalystGatewayClient
    indexer: IndexerGatewayClient


@dataclass(slots=True)
class GatewayDependencies:
    """Gateway runtime collaborators."""

    orchestrator: OrchestratorGatewayClient
    specialists: GatewaySpecialistClients
    gateway_settings: GatewaySettings
    orchestrator_settings: OrchestratorSettings


@dataclass(slots=True)
class ChatRunResult:
    """Completed chat result used for non-stream responses."""

    session_id: str
    correlation_id: str
    routing: dict[str, Any]
    agent_results: list[dict[str, Any]]
    answer: str
    done: dict[str, Any]


def new_session_id() -> str:
    return str(uuid.uuid4())


def new_job_id() -> str:
    return str(uuid.uuid4())


@dataclass(slots=True)
class ChatGatewayService:
    """Drive the chat flow while emitting transport-neutral events."""

    deps: GatewayDependencies

    async def run(self, message: str, session_id: str, correlation_id: str) -> ChatRunResult:
        routing: dict[str, Any] = {}
        agent_results: list[dict[str, Any]] = []
        answer_parts: list[str] = []
        done_event: dict[str, Any] = {}

        async for event in self.stream(message, session_id, correlation_id):
            if event.type == "routing":
                routing = event.data
            elif event.type == "agent_result":
                agent_results.append(event.data)
            elif event.type == "answer":
                chunk = event.data.get("chunk")
                if isinstance(chunk, str):
                    answer_parts.append(chunk)
            elif event.type == "done":
                done_event = event.data

        return ChatRunResult(
            session_id=session_id,
            correlation_id=correlation_id,
            routing=routing,
            agent_results=agent_results,
            answer="".join(answer_parts),
            done=done_event,
        )

    async def stream(
        self,
        message: str,
        session_id: str,
        correlation_id: str,
    ) -> AsyncIterator[ChatEvent]:
        started_at = time.perf_counter()
        settings = self.deps.gateway_settings
        degraded = False
        cached = False
        answer = ""
        routing_mode = "orchestrator"
        tokens: dict[str, Any] = {
            "total": 0,
            "prompt": 0,
            "completion": 0,
            "llm_calls": 0,
            "by_purpose": {},
        }

        try:
            payload = await self.deps.orchestrator.handle_query(
                message, session_id, correlation_id=correlation_id
            )
            answer = str(payload.get("answer", ""))
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                cached = bool(metadata.get("cached", False))
                degraded = bool(metadata.get("degraded", False))
                routing_mode = str(metadata.get("routing_mode", routing_mode))
                maybe_tokens = metadata.get("tokens")
                if isinstance(maybe_tokens, dict):
                    tokens = maybe_tokens
        except AgentUnavailableError:
            raise
        except Exception as exc:
            degraded = True
            answer = f"Degraded response: failed to run orchestrator ({exc})."

        # Keep the SSE event types stable for clients/tests.
        yield ChatEvent(
            type="routing",
            correlation_id=correlation_id,
            data={
                "mode": "orchestrator",
                "routing_mode": routing_mode,
                "agents": ["orchestrator"],
                "cached": cached,
                "degraded": degraded,
            },
        )
        yield ChatEvent(
            type="agent_result",
            correlation_id=correlation_id,
            data={
                "agent": "orchestrator",
                "ok": not degraded,
                "cached": cached,
                "degraded": degraded,
            },
        )

        for chunk in _chunk_text(answer, settings.answer_chunk_chars):
            yield ChatEvent(
                type="answer",
                correlation_id=correlation_id,
                data={"chunk": chunk},
            )

        yield ChatEvent(
            type="done",
            correlation_id=correlation_id,
            data=_done_payload(
                correlation_id=correlation_id,
                cached=cached,
                degraded=degraded,
                started_at=started_at,
                routing_mode=routing_mode,
                tokens=tokens,
            ),
        )


@dataclass(slots=True)
class IndexJobRegistry:
    """In-process background-job registry for index runs."""

    jobs: dict[str, IndexJobRecord] = field(default_factory=dict)

    def create(
        self,
        *,
        mode: Literal["full", "incremental"],
        correlation_id: str,
    ) -> IndexJobRecord:
        record = IndexJobRecord(
            job_id=new_job_id(),
            mode=mode,
            correlation_id=correlation_id,
        )
        self.jobs[record.job_id] = record
        return record

    def get(self, job_id: str) -> IndexJobRecord | None:
        return self.jobs.get(job_id)


def _chunk_text(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        return [text]
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] or [""]


def _done_payload(
    *,
    correlation_id: str,
    cached: bool,
    degraded: bool,
    started_at: float,
    routing_mode: str,
    tokens: dict[str, Any],
) -> dict[str, Any]:
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    return {
        "correlation_id": correlation_id,
        "cached": cached,
        "degraded": degraded,
        "routing_mode": routing_mode,
        "latency_ms": latency_ms,
        "tokens": tokens,
    }


def _default_snippet_range() -> tuple[int, int]:
    return (1, 120)


async def _execute_plan_stream(
    plan: ExecutionPlan,
    *,
    message: str,
    context: ConversationContext | None,
    clients: GatewaySpecialistClients,
    settings: OrchestratorSettings,
    correlation_id: str,
) -> AsyncIterator[AgentOutput]:
    _ = message, context, correlation_id
    graph_entities: list[dict[str, Any]] = []
    graph_available = True

    async def run_graph() -> AgentOutput:
        nonlocal graph_entities, graph_available
        try:
            entities = plan.intent.entities or []
            if not entities:
                return AgentOutput(
                    agent="graph_query",
                    ok=True,
                    output={"entities": [], "queried_entities": []},
                )
            tasks = [
                clients.graph_query.find_entity(name=entity, correlation_id=correlation_id)
                for entity in entities
            ]
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=settings.graph_query_timeout_s,
            )
            graph_entities = [dict(result) for result in results if result is not None]
            return AgentOutput(
                agent="graph_query",
                ok=True,
                output={"entities": graph_entities, "queried_entities": entities},
            )
        except Exception as exc:
            graph_available = False
            return AgentOutput(
                agent="graph_query",
                ok=False,
                degraded_note="graph_query unavailable; using raw-file fallback",
                error=str(exc),
            )

    async def run_code() -> AgentOutput:
        try:
            entities = plan.intent.entities or []
            if not entities:
                return AgentOutput(
                    agent="code_analyst",
                    ok=True,
                    output={"snippets": []},
                )
            start, end = _default_snippet_range()
            tasks: list[asyncio.Task[Any]] = []
            if graph_available and graph_entities:
                for i, entity in enumerate(entities):
                    hit = graph_entities[i] if i < len(graph_entities) else {}
                    file_path = (
                        hit.get("file_path")
                        or hit.get("filePath")
                        or (entity if isinstance(entity, str) else None)
                    )
                    line_start = hit.get("line_start") or hit.get("lineStart") or start
                    line_end = hit.get("line_end") or hit.get("lineEnd") or end
                    tasks.append(
                        asyncio.create_task(
                            clients.code_analyst.get_code_snippet(
                                file_path=str(file_path) if file_path else str(entity),
                                line_start=int(line_start) if line_start else start,
                                line_end=int(line_end) if line_end else end,
                                correlation_id=correlation_id,
                            )
                        )
                    )
            else:
                for entity in entities:
                    tasks.append(
                        asyncio.create_task(
                            clients.code_analyst.get_code_snippet(
                                file_path=entity,
                                line_start=start,
                                line_end=end,
                                correlation_id=correlation_id,
                            )
                        )
                    )
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=settings.code_analyst_timeout_s,
            )
            return AgentOutput(
                agent="code_analyst",
                ok=True,
                output={"snippets": results},
            )
        except Exception as exc:
            return AgentOutput(
                agent="code_analyst",
                ok=False,
                degraded_note="code_analyst unavailable; returning graph facts only",
                error=str(exc),
            )

    async def run_index() -> AgentOutput:
        try:
            report = await asyncio.wait_for(
                clients.indexer.index_repository(
                    repo_url=None,
                    correlation_id=correlation_id,
                ),
                timeout=settings.indexer_timeout_s,
            )
            return AgentOutput(agent="indexer", ok=True, output=report)
        except Exception as exc:
            return AgentOutput(
                agent="indexer",
                ok=False,
                degraded_note="indexer unavailable",
                error=str(exc),
            )

    runners: dict[AgentName, Any] = {
        "graph_query": run_graph,
        "code_analyst": run_code,
        "indexer": run_index,
    }

    for phase in plan.phases:
        tasks = {
            asyncio.create_task(runners[agent]()): agent
            for agent in phase
            if agent in runners
        }
        for task in asyncio.as_completed(tasks):
            yield await task
