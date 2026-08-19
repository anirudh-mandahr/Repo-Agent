"""Gateway-facing chat and index orchestration models/services."""

from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from core.exceptions import (
    AgentUnavailableError,
    GraphLookupError,
    RoutingError,
    SchemaValidationError,
    SynthesisError,
)
from core.memory import ConversationContext
from core.observability.ledger import empty_token_totals
from core.orchestration.models import ExecutionPlan, QueryIntent
from core.settings import GatewaySettings, OrchestratorSettings

DEFAULT_MAX_MESSAGE_LENGTH = 8000
DEFAULT_RATE_LIMIT_REQUESTS = 60
DEFAULT_RATE_LIMIT_WINDOW_S = 60.0
_PROPAGATED_CHAT_ERRORS = (
    AgentUnavailableError,
    GraphLookupError,
    RoutingError,
    SchemaValidationError,
    SynthesisError,
)


class ChatRequest(BaseModel):
    """HTTP request payload for the gateway chat endpoint."""

    message: str = Field(..., max_length=DEFAULT_MAX_MESSAGE_LENGTH)
    session_id: str | None = None
    stream: bool = False


class IndexRequest(BaseModel):
    """Kick off a full rebuild or a hash-skipping incremental index."""

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
    tools_invoked: list[str] = Field(default_factory=list)


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
        """Return a serializable snapshot of this job record.

        Returns:
            The current job status payload.
        """
        return IndexJobStatus(
            job_id=self.job_id,
            status=self.status,
            mode=self.mode,
            correlation_id=self.correlation_id,
            report=self.report,
            error=self.error,
        )


class OrchestratorGatewayClient(Protocol):
    """Orchestrator MCP client used by the gateway."""

    async def get_conversation_context(
        self,
        session_id: str,
        token_budget: int,
        *,
        correlation_id: str,
    ) -> ConversationContext:
        """Load conversation memory for ``session_id``.

        Args:
            session_id: Conversation id.
            token_budget: Maximum tokens of context to return.
            correlation_id: Request correlation id.

        Returns:
            Conversation summary plus recent turns.
        """
        ...

    async def handle_query(
        self,
        query: str,
        session_id: str,
        *,
        correlation_id: str,
    ) -> dict[str, Any]:
        """Run the full orchestrator loop.

        Args:
            query: User question.
            session_id: Conversation id.
            correlation_id: Request correlation id.

        Returns:
            Mapping with ``answer`` and ``metadata``.
        """
        ...

    async def analyze_query(
        self,
        query: str,
        context: ConversationContext,
        *,
        correlation_id: str,
    ) -> QueryIntent:
        """Classify a query into a routing intent.

        Args:
            query: User question.
            context: Prior conversation context.
            correlation_id: Request correlation id.

        Returns:
            Structured routing intent.
        """
        ...

    async def route_to_agents(
        self,
        intent: QueryIntent,
        *,
        correlation_id: str,
    ) -> ExecutionPlan:
        """Turn an intent into a flat specialist execution plan.

        Args:
            intent: Routed query intent.
            correlation_id: Request correlation id.

        Returns:
            Execution plan with ``agents`` (not sequenced phases).
        """
        ...

    async def synthesize_response(
        self,
        query: str,
        agent_outputs: dict[str, Any],
        context: ConversationContext,
        *,
        correlation_id: str,
    ) -> str:
        """Merge specialist outputs into a final answer.

        Args:
            query: User question.
            agent_outputs: Per-agent payloads.
            context: Prior conversation context.
            correlation_id: Request correlation id.

        Returns:
            Final answer text.
        """
        ...


class GraphQueryGatewayClient(Protocol):
    """Graph Query MCP client used by the gateway."""

    async def find_entity(
        self,
        name: str,
        entity_type: str | None = None,
        *,
        correlation_id: str,
    ) -> Any:
        """Look up an entity by name.

        Args:
            name: Entity name.
            entity_type: Optional label filter.
            correlation_id: Request correlation id.

        Returns:
            Match payload.
        """
        ...


class CodeAnalystGatewayClient(Protocol):
    """Code Analyst MCP client used by the gateway."""

    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        correlation_id: str,
    ) -> Any:
        """Fetch a numbered source snippet.

        Args:
            qualified_name: Optional graph coordinate.
            file_path: Optional repo-relative path.
            line_start: Inclusive start line.
            line_end: Inclusive end line.
            correlation_id: Request correlation id.

        Returns:
            Snippet payload.
        """
        ...


class IndexerGatewayClient(Protocol):
    """Indexer MCP client used by the gateway."""

    async def index_repository(
        self,
        repo_url: str | None = None,
        *,
        mode: Literal["full", "incremental"] = "incremental",
        correlation_id: str,
    ) -> Any:
        """Trigger a repository index.

        Args:
            repo_url: Optional clone URL override.
            mode: ``incremental`` skips unchanged file hashes; ``full`` re-parses
                every file. Must be forwarded to the indexer (never ignored).
            correlation_id: Request correlation id.

        Returns:
            Index report payload.
        """
        ...


class GatewaySpecialistClients(Protocol):
    """Specialist MCP clients used by gateway index and chat helpers."""

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
    tools_invoked: list[str] = field(default_factory=list)


def new_session_id() -> str:
    """Allocate a new conversation session id.

    Returns:
        A UUID4 string.
    """
    return str(uuid.uuid4())


def new_job_id() -> str:
    """Allocate a new index job id.

    Returns:
        A UUID4 string.
    """
    return str(uuid.uuid4())


@dataclass(slots=True)
class ChatGatewayService:
    """Drive the chat flow while emitting transport-neutral events."""

    deps: GatewayDependencies

    async def run(self, message: str, session_id: str, correlation_id: str) -> ChatRunResult:
        """Run a full chat turn and collect streaming events into one response.

        Args:
            message: User question.
            session_id: Conversation id.
            correlation_id: Request correlation id.

        Returns:
            Combined routing, agent, answer, and done payloads.
        """
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
            tools_invoked=expand_tools_invoked(list(routing.get("tools_invoked") or [])),
        )

    async def stream(
        self,
        message: str,
        session_id: str,
        correlation_id: str,
    ) -> AsyncIterator[ChatEvent]:
        """Stream transport-neutral chat events for one user message.

        Emits ``routing``, ``agent_result``, ``answer`` chunks, and ``done`` in
        that order so SSE and WebSocket clients share one protocol.

        Args:
            message: User question.
            session_id: Conversation id.
            correlation_id: Request correlation id.

        Yields:
            Chat events consumed by the gateway HTTP/SSE/WebSocket layer.
        """
        started_at = time.perf_counter()
        settings = self.deps.gateway_settings
        streamed_answer = False
        event_queue: asyncio.Queue[ChatEvent | object] = asyncio.Queue()
        _done_sentinel = object()
        streamed_routing = False

        async def on_token(chunk: str) -> None:
            nonlocal streamed_answer
            if not chunk:
                return
            streamed_answer = True
            await event_queue.put(
                ChatEvent(
                    type="answer",
                    correlation_id=correlation_id,
                    data={"chunk": chunk},
                )
            )

        async def on_event(event_type: str, data: dict[str, Any]) -> None:
            nonlocal streamed_routing
            if event_type == "routing":
                streamed_routing = True
                payload = dict(data)
                payload.setdefault("mode", "orchestrator")
                await event_queue.put(
                    ChatEvent(type="routing", correlation_id=correlation_id, data=payload)
                )
            elif event_type == "agent_result":
                await event_queue.put(
                    ChatEvent(type="agent_result", correlation_id=correlation_id, data=data)
                )

        handle = self.deps.orchestrator.handle_query
        call_kwargs: dict[str, Any] = {"correlation_id": correlation_id}
        if _accepts_kwarg(handle, "on_token"):
            call_kwargs["on_token"] = on_token
        if _accepts_kwarg(handle, "on_event"):
            call_kwargs["on_event"] = on_event

        async def _run_handle() -> dict[str, Any]:
            try:
                return await handle(message, session_id, **call_kwargs)
            finally:
                await event_queue.put(_done_sentinel)

        live_events = _accepts_kwarg(handle, "on_token") or _accepts_kwarg(handle, "on_event")
        if live_events:
            task = asyncio.create_task(_run_handle())
            while True:
                item = await event_queue.get()
                if item is _done_sentinel:
                    break
                if isinstance(item, ChatEvent):
                    yield item
            try:
                payload = await task
            except _PROPAGATED_CHAT_ERRORS:
                raise
            except Exception as exc:
                payload = {
                    "answer": f"Degraded response: failed to run orchestrator ({exc}).",
                    "metadata": {"degraded": True},
                }
            parsed = _payload_fields(payload)
            if not streamed_routing:
                async for event in _emit_routing_and_agent(
                    correlation_id=correlation_id,
                    cached=parsed.cached,
                    degraded=parsed.degraded,
                    routing_mode=parsed.routing_mode,
                    tools_invoked=parsed.tools_invoked,
                    metadata=parsed.metadata,
                ):
                    yield event
            if not streamed_answer:
                for chunk in _chunk_text(parsed.answer, settings.answer_chunk_chars):
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
                    cached=parsed.cached,
                    degraded=parsed.degraded,
                    started_at=started_at,
                    routing_mode=parsed.routing_mode,
                    tokens=parsed.tokens,
                    evidence_only=parsed.evidence_only,
                    degraded_reason=parsed.degraded_reason,
                    prompt_truncated=parsed.prompt_truncated,
                ),
            )
            return

        # Non-streaming orchestrator client (MCP): wait, then emit events.
        degraded = False
        cached = False
        answer = ""
        routing_mode = "orchestrator"
        tools_invoked: list[str] = []
        metadata: dict[str, Any] | None = None
        evidence_only = False
        degraded_reason: str | None = None
        prompt_truncated: dict[str, Any] | None = None
        tokens: dict[str, Any] = empty_token_totals()

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
                maybe_tools = metadata.get("tools_invoked")
                if isinstance(maybe_tools, list):
                    tools_invoked = [str(item) for item in maybe_tools]
                evidence_only = bool(metadata.get("evidence_only", False))
                maybe_reason = metadata.get("degraded_reason")
                if maybe_reason is not None:
                    degraded_reason = str(maybe_reason)
                maybe_truncated = metadata.get("prompt_truncated")
                if isinstance(maybe_truncated, dict):
                    prompt_truncated = maybe_truncated
        except _PROPAGATED_CHAT_ERRORS:
            raise
        except Exception as exc:
            degraded = True
            answer = f"Degraded response: failed to run orchestrator ({exc})."

        async for event in _emit_routing_and_agent(
            correlation_id=correlation_id,
            cached=cached,
            degraded=degraded,
            routing_mode=routing_mode,
            tools_invoked=tools_invoked,
            metadata=metadata if isinstance(metadata, dict) else None,
        ):
            yield event
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
                evidence_only=evidence_only,
                degraded_reason=degraded_reason,
                prompt_truncated=prompt_truncated,
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
        """Register a new in-process index job.

        Args:
            mode: Full rebuild or incremental update.
            correlation_id: Request correlation id stored on the job.

        Returns:
            The created job record.
        """
        record = IndexJobRecord(
            job_id=new_job_id(),
            mode=mode,
            correlation_id=correlation_id,
        )
        self.jobs[record.job_id] = record
        return record

    def get(self, job_id: str) -> IndexJobRecord | None:
        """Look up a job by id.

        Args:
            job_id: Identifier returned when the job was created.

        Returns:
            The job record, or ``None`` when unknown.
        """
        return self.jobs.get(job_id)


def _accepts_kwarg(fn: object, name: str) -> bool:
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()):
        return True
    return name in sig.parameters


@dataclass
class _ParsedPayload:
    answer: str
    metadata: dict[str, Any] | None
    cached: bool
    degraded: bool
    routing_mode: str
    tools_invoked: list[str]
    tokens: dict[str, Any]
    evidence_only: bool
    degraded_reason: str | None
    prompt_truncated: dict[str, Any] | None


def _payload_fields(payload: Any) -> _ParsedPayload:
    if not isinstance(payload, dict) and hasattr(payload, "answer"):
        payload = {
            "answer": getattr(payload, "answer", ""),
            "metadata": getattr(payload, "metadata", {}),
        }
    if not isinstance(payload, dict):
        payload = {"answer": str(payload), "metadata": {}}
    metadata = payload.get("metadata")
    meta = metadata if isinstance(metadata, dict) else {}
    tokens_raw = meta.get("tokens")
    tokens: dict[str, Any] = (
        tokens_raw if isinstance(tokens_raw, dict) else empty_token_totals()
    )
    tools = meta.get("tools_invoked")
    tools_invoked = [str(item) for item in tools] if isinstance(tools, list) else []
    truncated = meta.get("prompt_truncated")
    reason = meta.get("degraded_reason")
    return _ParsedPayload(
        answer=str(payload.get("answer", "")),
        metadata=meta or None,
        cached=bool(meta.get("cached", False)),
        degraded=bool(meta.get("degraded", False)),
        routing_mode=str(meta.get("routing_mode") or "orchestrator"),
        tools_invoked=tools_invoked,
        tokens=tokens,
        evidence_only=bool(meta.get("evidence_only", False)),
        degraded_reason=str(reason) if reason is not None else None,
        prompt_truncated=truncated if isinstance(truncated, dict) else None,
    )


async def _emit_routing_and_agent(
    *,
    correlation_id: str,
    cached: bool,
    degraded: bool,
    routing_mode: str,
    tools_invoked: list[str],
    metadata: dict[str, Any] | None,
) -> AsyncIterator[ChatEvent]:
    specialist_agents = _agents_from_tools(tools_invoked)
    breaker_state: dict[str, Any] = {}
    plan_iterations: list[dict[str, Any]] = []
    if isinstance(metadata, dict):
        maybe_breakers = metadata.get("circuit_breakers")
        if isinstance(maybe_breakers, dict):
            breaker_state = maybe_breakers
        maybe_iterations = metadata.get("plan_iterations")
        if isinstance(maybe_iterations, list):
            plan_iterations = [item for item in maybe_iterations if isinstance(item, dict)]

    if plan_iterations:
        for item in plan_iterations:
            agents = item.get("agents")
            if not isinstance(agents, list) or not agents:
                agents = specialist_agents or ["orchestrator"]
            yield ChatEvent(
                type="routing",
                correlation_id=correlation_id,
                data={
                    "mode": "orchestrator",
                    "routing_mode": str(item.get("routing_mode") or routing_mode),
                    "iteration": item.get("iteration"),
                    "agents": agents,
                    "tools_invoked": item.get("tools_invoked") or tools_invoked,
                    "search_terms": item.get("search_terms") or [],
                    "sufficient": item.get("sufficient"),
                    "reason": item.get("reason"),
                    "refinement": item.get("refinement"),
                    "cached": cached,
                    "degraded": degraded,
                    "circuit_breakers": breaker_state,
                },
            )
    else:
        yield ChatEvent(
            type="routing",
            correlation_id=correlation_id,
            data={
                "mode": "orchestrator",
                "routing_mode": routing_mode,
                "agents": specialist_agents or ["orchestrator"],
                "tools_invoked": tools_invoked,
                "cached": cached,
                "degraded": degraded,
                "circuit_breakers": breaker_state,
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


def _agents_from_tools(tools_invoked: list[str]) -> list[str]:
    agents: list[str] = []
    for tool in tools_invoked:
        if "." not in tool:
            continue
        agent = tool.split(".", 1)[0]
        if agent and agent not in agents:
            agents.append(agent)
    return agents


def expand_tools_invoked(tools_invoked: list[str]) -> list[str]:
    """Include both `agent.tool` and bare `tool` names for HTTP clients.
    
    Args:
        tools_invoked: list[str].

    Returns:
        list[str].
    """
    expanded: list[str] = []
    seen: set[str] = set()
    for item in tools_invoked:
        candidates = [item]
        if "." in item:
            candidates.append(item.split(".", 1)[1])
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.add(candidate)
                expanded.append(candidate)
    return expanded


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
    evidence_only: bool = False,
    degraded_reason: str | None = None,
    prompt_truncated: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latency_ms = int((time.perf_counter() - started_at) * 1000)
    payload: dict[str, Any] = {
        "correlation_id": correlation_id,
        "cached": cached,
        "degraded": degraded,
        "routing_mode": routing_mode,
        "latency_ms": latency_ms,
        "tokens": tokens,
    }
    if evidence_only:
        payload["evidence_only"] = True
    if degraded_reason is not None:
        payload["degraded_reason"] = degraded_reason
    if prompt_truncated is not None:
        payload["prompt_truncated"] = prompt_truncated
    return payload


@dataclass
class SlidingWindowRateLimiter:
    """In-process sliding-window limiter keyed by client identity."""

    max_requests: int
    window_s: float
    _hits: dict[str, deque[float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Return whether ``key`` may proceed and record the attempt when allowed.

        Args:
            key: Client identity (API key or peer address).
            now: Optional monotonic timestamp for tests.

        Returns:
            ``True`` when the request is under the configured cap.
        """
        if self.max_requests <= 0:
            return True
        timestamp = time.monotonic() if now is None else now
        with self._lock:
            bucket = self._hits.setdefault(key, deque())
            cutoff = timestamp - self.window_s
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                return False
            bucket.append(timestamp)
            return True

