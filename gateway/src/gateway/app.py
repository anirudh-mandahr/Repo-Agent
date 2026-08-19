"""FastAPI gateway with chat, indexing, health, graph, and metrics endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any, Literal, cast

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.exceptions import WebSocketException
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel
from sse_starlette import EventSourceResponse

from core.exceptions import (
    AgentError,
    AgentUnavailableError,
    CircuitBreakerOpenError,
    GraphLookupError,
    RoutingError,
    SchemaValidationError,
    SynthesisError,
)
from core.gateway import (
    ChatEvent,
    ChatGatewayService,
    ChatRequest,
    ChatResponse,
    GatewayDependencies,
    IndexJobAccepted,
    IndexJobRegistry,
    IndexJobStatus,
    IndexRequest,
    SlidingWindowRateLimiter,
    new_session_id,
)
from core.health import AggregateHealth, HealthStatus
from core.logging import bind_correlation_id, configure_logging, get_correlation_id, get_logger
from core.mcp.auth import compare_secrets
from core.observability.metrics import METRICS_CONTENT_TYPE, record_http_result, render_metrics
from core.observability.tracing import configure_tracing, start_span
from core.querying.service import GraphStatistics
from core.settings import GatewaySettings, OrchestratorSettings
from gateway.mcp_client import (
    GatewayOrchestratorClient,
    GatewaySpecialistClients,
    build_gateway_pool,
    call_agent_health,
    call_tool_json,
)

settings = GatewaySettings.from_env()
configure_logging(settings.log_level)
log = get_logger("gateway")

_PUBLIC_PATHS = frozenset(
    {
        "/health",
        "/api/agents/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)
_OPENAPI_TAGS = [
    {"name": "chat", "description": "Natural-language chat over the indexed repository."},
    {"name": "index", "description": "Background indexing jobs."},
    {"name": "ops", "description": "Health, metrics, and operational status."},
    {"name": "graph", "description": "Read-only knowledge-graph statistics."},
]


class GatewayErrorBody(BaseModel):
    """JSON body returned by typed gateway exception handlers."""

    detail: str
    correlation_id: str | None = None
    agent: str | None = None
    degraded: bool | None = None
    tools_invoked: list[str] | None = None


_ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    401: {"model": GatewayErrorBody, "description": "Invalid API key"},
    422: {"model": GatewayErrorBody, "description": "Routing or schema validation failed"},
    429: {"model": GatewayErrorBody, "description": "Rate limit exceeded"},
    503: {
        "model": GatewayErrorBody,
        "description": (
            "Agent unavailable, circuit breaker open, graph lookup failed, or synthesis failed"
        ),
    },
}


def build_dependencies(
    settings: GatewaySettings,
    pool: Any | None = None,
) -> GatewayDependencies:
    mcp_pool = pool or build_gateway_pool(settings)
    return GatewayDependencies(
        orchestrator=GatewayOrchestratorClient(settings, mcp_pool),
        specialists=cast(Any, GatewaySpecialistClients(settings, mcp_pool)),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings.from_env(),
    )


def create_app(
    settings: GatewaySettings | None = None,
    *,
    deps: GatewayDependencies | None = None,
) -> FastAPI:
    gateway_settings = settings or GatewaySettings.from_env()
    configure_logging(gateway_settings.log_level)
    configure_tracing("gateway")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            pool = getattr(app.state, "mcp_pool", None)
            if pool is not None:
                await pool.aclose()

    app = FastAPI(
        title="FastAPI repo chat gateway",
        description=(
            "HTTP, SSE, and WebSocket gateway in front of five MCP agents that "
            "index and answer questions about a Python repository stored in Neo4j."
        ),
        version="0.1.0",
        lifespan=lifespan,
        openapi_tags=_OPENAPI_TAGS,
    )
    app.state.settings = gateway_settings
    pool = build_gateway_pool(gateway_settings)
    app.state.mcp_pool = pool
    app.state.gateway_deps = deps or build_dependencies(gateway_settings, pool)
    app.state.chat_service = ChatGatewayService(app.state.gateway_deps)
    app.state.job_registry = IndexJobRegistry()
    app.state.rate_limiter = SlidingWindowRateLimiter(
        max_requests=gateway_settings.rate_limit_requests,
        window_s=gateway_settings.rate_limit_window_s,
    )

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema
        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
            tags=_OPENAPI_TAGS,
        )
        components = schema.setdefault("components", {})
        components["securitySchemes"] = {
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "Shared gateway API key.",
            }
        }
        schema["security"] = [{"ApiKeyAuth": []}]
        app.openapi_schema = schema
        return app.openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    @app.middleware("http")
    async def rate_limit_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        limiter: SlidingWindowRateLimiter = request.app.state.rate_limiter
        key = request.headers.get("x-api-key") or (
            request.client.host if request.client is not None else "anon"
        )
        if not limiter.allow(key):
            response = JSONResponse(status_code=429, content={"detail": "rate limit exceeded"})
            response.headers["x-correlation-id"] = get_correlation_id()
            return response
        return await call_next(request)

    @app.middleware("http")
    async def api_key_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        api_key = app.state.settings.api_key
        if path in _PUBLIC_PATHS or api_key is None:
            return await call_next(request)
        provided = request.headers.get("x-api-key") or ""
        if not compare_secrets(provided, api_key.get_secret_value()):
            response = JSONResponse(status_code=401, content={"detail": "invalid api key"})
            response.headers["x-correlation-id"] = get_correlation_id()
            return response
        return await call_next(request)

    @app.middleware("http")
    async def correlation_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        correlation_id = bind_correlation_id(request.headers.get("x-correlation-id"))
        started = perf_counter()
        try:
            with start_span(
                "gateway.http",
                correlation_id=correlation_id,
                http_path=request.url.path,
                http_method=request.method,
            ):
                response = await call_next(request)
        except Exception:
            record_http_result(
                "gateway",
                request.url.path,
                status_code=500,
                duration_s=perf_counter() - started,
            )
            raise
        record_http_result(
            "gateway",
            request.url.path,
            status_code=response.status_code,
            duration_s=perf_counter() - started,
        )
        response.headers["x-correlation-id"] = correlation_id
        return response

    @app.exception_handler(AgentUnavailableError)
    async def handle_agent_unavailable(
        request: Request,
        exc: AgentUnavailableError,
    ) -> JSONResponse:
        log.warning(
            "gateway.agent_unavailable",
            path=request.url.path,
            agent=exc.agent,
            correlation_id=exc.correlation_id,
        )
        return _agent_error_response(exc, status_code=503, degraded=True)

    @app.exception_handler(CircuitBreakerOpenError)
    async def handle_circuit_open(
        request: Request,
        exc: CircuitBreakerOpenError,
    ) -> JSONResponse:
        log.warning(
            "gateway.circuit_open",
            path=request.url.path,
            agent=exc.agent,
            correlation_id=exc.correlation_id,
        )
        return _agent_error_response(exc, status_code=503, degraded=True)

    @app.exception_handler(GraphLookupError)
    async def handle_graph_lookup_error(
        request: Request,
        exc: GraphLookupError,
    ) -> JSONResponse:
        log.warning(
            "gateway.graph_lookup_error",
            path=request.url.path,
            agent=exc.agent,
            correlation_id=exc.correlation_id,
        )
        return _agent_error_response(exc, status_code=503, degraded=True)

    @app.exception_handler(SynthesisError)
    async def handle_synthesis_error(
        request: Request,
        exc: SynthesisError,
    ) -> JSONResponse:
        log.warning(
            "gateway.synthesis_error",
            path=request.url.path,
            agent=exc.agent,
            correlation_id=exc.correlation_id,
        )
        return _agent_error_response(exc, status_code=503, degraded=True)

    @app.exception_handler(RoutingError)
    async def handle_routing_error(request: Request, exc: RoutingError) -> JSONResponse:
        log.warning(
            "gateway.routing_error",
            path=request.url.path,
            agent=exc.agent,
            correlation_id=exc.correlation_id,
            error=str(exc),
        )
        return _agent_error_response(exc, status_code=422)

    @app.exception_handler(SchemaValidationError)
    async def handle_schema_validation_error(
        request: Request,
        exc: SchemaValidationError,
    ) -> JSONResponse:
        log.warning(
            "gateway.schema_validation_error",
            path=request.url.path,
            agent=exc.agent,
            correlation_id=exc.correlation_id,
            error=str(exc),
        )
        return _agent_error_response(exc, status_code=422)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        correlation_id = get_correlation_id()
        log.exception(
            "gateway.unhandled_exception",
            path=request.url.path,
            correlation_id=correlation_id,
            error=str(exc),
        )
        response = JSONResponse(
            status_code=500,
            content={"detail": "internal server error", "correlation_id": correlation_id},
        )
        response.headers["x-correlation-id"] = correlation_id
        return response

    @app.post(
        "/api/chat",
        response_model=ChatResponse,
        tags=["chat"],
        summary="Chat with the orchestrator",
        responses=_ERROR_RESPONSES,
    )
    async def chat(payload: ChatRequest, request: Request) -> Response:
        session_id = payload.session_id or new_session_id()
        correlation_id = get_correlation_id()
        service: ChatGatewayService = request.app.state.chat_service
        if payload.stream:
            return EventSourceResponse(
                _sse_events(service.stream(payload.message, session_id, correlation_id))
            )

        result = await service.run(payload.message, session_id, correlation_id)
        body = ChatResponse(
            session_id=result.session_id,
            correlation_id=result.correlation_id,
            routing=result.routing,
            agent_results=result.agent_results,
            answer=result.answer,
            done=result.done,
            tools_invoked=result.tools_invoked,
        )
        response = JSONResponse(content=body.model_dump(mode="json"))
        response.headers["x-correlation-id"] = correlation_id
        return response

    @app.websocket("/ws/chat")
    async def chat_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        api_key = app.state.settings.api_key
        if api_key is not None:
            provided = websocket.headers.get("x-api-key") or ""
            if not compare_secrets(provided, api_key.get_secret_value()):
                await websocket.close(code=1008, reason="invalid api key")
                return

        limiter: SlidingWindowRateLimiter = websocket.app.state.rate_limiter
        key = _ws_rate_limit_key(websocket)
        if not limiter.allow(key):
            await websocket.close(code=1008, reason="rate limit exceeded")
            return

        connection_session_id = new_session_id()
        service: ChatGatewayService = websocket.app.state.chat_service
        while True:
            try:
                payload = ChatRequest.model_validate(await websocket.receive_json())
            except WebSocketException:
                raise
            except Exception:
                break
            if not limiter.allow(key):
                await websocket.close(code=1008, reason="rate limit exceeded")
                return
            correlation_id = bind_correlation_id()
            session_id = payload.session_id or connection_session_id
            async for event in service.stream(payload.message, session_id, correlation_id):
                await websocket.send_json(_ws_event(event))

    @app.post(
        "/api/index",
        response_model=IndexJobAccepted,
        tags=["index"],
        summary="Start a background index job",
        responses=_ERROR_RESPONSES,
    )
    async def start_index_job(
        payload: IndexRequest,
        background_tasks: BackgroundTasks,
        request: Request,
    ) -> IndexJobAccepted:
        registry: IndexJobRegistry = request.app.state.job_registry
        record = registry.create(mode=payload.mode, correlation_id=get_correlation_id())
        background_tasks.add_task(_run_index_job, request.app, record.job_id)
        return IndexJobAccepted(job_id=record.job_id)

    @app.get(
        "/api/index/status/{job_id}",
        response_model=IndexJobStatus,
        tags=["index"],
        summary="Poll an index job",
    )
    async def index_status(job_id: str, request: Request) -> IndexJobStatus:
        registry: IndexJobRegistry = request.app.state.job_registry
        record = registry.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return record.to_status()

    @app.get(
        "/api/agents/health",
        response_model=AggregateHealth,
        response_model_exclude_none=True,
        tags=["ops"],
        summary="Aggregate MCP agent health",
    )
    async def agents_health(request: Request) -> AggregateHealth:
        gateway_settings_local: GatewaySettings = request.app.state.settings
        urls = gateway_settings_local.agent_urls()
        mcp_pool = getattr(request.app.state, "mcp_pool", None)

        async def _check(name: str, url: str) -> HealthStatus:
            try:
                return await asyncio.wait_for(
                    call_agent_health(
                        name,
                        url,
                        pool=mcp_pool,
                        timeout_s=gateway_settings_local.health_timeout_s,
                    ),
                    timeout=gateway_settings_local.health_timeout_s,
                )
            except TimeoutError:
                return HealthStatus(status="error", agent=name, detail="timed out")

        results = await asyncio.gather(*[_check(name, url) for name, url in urls.items()])
        agents = {
            name: status for (name, _url), status in zip(urls.items(), results, strict=True)
        }
        breaker_snapshots = mcp_pool.breakers.snapshot() if mcp_pool is not None else {}
        overall: Literal["ok", "degraded"] = (
            "ok"
            if all(item.status == "ok" for item in agents.values())
            and not any(item.state == "open" for item in breaker_snapshots.values())
            else "degraded"
        )
        return AggregateHealth(
            status=overall,
            agents=agents,
            circuit_breakers=breaker_snapshots,
        )

    @app.get(
        "/api/graph/statistics",
        response_model=GraphStatistics,
        tags=["graph"],
        summary="Graph label counts and index_version",
        responses=_ERROR_RESPONSES,
    )
    async def graph_statistics(request: Request) -> GraphStatistics:
        gateway_settings_local: GatewaySettings = request.app.state.settings
        mcp_pool = getattr(request.app.state, "mcp_pool", None)
        payload = await call_tool_json(
            gateway_settings_local.graph_query_url,
            "get_statistics",
            {},
            correlation_id=get_correlation_id(),
            timeout_s=gateway_settings_local.request_timeout_s,
            pool=mcp_pool,
            agent="graph_query",
        )
        return GraphStatistics.model_validate(payload)

    @app.get(
        "/health",
        response_model=AggregateHealth,
        response_model_exclude_none=True,
        tags=["ops"],
        summary="Alias for agents health",
    )
    async def health_alias(request: Request) -> AggregateHealth:
        return await agents_health(request)

    @app.get(
        "/metrics",
        tags=["ops"],
        summary="Prometheus metrics (requires API key when configured)",
        response_class=PlainTextResponse,
        responses={401: {"model": GatewayErrorBody, "description": "Invalid API key"}},
    )
    async def metrics() -> PlainTextResponse:
        return PlainTextResponse(content=render_metrics(), media_type=METRICS_CONTENT_TYPE)

    return app


def _agent_error_response(
    exc: AgentError,
    *,
    status_code: int,
    degraded: bool = False,
) -> JSONResponse:
    content: dict[str, Any] = {
        "detail": str(exc),
        "correlation_id": exc.correlation_id,
        "agent": exc.agent,
    }
    if degraded:
        content["degraded"] = True
        content["tools_invoked"] = []
    response = JSONResponse(status_code=status_code, content=content)
    response.headers["x-correlation-id"] = exc.correlation_id or get_correlation_id()
    return response


def _ws_rate_limit_key(websocket: WebSocket) -> str:
    header_key = websocket.headers.get("x-api-key") or ""
    if header_key:
        return header_key
    if websocket.client is not None:
        return websocket.client.host
    return "anon"


async def _sse_events(events: AsyncIterator[ChatEvent]) -> AsyncIterator[dict[str, Any]]:
    async for event in events:
        yield {
            "event": event.type,
            "data": json.dumps(
                {
                    "type": event.type,
                    "correlation_id": event.correlation_id,
                    **event.data,
                }
            ),
        }


def _ws_event(event: ChatEvent) -> dict[str, Any]:
    return {
        "type": event.type,
        "correlation_id": event.correlation_id,
        "data": event.data,
    }


async def _run_index_job(app: FastAPI, job_id: str) -> None:
    registry: IndexJobRegistry = app.state.job_registry
    record = registry.get(job_id)
    if record is None:
        return
    record.status = "running"
    bind_correlation_id(record.correlation_id)
    deps: GatewayDependencies = app.state.gateway_deps
    try:
        report = await deps.specialists.indexer.index_repository(
            repo_url=None,
            mode=record.mode,
            correlation_id=record.correlation_id,
        )
        record.status = "done"
        record.report = dict(report) if isinstance(report, dict) else {"result": report}
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)


app = create_app(settings)
