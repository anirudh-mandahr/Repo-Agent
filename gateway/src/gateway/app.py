"""FastAPI gateway with chat, indexing, health, and graph endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, cast

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.exceptions import WebSocketException
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

from core.exceptions import AgentUnavailableError, RoutingError
from core.gateway import (
    ChatEvent,
    ChatGatewayService,
    ChatRequest,
    ChatResponse,
    GatewayDependencies,
    IndexJobAccepted,
    IndexJobRegistry,
    IndexRequest,
    new_session_id,
)
from core.health import AggregateHealth, HealthStatus
from core.logging import bind_correlation_id, configure_logging, get_correlation_id, get_logger
from core.settings import GatewaySettings, OrchestratorSettings
from gateway.mcp_client import (
    GatewayOrchestratorClient,
    GatewaySpecialistClients,
    call_agent_health,
    call_tool_json,
)

settings = GatewaySettings.from_env()
configure_logging(settings.log_level)
log = get_logger("gateway")


def build_dependencies(settings: GatewaySettings) -> GatewayDependencies:
    return GatewayDependencies(
        orchestrator=GatewayOrchestratorClient(settings),
        specialists=cast(Any, GatewaySpecialistClients(settings)),
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
    app = FastAPI(title="FastAPI repo chat gateway", version="0.1.0")
    app.state.settings = gateway_settings
    app.state.gateway_deps = deps or build_dependencies(gateway_settings)
    app.state.chat_service = ChatGatewayService(app.state.gateway_deps)
    app.state.job_registry = IndexJobRegistry()

    @app.middleware("http")
    async def correlation_id_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        correlation_id = bind_correlation_id(request.headers.get("x-correlation-id"))
        try:
            response = await call_next(request)
        except Exception:
            raise
        response.headers["x-correlation-id"] = correlation_id
        return response

    @app.middleware("http")
    async def api_key_middleware(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        path = request.url.path
        api_key = app.state.settings.api_key
        if path == "/api/agents/health" or api_key is None:
            return await call_next(request)
        provided = request.headers.get("x-api-key")
        if provided != api_key.get_secret_value():
            response = JSONResponse(status_code=401, content={"detail": "invalid api key"})
            response.headers["x-correlation-id"] = get_correlation_id()
            return response
        return await call_next(request)

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
        response = JSONResponse(
            status_code=200,
            content={
                "detail": str(exc),
                "degraded": True,
                "correlation_id": exc.correlation_id,
                "agent": exc.agent,
            },
        )
        response.headers["x-correlation-id"] = exc.correlation_id or get_correlation_id()
        return response

    @app.exception_handler(RoutingError)
    async def handle_routing_error(request: Request, exc: RoutingError) -> JSONResponse:
        log.warning(
            "gateway.routing_error",
            path=request.url.path,
            agent=exc.agent,
            correlation_id=exc.correlation_id,
            error=str(exc),
        )
        response = JSONResponse(
            status_code=422,
            content={"detail": str(exc), "correlation_id": exc.correlation_id},
        )
        response.headers["x-correlation-id"] = exc.correlation_id or get_correlation_id()
        return response

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

    @app.post("/api/chat")
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
        )
        response = JSONResponse(content=body.model_dump(mode="json"))
        response.headers["x-correlation-id"] = correlation_id
        return response

    @app.websocket("/ws/chat")
    async def chat_ws(websocket: WebSocket) -> None:
        await websocket.accept()
        api_key = app.state.settings.api_key
        if api_key is not None and websocket.headers.get("x-api-key") != api_key.get_secret_value():
            await websocket.close(code=1008, reason="invalid api key")
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
            correlation_id = bind_correlation_id()
            session_id = payload.session_id or connection_session_id
            async for event in service.stream(payload.message, session_id, correlation_id):
                await websocket.send_json(_ws_event(event))

    @app.post("/api/index")
    async def start_index_job(
        payload: IndexRequest,
        background_tasks: BackgroundTasks,
        request: Request,
    ) -> IndexJobAccepted:
        registry: IndexJobRegistry = request.app.state.job_registry
        record = registry.create(mode=payload.mode, correlation_id=get_correlation_id())
        background_tasks.add_task(_run_index_job, request.app, record.job_id)
        return IndexJobAccepted(job_id=record.job_id)

    @app.get("/api/index/status/{job_id}")
    async def index_status(job_id: str, request: Request) -> JSONResponse:
        registry: IndexJobRegistry = request.app.state.job_registry
        record = registry.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JSONResponse(content=record.to_status().model_dump(mode="json"))

    @app.get("/api/agents/health")
    async def agents_health(request: Request) -> JSONResponse:
        gateway_settings: GatewaySettings = request.app.state.settings
        urls = gateway_settings.agent_urls()
        async def _check(name: str, url: str) -> HealthStatus:
            try:
                return await asyncio.wait_for(
                    call_agent_health(name, url),
                    timeout=gateway_settings.health_timeout_s,
                )
            except TimeoutError:
                return HealthStatus(status="error", agent=name, detail="timed out")

        results = await asyncio.gather(*[_check(name, url) for name, url in urls.items()])
        agents = {
            name: status for (name, _url), status in zip(urls.items(), results, strict=True)
        }
        overall: Literal["ok", "degraded"] = (
            "ok" if all(item.status == "ok" for item in agents.values()) else "degraded"
        )
        payload = AggregateHealth(status=overall, agents=agents)
        return JSONResponse(content=payload.model_dump(mode="json", exclude_none=True))

    @app.get("/api/graph/statistics")
    async def graph_statistics(request: Request) -> JSONResponse:
        gateway_settings: GatewaySettings = request.app.state.settings
        payload = await call_tool_json(
            gateway_settings.graph_query_url,
            "get_statistics",
            {},
            correlation_id=get_correlation_id(),
            timeout_s=gateway_settings.request_timeout_s,
        )
        return JSONResponse(content=payload)

    @app.get("/health")
    async def health_alias(request: Request) -> JSONResponse:
        return await agents_health(request)

    return app


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
            correlation_id=record.correlation_id,
        )
        record.status = "done"
        record.report = dict(report) if isinstance(report, dict) else {"result": report}
    except Exception as exc:
        record.status = "failed"
        record.error = str(exc)


app = create_app(settings)
