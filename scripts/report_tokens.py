"""Report token usage for the sample routing queries via `POST /api/chat`."""

from __future__ import annotations

import io
import json
import sys
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from core.gateway import GatewayDependencies
from core.llm.stub import StubProvider
from core.orchestration.models import QueryIntent
from core.settings import GatewaySettings, OrchestratorSettings
from gateway.app import create_app
from orchestrator.service import OrchestratorService

EVAL_PATH = Path(__file__).resolve().parents[1] / "evals" / "routing.jsonl"


class _MemoryClient:
    def __init__(self) -> None:
        self._cache: dict[str, object] = {}

    async def get_context(self, session_id: str, token_budget: int = 3000) -> object:
        _ = session_id, token_budget
        return SimpleNamespace(summary="", recent_turns=[])

    async def get_cached_response(self, cache_key: str) -> object | None:
        return self._cache.get(cache_key)

    async def cache_response(self, cache_key: str, response_json: object) -> None:
        self._cache[cache_key] = {"response_json": response_json}

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        _ = session_id, role, content


class _GraphQueryClient:
    async def get_statistics(self) -> object:
        return SimpleNamespace(index_version="idx-report")

    async def find_entity(self, name: str, entity_type: str | None = None) -> object:
        _ = entity_type
        return {"file_path": f"{name}.py", "line_start": 1, "line_end": 20}


class _CodeAnalystClient:
    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> object:
        _ = qualified_name
        return {
            "file_path": file_path or "",
            "line_start": line_start,
            "line_end": line_end,
            "text": f"snippet for {file_path}",
            "error": None,
        }


class _IndexerClient:
    async def index_repository(self, repo_url: str | None = None) -> object:
        return {"status": "ok", "repo_url": repo_url}


class _LocalOrchestratorClient:
    def __init__(self, query: str, expected_agents: list[str], expected_mode: str) -> None:
        responses: list[object] = []
        if expected_mode == "llm":
            responses.append(
                QueryIntent(
                    routing_mode="llm",
                    intent="explanation" if len(expected_agents) > 1 else "lookup",
                    entities=["FastAPI"],
                    target_agents=expected_agents,  # type: ignore[arg-type]
                    reasoning="report",
                ).model_dump()
            )
        responses.append(f"Answer for: {query}")
        self._service = OrchestratorService(
            StubProvider(responses),
            settings=OrchestratorSettings(routing_strategy="rules_first"),
        )
        self._clients = SimpleNamespace(
            memory=_MemoryClient(),
            graph_query=_GraphQueryClient(),
            code_analyst=_CodeAnalystClient(),
            indexer=_IndexerClient(),
        )

    async def handle_query(
        self,
        query: str,
        session_id: str,
        *,
        correlation_id: str,
    ) -> dict[str, object]:
        result = await self._service.handle_query(
            query,
            session_id,
            clients=self._clients,  # type: ignore[arg-type]
            correlation_id=correlation_id,
        )
        return {"answer": result.answer, "metadata": result.metadata}


def _cases() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in EVAL_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _make_client(query: str, expected_agents: list[str], expected_mode: str) -> TestClient:
    settings = GatewaySettings(host="127.0.0.1", port=8000)
    deps = GatewayDependencies(
        orchestrator=_LocalOrchestratorClient(query, expected_agents, expected_mode),  # type: ignore[arg-type]
        specialists=SimpleNamespace(indexer=object(), graph_query=object(), code_analyst=object()),
        gateway_settings=settings,
        orchestrator_settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    return TestClient(create_app(settings, deps=deps))


def main() -> None:
    print("| query | routing_mode | llm_calls | total_tokens | cached_total_tokens |")
    print("| --- | --- | ---: | ---: | ---: |")
    for case in _cases():
        query = str(case["query"])
        expected_agents = [str(agent) for agent in case["expected_agents"]]
        expected_mode = str(case["expected_mode"])
        client = _make_client(query, expected_agents, expected_mode)
        session_id = str(uuid.uuid4())
        with redirect_stdout(io.StringIO()):
            payload = {"message": query, "session_id": session_id}
            first = client.post("/api/chat", json=payload).json()
            second = client.post("/api/chat", json=payload).json()
        first_done = dict(first.get("done", {}))
        second_done = dict(second.get("done", {}))
        first_tokens = dict(first_done.get("tokens", {}))
        second_tokens = dict(second_done.get("tokens", {}))
        routing = dict(first.get("routing", {}))
        routing_mode = routing.get("routing_mode", first_done.get("routing_mode", "unknown"))
        escaped_query = query.replace("|", "\\|")
        print(
            f"| {escaped_query} | {routing_mode} | {first_tokens.get('llm_calls', 0)} "
            f"| {first_tokens.get('total', 0)} | {second_tokens.get('total', 0)} |"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"tokens report failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
