"""Request-level budget, derived synthesis timeout, and quality comparison."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.eval.budget_compare import (
    choose_truncation_order,
    quality_dropped,
    run_budget_comparison,
    score_truncation_orders,
)
from core.eval.harness import load_qa_cases
from core.eval.models import EvalScorecard, TierScorecard, TruncationOrderScore
from core.llm.provider import TokenUsage
from core.llm.stub import StubProvider
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger
from core.orchestration.budget import RequestBudget
from core.orchestration.models import QueryIntent
from core.orchestration.prompt_budget import DEFAULT_TRUNCATION_ORDER
from core.orchestration.service import OrchestratorService
from core.settings import OrchestratorSettings


class _Memory:
    def __init__(self) -> None:
        self.calls = 0

    async def get_context(self, session_id: str, token_budget: int = 3000) -> ConversationContext:
        _ = session_id, token_budget
        return ConversationContext()

    async def get_cached_response(self, cache_key: str) -> object | None:
        _ = cache_key
        return None

    async def cache_response(self, cache_key: str, response_json: object) -> None:
        _ = cache_key, response_json

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        _ = session_id, role, content
        self.calls += 1
        raise RuntimeError("memory write failed")


class _Graph:
    def __init__(self) -> None:
        self.find_calls = 0

    async def get_statistics(self) -> object:
        return SimpleNamespace(index_version="idx-1")

    async def find_entity(self, name: str, entity_type: str | None = None) -> object:
        _ = entity_type
        self.find_calls += 1
        return {
            "file_path": "fastapi/applications.py",
            "line_start": 10,
            "line_end": 40,
            "qualified_name": "fastapi.applications.FastAPI",
            "name": name,
        }

    async def get_dependencies(self, name: str) -> object:
        return {"name": name, "neighbors": []}

    async def get_dependents(self, name: str) -> object:
        return {"name": name, "neighbors": []}

    async def find_related(self, name: str, relationship_type: str) -> object:
        return {"name": name, "relationship_type": relationship_type, "neighbors": []}

    async def trace_imports(self, module: str, depth: int = 5) -> object:
        _ = depth
        return {"module": module, "paths": []}


class _Code:
    def __init__(self) -> None:
        self.calls = 0

    async def get_code_snippet(self, **kwargs: object) -> object:
        self.calls += 1
        return {
            "file_path": kwargs.get("file_path") or "fastapi/applications.py",
            "line_start": 10,
            "line_end": 12,
            "text": "class FastAPI:\n    pass\n",
            "error": None,
        }

    async def explain_implementation(self, qualified_name: str) -> object:
        self.calls += 1
        return {"qualified_name": qualified_name, "explanation": "explained", "error": None}

    async def analyze_function(self, qualified_name: str) -> object:
        self.calls += 1
        return {"qualified_name": qualified_name, "summary": "analyzed", "error": None}

    async def analyze_class(self, qualified_name: str) -> object:
        self.calls += 1
        return {"qualified_name": qualified_name, "summary": "analyzed class", "error": None}

    async def compare_implementations(self, name_a: str, name_b: str) -> object:
        self.calls += 1
        return {"name_a": name_a, "name_b": name_b, "summary": "compared", "error": None}

    async def find_patterns(self, pattern: str) -> object:
        self.calls += 1
        return {"pattern": pattern, "instances": [], "error": None}


class _Indexer:
    async def index_repository(self, repo_url: str | None = None) -> object:
        return {"status": "ok", "repo_url": repo_url}


def _clients(graph: _Graph | None = None, code: _Code | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        memory=_Memory(),
        graph_query=graph or _Graph(),
        code_analyst=code or _Code(),
        indexer=_Indexer(),
    )


def test_request_budget_reads_ledger_cost_without_recomputing() -> None:
    ledger = TokenLedger()
    ledger.open("corr")
    ledger.record(
        "corr",
        "routing",
        TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120, model="stub"),
    )
    snapshot = ledger.snapshot("corr")
    budget = RequestBudget(
        deadline_monotonic=time.monotonic() + 55,
        token_ceiling=10_000,
        cost_usd_max=0.0001,
        safety_margin_s=2.0,
        min_synthesis_timeout_s=0.0,
        max_synthesis_timeout_s=20.0,
    )
    assert snapshot["cost_usd"] == ledger.snapshot("corr")["cost_usd"]
    assert budget.check(ledger, "corr") == "cost"


def test_derived_synthesis_timeout_uses_remaining_minus_margin() -> None:
    now = time.monotonic()
    budget = RequestBudget(
        deadline_monotonic=now + 10,
        token_ceiling=None,
        cost_usd_max=None,
        safety_margin_s=2.0,
        min_synthesis_timeout_s=0.0,
        max_synthesis_timeout_s=20.0,
        _clock=lambda: now,
    )
    assert budget.synthesis_timeout_s(now) == 8.0
    exhausted = RequestBudget(
        deadline_monotonic=now,
        token_ceiling=None,
        cost_usd_max=None,
        safety_margin_s=2.0,
        min_synthesis_timeout_s=0.0,
        max_synthesis_timeout_s=20.0,
        _clock=lambda: now,
    )
    assert exhausted.synthesis_timeout_s(now) == 0.0


def test_plan_consuming_full_deadline_leaves_synthesis_time() -> None:
    class _Clock:
        def __init__(self, value: float) -> None:
            self.value = value

        def __call__(self) -> float:
            return self.value

    started = 1_000.0
    clock = _Clock(started)
    settings = OrchestratorSettings()
    budget = RequestBudget.from_settings(settings, now=started, clock=clock)
    clock.value = started + settings.plan_deadline_s
    assert budget.remaining_s() == pytest.approx(settings.synthesis_reserve_s)
    assert budget.allow_new_call(None, "corr") is False
    assert budget.synthesis_timeout_s() > 0
    derived = settings.synthesis_reserve_s - settings.synthesis_safety_margin_s
    assert budget.synthesis_timeout_s() == pytest.approx(derived)
    from core.llm.pricing import measured_synthesis_p95_s
    from core.settings import LLMSettings

    p95 = measured_synthesis_p95_s(LLMSettings().resolve_model("synthesis"))
    assert p95 is not None
    assert budget.synthesis_timeout_s() >= p95


@pytest.mark.asyncio
async def test_token_ceiling_stops_new_specialist_calls() -> None:
    graph = _Graph()
    code = _Code()
    ledger = TokenLedger()
    intent = QueryIntent(
        intent="lookup",
        entities=["FastAPI"],
        target_agents=["graph_query"],
        reasoning="lookup",
    )
    service = OrchestratorService(
        StubProvider([intent.model_dump(), "FINAL ANSWER"]),
        settings=OrchestratorSettings(
            routing_strategy="llm_first",
            request_budgets_enabled=True,
            request_token_budget=50,
            request_cost_usd_max=0.0,
            request_deadline_s=55,
        ),
        token_ledger=ledger,
    )
    result = await service.handle_query(
        "What is the FastAPI class?",
        "s1",
        clients=_clients(graph, code),  # type: ignore[arg-type]
        correlation_id="corr-tokens",
    )
    assert result.metadata.get("budget_exhausted") == "tokens"
    assert graph.find_calls == 0
    assert code.calls == 0
    assert result.answer


@pytest.mark.asyncio
async def test_deadline_exhaustion_skips_follow_up_and_synthesizes() -> None:
    graph = _Graph()
    service = OrchestratorService(
        StubProvider(["FINAL ANSWER"]),
        settings=OrchestratorSettings(
            routing_strategy="rules_first",
            request_budgets_enabled=True,
            request_deadline_s=0.0,
            plan_deadline_s=0.0,
            synthesis_reserve_s=0.0,
            request_token_budget=0,
            request_cost_usd_max=0.0,
            synthesis_safety_margin_s=0.0,
        ),
    )
    result = await service.handle_query(
        "What is the FastAPI class?",
        "s1",
        clients=_clients(graph),  # type: ignore[arg-type]
        correlation_id="corr-deadline",
    )
    assert result.metadata.get("budget_exhausted") == "deadline"
    assert result.answer


@pytest.mark.asyncio
async def test_append_turn_failure_is_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class _Log:
        def info(self, event: str, **kwargs: Any) -> None:
            _ = event, kwargs

        def warning(self, event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

    monkeypatch.setattr("core.orchestration.service.log", _Log())
    service = OrchestratorService(
        StubProvider(["FINAL ANSWER"]),
        settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    await service.handle_query(
        "What is the FastAPI class?",
        "s1",
        clients=_clients(),  # type: ignore[arg-type]
        correlation_id="corr-log",
    )
    names = [name for name, _fields in events]
    assert "orchestrator.append_turn_failed" in names
    assert any(fields.get("correlation_id") == "corr-log" for _n, fields in events)


class _FailingMemory(_Memory):
    async def get_context(self, session_id: str, token_budget: int = 3000) -> ConversationContext:
        _ = session_id, token_budget
        raise RuntimeError("memory agent down")


class _FailingGraph(_Graph):
    async def get_statistics(self) -> object:
        raise RuntimeError("graph statistics down")


@pytest.mark.asyncio
async def test_memory_context_failure_is_logged_and_surfaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class _Log:
        def info(self, event: str, **kwargs: Any) -> None:
            _ = event, kwargs

        def warning(self, event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

    monkeypatch.setattr("core.orchestration.service.log", _Log())
    service = OrchestratorService(
        StubProvider(["FINAL ANSWER"]),
        settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    clients = SimpleNamespace(
        memory=_FailingMemory(),
        graph_query=_Graph(),
        code_analyst=_Code(),
        indexer=_Indexer(),
    )
    result = await service.handle_query(
        "What is the FastAPI class?",
        "s1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-mem-fail",
    )
    assert result.metadata["memory_available"] is False
    assert result.metadata["graph_statistics_available"] is True
    assert result.metadata["note"] == "memory unavailable; proceeding statelessly"
    names = [name for name, _fields in events]
    assert "orchestrator.memory_context_failed" in names
    assert any(
        name == "orchestrator.memory_context_failed"
        and fields.get("correlation_id") == "corr-mem-fail"
        for name, fields in events
    )


@pytest.mark.asyncio
async def test_graph_statistics_failure_is_logged_and_surfaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, dict[str, Any]]] = []

    class _Log:
        def info(self, event: str, **kwargs: Any) -> None:
            _ = event, kwargs

        def warning(self, event: str, **kwargs: Any) -> None:
            events.append((event, kwargs))

    monkeypatch.setattr("core.orchestration.service.log", _Log())
    service = OrchestratorService(
        StubProvider(["FINAL ANSWER"]),
        settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    clients = SimpleNamespace(
        memory=_Memory(),
        graph_query=_FailingGraph(),
        code_analyst=_Code(),
        indexer=_Indexer(),
    )
    result = await service.handle_query(
        "What is the FastAPI class?",
        "s1",
        clients=clients,  # type: ignore[arg-type]
        correlation_id="corr-stats-fail",
    )
    assert result.metadata["memory_available"] is True
    assert result.metadata["graph_statistics_available"] is False
    assert result.metadata["index_version"] is None
    names = [name for name, _fields in events]
    assert "orchestrator.graph_statistics_failed" in names
    assert any(
        name == "orchestrator.graph_statistics_failed"
        and fields.get("correlation_id") == "corr-stats-fail"
        for name, fields in events
    )


@pytest.mark.asyncio
async def test_streamed_synthesis_records_ttft_under_two_seconds() -> None:
    from core.observability.metrics import render_metrics
    from core.orchestration.synthesis import synthesize_response

    chunks: list[str] = []

    async def on_token(chunk: str) -> None:
        chunks.append(chunk)

    llm = StubProvider(["hello from streamed synthesis"], stream_chunk_delay_s=0.01)
    started = time.perf_counter()
    first: float | None = None

    async def _mark(chunk: str) -> None:
        nonlocal first
        if first is None:
            first = time.perf_counter()
        await on_token(chunk)

    result = await synthesize_response(
        "What is the FastAPI class?",
        {
            "graph_query": {
                "ok": True,
                "output": {
                    "queried_entities": ["FastAPI"],
                    "candidates": [
                        {
                            "qualified_name": "fastapi.applications.FastAPI",
                            "file_path": "fastapi/applications.py",
                            "line_start": 10,
                            "line_end": 12,
                        }
                    ],
                },
            }
        },
        ConversationContext(),
        llm_provider=llm,
        settings=OrchestratorSettings(),
        on_token=_mark,
    )
    elapsed = time.perf_counter() - started
    assert result.evidence_only is False
    assert result.partial is False
    assert chunks
    assert first is not None
    assert (first - started) < 2.0
    assert elapsed >= (first - started)
    text = render_metrics().decode("utf-8")
    assert "repochat_time_to_first_token_seconds" in text
    assert "repochat_synthesis_duration_seconds" in text


@pytest.mark.asyncio
async def test_stream_error_mid_answer_keeps_partial_tokens() -> None:
    from core.orchestration.fallback import EVIDENCE_ONLY_HEADER
    from core.orchestration.synthesis import synthesize_response

    class _BoomStream:
        async def stream(self, *args: object, **kwargs: object) -> Any:
            _ = args, kwargs
            yield "FastAPI subclasses Starlette. ", None
            raise RuntimeError("stream died")

    result = await synthesize_response(
        "What is the FastAPI class?",
        {
            "graph_query": {
                "ok": True,
                "output": {
                    "queried_entities": ["FastAPI"],
                    "candidates": [
                        {
                            "qualified_name": "fastapi.applications.FastAPI",
                            "file_path": "fastapi/applications.py",
                            "line_start": 10,
                            "line_end": 12,
                        }
                    ],
                },
            }
        },
        ConversationContext(),
        llm_provider=_BoomStream(),  # type: ignore[arg-type]
        settings=OrchestratorSettings(),
    )
    assert result.evidence_only is False
    assert result.partial is True
    assert result.degraded_reason == "RuntimeError"
    assert "FastAPI subclasses Starlette" in result.answer
    assert EVIDENCE_ONLY_HEADER not in result.answer


@pytest.mark.asyncio
async def test_budget_quality_does_not_drop_per_tier(tmp_path: Path) -> None:
    cases = load_qa_cases()[:8]
    comparison = await run_budget_comparison(cases)
    assert comparison.no_quality_drop
    assert not quality_dropped(comparison.enabled, comparison.disabled)
    assert comparison.chosen_truncation_order == DEFAULT_TRUNCATION_ORDER
    source = tmp_path / "fastapi"
    source.mkdir()
    (source / "applications.py").write_text(
        "\n".join(f"line {i}" for i in range(20)), encoding="utf-8"
    )
    orders = await score_truncation_orders(
        {
            "graph_query": {
                "ok": True,
                "output": {
                    "candidates": [
                        {
                            "qualified_name": "fastapi.applications.FastAPI",
                            "file_path": "fastapi/applications.py",
                            "line_start": 10,
                            "line_end": 12,
                        }
                    ]
                    * 20
                },
            },
            "code_analyst": {
                "ok": True,
                "output": {
                    "snippets": [
                        {
                            "file_path": "fastapi/applications.py",
                            "line_start": 10,
                            "line_end": 12,
                            "text": "class FastAPI:\n" + ("x" * 20_000),
                            "error": None,
                        }
                    ]
                },
            },
        },
        query="Compare FastAPI and APIRouter implementations in the codebase",
        repo_root=tmp_path,
    )
    assert all(item.citation_precision == 1.0 for item in orders)
    assert choose_truncation_order(orders) == DEFAULT_TRUNCATION_ORDER


def test_quality_dropped_detects_regression() -> None:
    def _card(cite: float, ground: float) -> EvalScorecard:
        return EvalScorecard(
            tiers=[
                TierScorecard(
                    tier="simple",
                    n=1,
                    passed=1,
                    executed_agents=1.0,
                    citation_precision=cite,
                    groundedness=ground,
                    entity_recall=1.0,
                    retrieval_correctness=1.0,
                    refusal=None,
                    degraded_fails=0,
                )
            ]
        )

    assert not quality_dropped(_card(1.0, 1.0), _card(1.0, 1.0))
    assert quality_dropped(_card(0.5, 1.0), _card(1.0, 1.0))
    assert quality_dropped(_card(1.0, 0.5), _card(1.0, 1.0))


def test_choose_truncation_order_prefers_measured_winner() -> None:
    lists_win = [
        TruncationOrderScore("snippets_then_lists", 0.5, 0.5),
        TruncationOrderScore("lists_then_snippets", 1.0, 1.0),
    ]
    assert choose_truncation_order(lists_win) == "lists_then_snippets"


@pytest.mark.asyncio
async def test_chat_gateway_streams_answer_events_live() -> None:
    from core.gateway import ChatGatewayService, GatewayDependencies
    from core.settings import GatewaySettings

    class _Orch:
        async def handle_query(
            self,
            query: str,
            session_id: str,
            *,
            correlation_id: str,
            on_token: Any | None = None,
            on_event: Any | None = None,
        ) -> dict[str, Any]:
            _ = query, session_id
            if on_event is not None:
                await on_event(
                    "routing",
                    {"iteration": 1, "agents": ["graph_query"], "sufficient": True},
                )
                await on_event("agent_result", {"agent": "orchestrator", "ok": True})
            if on_token is not None:
                await on_token("Hel")
                await on_token("lo")
            return {
                "answer": "Hello",
                "metadata": {
                    "routing_mode": "rules",
                    "cached": False,
                    "degraded": False,
                    "tokens": {"total": 1, "prompt": 1, "completion": 0, "llm_calls": 1},
                    "tools_invoked": ["graph_query.find_entity"],
                },
            }

    deps = GatewayDependencies(
        orchestrator=_Orch(),  # type: ignore[arg-type]
        specialists=SimpleNamespace(indexer=object(), graph_query=object(), code_analyst=object()),
        gateway_settings=GatewaySettings(),
        orchestrator_settings=OrchestratorSettings(),
    )
    events = [
        event async for event in ChatGatewayService(deps).stream("What is FastAPI?", "s", "c")
    ]
    types = [event.type for event in events]
    assert types[0] == "routing"
    assert "answer" in types
    assert types[-1] == "done"
    chunks = [event.data.get("chunk") for event in events if event.type == "answer"]
    assert chunks == ["Hel", "lo"]


@pytest.mark.asyncio
async def test_handle_query_streams_tokens_to_callback() -> None:
    tokens: list[str] = []

    async def on_token(chunk: str) -> None:
        tokens.append(chunk)

    service = OrchestratorService(
        StubProvider(["streamed final answer"]),
        settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    result = await service.handle_query(
        "What is the FastAPI class?",
        "s1",
        clients=_clients(),  # type: ignore[arg-type]
        correlation_id="corr-stream",
        on_token=on_token,
    )
    assert tokens
    assert "streamed final answer" in "".join(tokens)
    assert "streamed final answer" in result.answer


def test_scaled_synthesis_reserve_grows_with_prompt_tokens() -> None:
    from core.orchestration.prompt_budget import (
        REFERENCE_SYNTHESIS_PROMPT_TOKENS,
        SYNTHESIS_EXTRA_SECONDS_PER_TOKEN,
        scaled_synthesis_reserve_s,
    )

    base = 20.0
    assert scaled_synthesis_reserve_s(0, base_reserve_s=base) == base
    assert scaled_synthesis_reserve_s(
        REFERENCE_SYNTHESIS_PROMPT_TOKENS, base_reserve_s=base
    ) == base
    extra = 4_000
    assert scaled_synthesis_reserve_s(
        REFERENCE_SYNTHESIS_PROMPT_TOKENS + extra, base_reserve_s=base
    ) == pytest.approx(base + extra * SYNTHESIS_EXTRA_SECONDS_PER_TOKEN)


def test_large_prompt_uncaps_typical_synthesis_timeout() -> None:
    from core.orchestration.prompt_budget import REFERENCE_SYNTHESIS_PROMPT_TOKENS

    now = 1_000.0
    settings = OrchestratorSettings()
    budget = RequestBudget.from_settings(settings, now=now, clock=lambda: now)
    typical = budget.synthesis_timeout_s(now, prompt_tokens=1_000)
    assert typical == pytest.approx(settings.synthesis_timeout_s)
    large_tokens = REFERENCE_SYNTHESIS_PROMPT_TOKENS + 4_000
    scaled = budget.synthesis_timeout_s(now, prompt_tokens=large_tokens)
    assert scaled > settings.synthesis_timeout_s
    budget.apply_prompt_reserve(large_tokens)
    assert budget.synthesis_reserve_s > settings.synthesis_reserve_s
    assert budget.synthesis_timeout_s(now) == pytest.approx(scaled)


def test_apply_prompt_reserve_reduces_specialist_window() -> None:
    from core.orchestration.prompt_budget import REFERENCE_SYNTHESIS_PROMPT_TOKENS

    class _Clock:
        def __init__(self, value: float) -> None:
            self.value = value

        def __call__(self) -> float:
            return self.value

    started = 1_000.0
    clock = _Clock(started)
    settings = OrchestratorSettings()
    budget = RequestBudget.from_settings(settings, now=started, clock=clock)
    clock.value = started + 10.0
    before = budget.specialist_remaining_s()
    budget.apply_prompt_reserve(REFERENCE_SYNTHESIS_PROMPT_TOKENS + 5_000)
    after = budget.specialist_remaining_s()
    assert budget.synthesis_reserve_s > settings.synthesis_reserve_s
    assert after < before
    assert after == pytest.approx(budget.remaining_s() - budget.synthesis_reserve_s)


@pytest.mark.asyncio
async def test_stream_timeout_keeps_emitted_tokens() -> None:
    import asyncio

    from core.orchestration.fallback import EVIDENCE_ONLY_HEADER
    from core.orchestration.synthesis import synthesize_response

    class _PartialThenHang:
        async def stream(self, *args: object, **kwargs: object) -> Any:
            _ = args, kwargs
            yield "FastAPI subclasses Starlette and ", None
            yield "APIRouter groups path operations.", None
            await asyncio.sleep(5)

    result = await synthesize_response(
        "What is the FastAPI class?",
        {
            "graph_query": {
                "ok": True,
                "output": {
                    "queried_entities": ["FastAPI"],
                    "candidates": [
                        {
                            "qualified_name": "fastapi.applications.FastAPI",
                            "file_path": "fastapi/applications.py",
                            "line_start": 10,
                            "line_end": 12,
                        }
                    ],
                },
            }
        },
        ConversationContext(),
        llm_provider=_PartialThenHang(),  # type: ignore[arg-type]
        settings=OrchestratorSettings(synthesis_timeout_s=0.05),
    )
    assert result.evidence_only is False
    assert result.partial is True
    assert result.degraded_reason == "TimeoutError"
    assert "FastAPI subclasses Starlette" in result.answer
    assert "APIRouter groups path operations" in result.answer
    assert EVIDENCE_ONLY_HEADER not in result.answer


@pytest.mark.asyncio
async def test_c05_fanout_does_not_starve_synthesis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison-shaped mixed plan; plan spends its deadline, latency scales with prompt size."""
    import asyncio

    from core.orchestration.fallback import EVIDENCE_ONLY_HEADER
    from core.orchestration.prompt_budget import REFERENCE_SYNTHESIS_PROMPT_TOKENS

    monkeypatch.setattr("core.llm.pricing.measured_synthesis_p95_s", lambda _model: 0.05)

    plan_deadline_s = 0.25
    starving_cap_s = 0.08
    settings = OrchestratorSettings(
        routing_strategy="llm_first",
        plan_deadline_s=plan_deadline_s,
        request_deadline_s=4.0,
        synthesis_reserve_s=0.8,
        synthesis_timeout_s=starving_cap_s,
        synthesis_safety_margin_s=0.05,
        request_budgets_enabled=True,
        request_token_budget=50_000,
        request_cost_usd_max=1.0,
    )

    bulky = "class FastAPI:\n" + ("x" * 24_000)

    class _FanoutGraph(_Graph):
        async def find_entity(self, name: str, entity_type: str | None = None) -> object:
            _ = entity_type
            await asyncio.sleep(plan_deadline_s)
            if name == "APIRouter":
                return {
                    "file_path": "fastapi/routing.py",
                    "line_start": 10,
                    "line_end": 40,
                    "qualified_name": "fastapi.routing.APIRouter",
                    "name": name,
                    "entity_type": "Class",
                    "type": "Class",
                }
            if name == "FastAPI":
                return {
                    "file_path": "fastapi/applications.py",
                    "line_start": 10,
                    "line_end": 40,
                    "qualified_name": "fastapi.applications.FastAPI",
                    "name": name,
                    "entity_type": "Class",
                    "type": "Class",
                }
            return {
                "file_path": "fastapi/applications.py",
                "line_start": 42,
                "line_end": 80,
                "qualified_name": "fastapi.applications.FastAPI.__init__",
                "name": name,
                "entity_type": "Function",
                "type": "Function",
            }

        async def get_dependents(self, name: str) -> object:
            return {
                "name": name,
                "neighbors": [
                    {
                        "name": f"dep_{index}",
                        "qualified_name": f"pkg.mod.dep_{index}",
                        "file_path": f"pkg/mod_{index}.py",
                        "relationship_type": "DEPENDS_ON",
                    }
                    for index in range(40)
                ],
            }

    class _FanoutCode(_Code):
        async def get_code_snippet(self, **kwargs: object) -> object:
            self.calls += 1
            return {
                "file_path": kwargs.get("file_path") or "fastapi/applications.py",
                "line_start": 10,
                "line_end": 12,
                "text": bulky,
                "error": None,
            }

        async def explain_implementation(self, qualified_name: str) -> object:
            self.calls += 1
            return {
                "qualified_name": qualified_name,
                "explanation": bulky,
                "error": None,
            }

        async def compare_implementations(self, name_a: str, name_b: str) -> object:
            self.calls += 1
            return {
                "name_a": name_a,
                "name_b": name_b,
                "summary": bulky,
                "similarities": [bulky],
                "differences": [bulky],
                "error": None,
            }

    intent = QueryIntent(
        intent="mixed",
        entities=["FastAPI", "APIRouter"],
        target_agents=["graph_query", "code_analyst"],
        reasoning="c05 comparison-shaped mixed plan",
    )
    answer = "FastAPI subclasses Starlette; APIRouter owns the route table."
    llm = StubProvider(
        [intent.model_dump(), answer],
        delay_per_prompt_token_s=0.00004,
    )
    service = OrchestratorService(llm, settings=settings)
    result = await service.handle_query(
        "Compare FastAPI and APIRouter implementations, then show who depends on them",
        "s-c05",
        clients=_clients(_FanoutGraph(), _FanoutCode()),  # type: ignore[arg-type]
        correlation_id="corr-c05-budget",
    )
    tools = set(result.metadata.get("tools_invoked") or [])
    expected_tools = {
        "graph_query.find_entity",
        "graph_query.get_dependents",
        "code_analyst.compare_implementations",
    }
    unused_tools = {
        "graph_query.get_dependencies",
        "graph_query.trace_imports",
        "graph_query.find_related",
        "code_analyst.explain_implementation",
        "code_analyst.analyze_class",
        "code_analyst.analyze_function",
        "code_analyst.get_code_snippet",
    }
    assert expected_tools <= tools
    assert not tools.intersection(unused_tools)
    synthesis_calls = [call for call in llm.calls if call.purpose == "synthesis"]
    assert synthesis_calls
    prompt = "\n".join(message.content for message in synthesis_calls[0].messages)
    prompt_tokens = max(1, (len(prompt) + 3) // 4)
    assert prompt_tokens > REFERENCE_SYNTHESIS_PROMPT_TOKENS
    assert prompt_tokens * 0.00004 > starving_cap_s
    assert result.metadata.get("evidence_only") is not True
    assert result.metadata.get("degraded") is not True
    assert EVIDENCE_ONLY_HEADER not in result.answer
    assert "APIRouter owns the route table" in result.answer

