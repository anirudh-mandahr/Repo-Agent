"""Offline model bake-off harness: no network, no Neo4j."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from core.eval.model_bakeoff import (
    FrozenPayload,
    capture_synthesis_payloads,
    format_purpose_table,
    parse_models_arg,
    render_model_bakeoff_markdown,
    run_router_bakeoff,
    run_synthesis_bakeoff,
    write_frozen_payloads,
    write_model_bakeoff_section,
)
from core.eval.models import QaCase, TurnSpec
from core.exceptions import SchemaValidationError
from core.llm.provider import LLMPurpose, LLMResult, Message, TokenUsage, complete_with_schema_retry
from core.orchestration.models import QueryIntent


class _FixedRouter:
    def __init__(self, agents: list[str], *, fail_first: bool = False) -> None:
        self.agents = agents
        self.fail_first = fail_first
        self.calls = 0

    async def complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResult:
        _ = messages, max_tokens, temperature, purpose

        async def invoke(_attempt: list[Message]) -> tuple[Any, TokenUsage]:
            self.calls += 1
            usage = TokenUsage(
                prompt_tokens=20,
                completion_tokens=5,
                total_tokens=25,
                model="stub-router",
            )
            if self.fail_first and self.calls % 2 == 1:
                return "not-json", usage
            return (
                QueryIntent(
                    intent="lookup",
                    target_agents=self.agents,  # type: ignore[arg-type]
                    reasoning="test",
                ).model_dump(),
                usage,
            )

        return await complete_with_schema_retry(
            invoke, messages, response_model, purpose=purpose, agent=agent
        )


class _FixedSynthesizer:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    async def complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResult:
        _ = messages, response_model, purpose, agent, max_tokens, temperature
        return LLMResult(
            text=self.answer,
            usage=TokenUsage(
                prompt_tokens=50,
                completion_tokens=10,
                total_tokens=60,
                model="stub-synth",
            ),
        )


class _AlwaysHitGraph:
    def run_read(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        timeout_s: float = 10,
    ) -> list[dict[str, Any]]:
        _ = query, params, timeout_s
        return [{"count": 1}]


def _case(query: str, agents: list[str], *, tier: str = "simple") -> QaCase:
    return QaCase(
        id=f"{tier}-{query}",
        tier=tier,
        turns=(
            TurnSpec(
                query=query,
                expected_agents=agents,
                expected_mode="llm",
                min_agents=len(agents),
                out_of_scope=tier == "trap",
            ),
        ),
        out_of_scope=tier == "trap",
    )


def test_parse_models_arg_default_and_unique() -> None:
    models = parse_models_arg(None)
    assert "anthropic/claude-sonnet-4.5" in models
    assert parse_models_arg("a, b, a") == ("a", "b")


async def test_router_bakeoff_scores_exact_match_and_schema_retry() -> None:
    cases = [
        _case("What is FastAPI?", ["graph_query"]),
        _case("How does Django work?", ["graph_query", "code_analyst"], tier="trap"),
    ]
    report = await run_router_bakeoff(
        ["stub-a", "stub-b"],
        cases,
        repeats=3,
        provider_factory=lambda model: _FixedRouter(
            ["graph_query"] if model == "stub-a" else ["code_analyst"],
            fail_first=model == "stub-b",
        ),
    )
    assert report.purpose == "routing"
    assert report.repeats == 3
    assert len(report.rows) == 2
    by_model = {row.model: row for row in report.rows}
    assert by_model["stub-a"].quality.mean == 1.0
    assert by_model["stub-a"].quality.spread == 0.0
    assert by_model["stub-a"].schema_failure.mean == 0.0
    assert by_model["stub-b"].quality.mean == 0.0
    assert by_model["stub-b"].schema_failure.mean == 1.0
    markdown = format_purpose_table(report)
    assert "winner" not in markdown.lower()
    assert "stub-a" in markdown
    assert "as of" in markdown
    assert "mean ± half-range" in markdown or "half-range" in markdown


async def test_router_single_repeat_does_not_claim_a_winner() -> None:
    report = await run_router_bakeoff(
        ["only"],
        [_case("What is FastAPI?", ["graph_query"])],
        repeats=1,
        provider_factory=lambda _model: _FixedRouter(["graph_query"]),
    )
    text = format_purpose_table(report)
    assert "winner" not in text.lower()
    assert any("repeats=1" in note for note in report.notes)


async def test_synthesis_replay_uses_frozen_payloads() -> None:
    payloads = [
        FrozenPayload(
            id="s01",
            query="What is the FastAPI class?",
            tier="simple",
            out_of_scope=False,
            expected_agents=["graph_query"],
            expected_entities=["FastAPI"],
            agent_outputs={
                "graph_query": {
                    "agent": "graph_query",
                    "ok": True,
                    "output": {
                        "entities": [
                            {
                                "name": "FastAPI",
                                "qualified_name": "fastapi.applications.FastAPI",
                                "file_path": "fastapi/applications.py",
                                "line_start": 1,
                                "line_end": 40,
                            }
                        ]
                    },
                }
            },
            captured_at="2026-08-18",
        ),
        FrozenPayload(
            id="t01",
            query="How does Django's ORM lazy-load querysets?",
            tier="trap",
            out_of_scope=True,
            expected_agents=["graph_query", "code_analyst"],
            expected_entities=[],
            agent_outputs={},
            captured_at="2026-08-18",
        ),
    ]
    report = await run_synthesis_bakeoff(
        ["stub-synth"],
        payloads,
        repeats=3,
        provider_factory=lambda _model: _FixedSynthesizer(
            "FastAPI is defined in fastapi/applications.py:1. "
            "This topic is not in the indexed FastAPI codebase."
        ),
        graph_client=_AlwaysHitGraph(),
    )
    assert report.captured_at == "2026-08-18"
    row = report.rows[0]
    assert row.quality.mean == 1.0
    assert row.trap_pass.mean == 1.0
    assert row.schema_failure.mean == 0.0
    markdown = format_purpose_table(report)
    assert "captured 2026-08-18" in markdown
    assert "winner" not in markdown.lower()


async def test_capture_writes_fixtures(tmp_path: Path) -> None:
    path = tmp_path / "payloads.jsonl"
    cases = [_case("What is FastAPI?", ["graph_query"])]
    payloads = await capture_synthesis_payloads(cases, path=path)
    assert path.is_file()
    assert payloads[0].query == "What is FastAPI?"
    assert payloads[0].agent_outputs


def test_readme_patch_inserts_bakeoff_tables(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\n<!-- BEGIN_EVAL_REPORT -->\nold\n<!-- END_EVAL_REPORT -->\n")
    from core.eval.model_bakeoff import MeanSpread, ModelBakeoffRow, PurposeBakeoff
    from core.llm.pricing import ModelRates

    zero = MeanSpread(mean=1.0, spread=0.0, values=(1.0, 1.0, 1.0))
    report = PurposeBakeoff(
        purpose="routing",
        rows=[
            ModelBakeoffRow(
                model="anthropic/claude-sonnet-4.5",
                quality=zero,
                trap_pass=zero,
                schema_failure=MeanSpread(mean=0.0, spread=0.0, values=(0.0, 0.0, 0.0)),
                p50_latency_ms=zero,
                p95_latency_ms=zero,
                cost_per_query=zero,
                cost_per_1000=zero,
                rates=ModelRates(),
                price_label="sonnet price",
                repeats=3,
            )
        ],
        captured_at=None,
        price_as_of="2026-08-18",
        repeats=3,
        n_queries=2,
    )
    write_model_bakeoff_section([report], readme_path=readme)
    text = readme.read_text()
    assert "BEGIN_MODEL_BAKEOFF" in text
    assert "BEGIN_EVAL_REPORT" in text
    assert "winner" not in text.lower()
    assert render_model_bakeoff_markdown([report]).count("BEGIN_MODEL_BAKEOFF") == 1


def test_write_and_load_frozen_payloads(tmp_path: Path) -> None:
    path = tmp_path / "one.jsonl"
    payload = FrozenPayload(
        id="s01",
        query="q",
        tier="simple",
        out_of_scope=False,
        expected_agents=["graph_query"],
        expected_entities=["FastAPI"],
        agent_outputs={"graph_query": {"ok": True}},
        captured_at="2026-08-18",
    )
    write_frozen_payloads(path, [payload])
    from core.eval.model_bakeoff import load_frozen_payloads

    loaded = load_frozen_payloads(path)
    assert loaded[0].id == "s01"
    assert loaded[0].agent_outputs["graph_query"]["ok"] is True


def test_schema_validation_error_is_importable() -> None:
    assert SchemaValidationError is not None
