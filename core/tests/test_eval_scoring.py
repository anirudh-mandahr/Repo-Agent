"""Executed-plan scoring, answer quality, and per-tier scorecards."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.eval.harness import build_tier_scorecards, stub_clients
from core.eval.models import THRESHOLD_GROUNDEDNESS, EvalScorecard, MetricScore, TurnScore, TurnSpec
from core.eval.report import format_scorecard_text, render_readme_section, write_readme_section
from core.eval.scoring import (
    agents_from_tools_invoked,
    citation_exists_on_disk,
    claim_supported,
    executed_agents_ok,
    extract_citations,
    is_checkable_claim,
    router_only_agents,
    score_citation_precision,
    score_entity_recall,
    score_groundedness,
    score_retrieval_correctness,
    score_turn,
)
from core.orchestration.fallback import EVIDENCE_ONLY_HEADER

QUERY_7 = "Reindex the repository and trace how APIRouter imports connect to FastAPI"
QUERY_9 = "Explain how dependency injection, APIRouter, and get_openapi connect across the codebase"


def test_router_only_check_passes_when_executed_plan_diverges_query_7() -> None:
    turn = TurnSpec(
        query=QUERY_7,
        expected_agents=["indexer", "graph_query", "code_analyst"],
        expected_mode="llm",
        min_agents=3,
    )
    old_agents, old_mode = router_only_agents(turn)
    assert old_mode == "llm"
    assert old_agents == turn.expected_agents
    tools = ["indexer.index_repository", "graph_query.find_entity"]
    assert agents_from_tools_invoked(tools) == ["indexer", "graph_query"]
    assert not executed_agents_ok(turn, tools, routing_mode="llm")


def test_router_only_check_passes_when_executed_plan_diverges_query_9() -> None:
    turn = TurnSpec(
        query=QUERY_9,
        expected_agents=["graph_query", "code_analyst"],
        expected_mode="llm",
        min_agents=2,
    )
    old_agents, old_mode = router_only_agents(turn)
    assert old_mode == "llm"
    assert old_agents == turn.expected_agents
    tools = ["graph_query.find_entity"]
    assert not executed_agents_ok(turn, tools, routing_mode="llm")


def test_executed_agents_ok_matches_tools_invoked() -> None:
    turn = TurnSpec(
        query="What is the FastAPI class?",
        expected_agents=["graph_query"],
        expected_mode="rules",
        min_agents=1,
    )
    assert executed_agents_ok(
        turn,
        ["graph_query.find_entity"],
        routing_mode="rules",
    )


def test_citation_precision_accepts_disk_lines(tmp_path: Path) -> None:
    source = tmp_path / "fastapi"
    source.mkdir()
    file_path = source / "applications.py"
    file_path.write_text("class FastAPI:\n    pass\n", encoding="utf-8")
    answer = "The FastAPI class lives in fastapi/applications.py:1-2."
    assert extract_citations(answer) == [("fastapi/applications.py", 1)]
    assert citation_exists_on_disk("fastapi/applications.py", 1, tmp_path)
    precision, invalid = score_citation_precision(answer, repo_root=tmp_path)
    assert precision == 1.0
    assert invalid == []


def test_extract_citations_accepts_prose_line_ranges() -> None:
    answer = (
        "get_openapi is defined in `fastapi/openapi/utils.py` (lines 585–679) "
        "and FastAPI lives in fastapi/applications.py:42-4774."
    )
    cites = extract_citations(answer)
    assert ("fastapi/openapi/utils.py", 585) in cites
    assert ("fastapi/applications.py", 42) in cites


def test_citation_precision_rejects_missing_file(tmp_path: Path) -> None:
    precision, invalid = score_citation_precision(
        "See missing/nope.py:99.",
        repo_root=tmp_path,
    )
    assert precision == 0.0
    assert invalid == ["missing/nope.py:99"]


def test_citation_precision_none_when_no_citations() -> None:
    precision, invalid = score_citation_precision("No file coordinates in this stub.")
    assert precision is None
    assert invalid == []


def test_groundedness_none_when_no_checkable_claims() -> None:
    score, ungrounded = score_groundedness(
        "Offline analysis stub (set OPENROUTER_API_KEY for full LLM-backed output).",
        {"graph_query": {"output": {"candidates": [{"name": "FastAPI"}]}}},
    )
    assert score is None
    assert ungrounded == []


def test_snake_case_and_qualified_names_are_checkable() -> None:
    claim = "get_openapi lives in fastapi.openapi.utils and is called by FastAPI."
    assert is_checkable_claim(claim)
    evidence = '{"qualified_name": "fastapi.openapi.utils.get_openapi", "name": "FastAPI"}'.lower()
    assert claim_supported(claim, evidence)
    assert not claim_supported(claim, '{"name": "FastAPI"}'.lower())


def test_tier_scorecard_excludes_unscored_quality_metrics() -> None:
    cited = TurnScore(
        case_id="s01",
        tier="simple",
        query="q",
        expected_agents=["graph_query"],
        executed_agents=["graph_query"],
        routing_mode="rules",
        agents_passed=True,
        citation_precision=1.0,
        citation_passed=True,
        groundedness=0.5,
        grounded_passed=False,
        entity_recall=1.0,
        entity_passed=True,
        degraded=False,
        evidence_only=False,
        refusal_ok=None,
    )
    empty = TurnScore(
        case_id="s02",
        tier="simple",
        query="q2",
        expected_agents=["graph_query"],
        executed_agents=["graph_query"],
        routing_mode="rules",
        agents_passed=True,
        citation_precision=None,
        citation_passed=True,
        groundedness=None,
        grounded_passed=True,
        entity_recall=1.0,
        entity_passed=True,
        degraded=False,
        evidence_only=False,
        refusal_ok=None,
    )
    cards = build_tier_scorecards([cited, empty])
    simple = {item.tier: item for item in cards}["simple"]
    assert simple.citation_precision == 1.0
    assert simple.citation_n == 1
    assert simple.citation_excluded == 1
    assert simple.groundedness == 0.5
    assert simple.groundedness_n == 1
    assert simple.groundedness_excluded == 1
    from core.eval.harness import _quality_metrics

    metrics = {item.name: item for item in _quality_metrics([cited, empty])}
    assert metrics["citation_precision"].n == 1
    assert metrics["citation_precision"].value == 1.0
    assert "excluded" in metrics["citation_precision"].detail
    assert metrics["groundedness"].n == 1
    assert metrics["groundedness"].value == 0.5


def test_groundedness_flags_claims_without_agent_evidence() -> None:
    answer = "FastAPI is defined in fastapi/applications.py:1 and also uses Django ORM internals."
    outputs = {
        "graph_query": {
            "output": {
                "candidates": [
                    {
                        "name": "FastAPI",
                        "file_path": "fastapi/applications.py",
                        "line_start": 1,
                    }
                ]
            }
        }
    }
    score, ungrounded = score_groundedness(answer, outputs)
    assert score < 1.0
    assert ungrounded


def test_groundedness_accepts_supported_claims() -> None:
    answer = "FastAPI is defined in fastapi/applications.py:1."
    outputs = {
        "graph_query": {
            "output": {
                "candidates": [
                    {
                        "name": "FastAPI",
                        "file_path": "fastapi/applications.py",
                        "line_start": 1,
                    }
                ]
            }
        }
    }
    score, ungrounded = score_groundedness(answer, outputs)
    assert score == 1.0
    assert ungrounded == []


def test_expected_entity_recall() -> None:
    recall, missing = score_entity_recall(
        "Sources:\n- fastapi/applications.py:1-40 (fastapi.applications.FastAPI)",
        ["FastAPI"],
    )
    assert recall == 1.0
    assert missing == []
    miss_recall, miss = score_entity_recall("no symbols here", ["APIRouter"])
    assert miss_recall == 0.0
    assert miss == ["APIRouter"]
    from_evidence, _missing = score_entity_recall(
        "Offline analysis stub.",
        ["FastAPI"],
        {"graph_query": {"output": {"candidates": [{"name": "FastAPI"}]}}},
    )
    assert from_evidence == 1.0


def test_non_trap_degraded_answer_fails() -> None:
    turn = TurnSpec(
        query="What is the FastAPI class?",
        expected_agents=["graph_query"],
        expected_mode="rules",
        min_agents=1,
    )
    score = score_turn(
        "s01",
        "simple",
        turn,
        answer=f"{EVIDENCE_ONLY_HEADER}\nretrieved hits",
        metadata={
            "tools_invoked": ["graph_query.find_entity"],
            "routing_mode": "rules",
            "degraded": True,
            "evidence_only": True,
            "tokens": {"total": 10, "prompt": 8, "completion": 2, "llm_calls": 1},
        },
        quality=False,
    )
    assert score.hard_fail
    assert not score.passed


def test_trap_degraded_answer_does_not_hard_fail() -> None:
    turn = TurnSpec(
        query="How does Django's ORM lazy-load querysets?",
        expected_agents=["graph_query", "code_analyst"],
        expected_mode="rules",
        min_agents=2,
        out_of_scope=True,
    )
    score = score_turn(
        "t01",
        "trap",
        turn,
        answer="This topic is not in the indexed FastAPI codebase.",
        metadata={
            "tools_invoked": ["graph_query.find_entity", "code_analyst.get_code_snippet"],
            "routing_mode": "rules",
            "degraded": True,
            "tokens": {"total": 0, "prompt": 0, "completion": 0, "llm_calls": 0},
        },
        quality=False,
    )
    assert not score.hard_fail
    assert score.passed


def test_tier_scorecard_reports_pass_rates_not_single_verdict() -> None:
    simple = TurnScore(
        case_id="s01",
        tier="simple",
        query="q",
        expected_agents=["graph_query"],
        executed_agents=["graph_query"],
        routing_mode="rules",
        agents_passed=True,
        citation_precision=1.0,
        citation_passed=True,
        groundedness=1.0,
        grounded_passed=True,
        entity_recall=1.0,
        entity_passed=True,
        degraded=False,
        evidence_only=False,
        refusal_ok=None,
    )
    medium_fail = TurnScore(
        case_id="m01",
        tier="medium",
        query="q2",
        expected_agents=["graph_query", "code_analyst"],
        executed_agents=["graph_query"],
        routing_mode="llm",
        agents_passed=False,
        citation_precision=1.0,
        citation_passed=True,
        groundedness=1.0,
        grounded_passed=True,
        entity_recall=1.0,
        entity_passed=True,
        degraded=False,
        evidence_only=False,
        refusal_ok=None,
    )
    cards = build_tier_scorecards([simple, medium_fail])
    by_tier = {item.tier: item for item in cards}
    assert by_tier["simple"].pass_rate == 1.0
    assert by_tier["medium"].pass_rate == 0.0
    assert by_tier["complex"].n == 0
    text = format_scorecard_text(EvalScorecard(turns=[simple, medium_fail], tiers=cards))
    assert "simple" in text
    assert "medium" in text
    assert "per-tier" in text.lower()
    recall_only = EvalScorecard(
        turns=[simple],
        tiers=cards,
        metrics=[
            MetricScore("hard_executed_agents", 1.0, True, "1/1", n=1, hits=1),
            MetricScore("hard_non_trap_not_degraded", 1.0, True, "ok", n=1, hits=1),
            MetricScore("citation_precision", 1.0, True, "1/1", n=1, hits=1),
            MetricScore("entity_recall", 0.5, False, "1/2", n=2, hits=1),
            MetricScore("refusal_accuracy", 0.83, False, "5/6", n=6, hits=5),
        ],
    )
    assert recall_only.hard_assertions_passed
    assert recall_only.ok


@pytest.mark.asyncio
async def test_handle_query_records_executed_agents_in_tools_invoked() -> None:
    from core.llm.offline_provider import OfflineProvider
    from core.orchestration.service import OrchestratorService
    from core.settings import OrchestratorSettings

    service = OrchestratorService(
        OfflineProvider(),
        settings=OrchestratorSettings(routing_strategy="rules_first"),
    )
    result = await service.handle_query(
        QUERY_7,
        "eval-q7",
        clients=stub_clients(),
        correlation_id="eval-q7",
    )
    tools = list(result.metadata["tools_invoked"])
    turn = TurnSpec(
        query=QUERY_7,
        expected_agents=["indexer", "graph_query", "code_analyst"],
        expected_mode="llm",
        min_agents=3,
    )
    assert executed_agents_ok(turn, tools, routing_mode=str(result.metadata["routing_mode"]))
    assert result.agent_outputs


def test_readme_section_includes_latency_tokens_cost_and_agents() -> None:
    turn = TurnScore(
        case_id="r1",
        tier="simple",
        query="What is the FastAPI class?",
        expected_agents=["graph_query"],
        executed_agents=["graph_query"],
        routing_mode="rules",
        agents_passed=True,
        citation_precision=1.0,
        citation_passed=True,
        groundedness=1.0,
        grounded_passed=True,
        entity_recall=1.0,
        entity_passed=True,
        degraded=False,
        evidence_only=False,
        refusal_ok=None,
        answer="FastAPI is the application class.",
        latency_ms=12,
        tokens_total=40,
        tokens_prompt=30,
        tokens_completion=10,
        llm_calls=1,
        cost_usd=0.0001,
        tools_invoked=["graph_query.find_entity"],
    )
    scorecard = EvalScorecard(turns=[turn], tiers=build_tier_scorecards([turn]))
    markdown = render_readme_section(scorecard, scorecard)
    assert "Agents actually invoked" in markdown
    assert "Latency" in markdown
    assert "Tokens" in markdown
    assert "Cost" in markdown
    assert "graph_query" in markdown
    assert "Retrieval" in markdown
    assert "Synthesis" in markdown
    assert "Grounding" in markdown
    assert "provider: offline (stub answers — routing/retrieval only)" in markdown


def test_load_qa_cases_includes_multiturn_tier() -> None:
    from core.eval.harness import load_qa_cases, load_routing_cases

    cases = load_qa_cases()
    assert len(cases) == 50
    assert any(case.tier == "multi-turn" for case in cases)
    assert len(load_routing_cases()) == 9
    from core.eval.harness import routing_qa_cases

    assert len(routing_qa_cases(cases)) == 9


@pytest.mark.asyncio
async def test_run_executed_plan_eval_scores_golden_query() -> None:
    from core.eval.harness import run_executed_plan_eval

    turn = TurnSpec(
        query="What is the FastAPI class?",
        expected_agents=["graph_query"],
        expected_mode="rules",
        min_agents=1,
    )
    card = await run_executed_plan_eval([turn], tier="simple")
    assert card.hard_assertions_passed
    assert card.turns[0].executed_agents == ["graph_query"]
    assert card.turns[0].latency_ms >= 0


def test_format_budget_comparison_table() -> None:
    from core.eval.models import BudgetComparison, TruncationOrderScore
    from core.eval.report import format_budget_comparison

    turn = TurnScore(
        case_id="r1",
        tier="simple",
        query="What is the FastAPI class?",
        expected_agents=["graph_query"],
        executed_agents=["graph_query"],
        routing_mode="rules",
        agents_passed=True,
        citation_precision=1.0,
        citation_passed=True,
        groundedness=1.0,
        grounded_passed=True,
        entity_recall=1.0,
        entity_passed=True,
        degraded=False,
        evidence_only=False,
        refusal_ok=None,
        answer="FastAPI is the application class.",
        latency_ms=12,
        tokens_total=40,
        tools_invoked=["graph_query.find_entity"],
    )
    scorecard = EvalScorecard(turns=[turn], tiers=build_tier_scorecards([turn]))
    markdown = format_budget_comparison(
        BudgetComparison(
            enabled=scorecard,
            disabled=scorecard,
            truncation_orders=[
                TruncationOrderScore("snippets_then_lists", 1.0, 1.0),
                TruncationOrderScore("lists_then_snippets", 0.9, 0.9),
            ],
            chosen_truncation_order="snippets_then_lists",
            latency_p50_ms=10,
            latency_p95_ms=20,
            no_quality_drop=True,
        )
    )
    assert "did not drop" in markdown
    assert "snippets_then_lists" in markdown
    assert "p50/p95" in markdown


def test_parse_eval_provider_defaults_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.eval.harness import eval_llm_provider, parse_eval_provider
    from core.llm.offline_provider import OfflineProvider

    monkeypatch.delenv("EVAL_PROVIDER", raising=False)
    assert parse_eval_provider() == "offline"
    assert parse_eval_provider("LIVE") == "live"
    with pytest.raises(ValueError, match="offline"):
        parse_eval_provider("openai")
    assert isinstance(eval_llm_provider("offline"), OfflineProvider)


def test_eval_llm_provider_live_uses_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.eval.harness import eval_llm_provider

    sentinel = object()
    monkeypatch.setattr("core.llm.factory.build_llm_provider", lambda: sentinel)
    assert eval_llm_provider("live") is sentinel


def test_eval_llm_provider_live_refuses_stub_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.eval.harness import eval_llm_provider
    from core.llm.offline_provider import OfflineProvider

    monkeypatch.setattr("core.llm.factory.build_llm_provider", OfflineProvider)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        eval_llm_provider("live")


def test_readme_labels_live_and_offline_distinct() -> None:
    turn = TurnScore(
        case_id="r1",
        tier="simple",
        query="What is the FastAPI class?",
        expected_agents=["graph_query"],
        executed_agents=["graph_query"],
        routing_mode="rules",
        agents_passed=True,
        citation_precision=1.0,
        citation_passed=True,
        groundedness=1.0,
        grounded_passed=True,
        entity_recall=1.0,
        entity_passed=True,
        degraded=False,
        evidence_only=False,
        refusal_ok=None,
        answer="FastAPI is defined in fastapi/applications.py:1.",
        latency_ms=12,
        tokens_total=40,
        tools_invoked=["graph_query.find_entity"],
    )
    live = EvalScorecard(
        turns=[turn],
        tiers=build_tier_scorecards([turn]),
        provider="live",
    )
    offline = EvalScorecard(
        turns=[turn],
        tiers=build_tier_scorecards([turn]),
        provider="offline",
    )
    markdown = render_readme_section(live, live, companion=offline)
    assert "provider: live" in markdown
    assert "provider: offline (stub answers — routing/retrieval only)" in markdown
    assert markdown.index("### Live quality measurement") < markdown.index(
        "### Offline routing/retrieval"
    )
    assert "### Per-turn layer verdicts (live)" in markdown
    assert "Retrieval FAIL means" in markdown
    assert "Synthesis FAIL means" in markdown


def test_format_metrics_marks_unscored_quality_as_na() -> None:
    from core.eval.report import format_metrics

    text = format_metrics(
        [
            MetricScore(
                "citation_precision",
                0.0,
                True,
                "0 scored (9 excluded: no citations)",
                n=0,
                hits=0,
            )
        ]
    )
    assert "n/a" in text
    assert "1.00" not in text


def test_format_metrics_table() -> None:
    from core.eval.report import format_metrics

    text = format_metrics([MetricScore("hard_executed_agents", 1.0, True, "9/9", n=9, hits=9)])
    assert "hard_executed_agents" in text
    assert "PASS" in text


def test_write_readme_replaces_marked_section(tmp_path: Path) -> None:
    from core.eval.models import BudgetComparison, TruncationOrderScore

    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- BEGIN_EVAL_REPORT -->\nPLACEHOLDER\n<!-- END_EVAL_REPORT -->\n\n# After\n",
        encoding="utf-8",
    )
    turn = TurnScore(
        case_id="r1",
        tier="simple",
        query="What is the FastAPI class?",
        expected_agents=["graph_query"],
        executed_agents=["graph_query"],
        routing_mode="rules",
        agents_passed=True,
        citation_precision=1.0,
        citation_passed=True,
        groundedness=1.0,
        grounded_passed=True,
        entity_recall=1.0,
        entity_passed=True,
        degraded=False,
        evidence_only=False,
        refusal_ok=None,
        answer="FastAPI is the application class.",
        latency_ms=12,
        tokens_total=40,
        tools_invoked=["graph_query.find_entity"],
    )
    scorecard = EvalScorecard(turns=[turn], tiers=build_tier_scorecards([turn]))
    comparison = BudgetComparison(
        enabled=scorecard,
        disabled=scorecard,
        truncation_orders=[TruncationOrderScore("snippets_then_lists", 1.0, 1.0)],
        chosen_truncation_order="snippets_then_lists",
        latency_p50_ms=10,
        latency_p95_ms=20,
        no_quality_drop=True,
    )
    write_readme_section(scorecard, scorecard, readme_path=readme, budget_comparison=comparison)
    text = readme.read_text(encoding="utf-8")
    assert "PLACEHOLDER" not in text
    assert "What is the FastAPI class?" in text
    assert "did not drop" in text
    assert "# After" in text
    assert "provider: offline (stub answers — routing/retrieval only)" in text


def _turn(
    *,
    case_id: str = "s01",
    tier: str = "simple",
    agents_passed: bool = True,
    citation_passed: bool = True,
    groundedness: float | None = 1.0,
    grounded_passed: bool = True,
    entity_recall: float = 1.0,
    entity_passed: bool = True,
    degraded: bool = False,
    evidence_only: bool = False,
    retrieval_correctness: float = 1.0,
    retrieval_passed: bool = True,
    executed_agents: list[str] | None = None,
) -> TurnScore:
    return TurnScore(
        case_id=case_id,
        tier=tier,
        query="q",
        expected_agents=["graph_query"],
        executed_agents=executed_agents or ["graph_query"],
        routing_mode="rules",
        agents_passed=agents_passed,
        citation_precision=1.0,
        citation_passed=citation_passed,
        groundedness=groundedness,
        grounded_passed=grounded_passed,
        entity_recall=entity_recall,
        entity_passed=entity_passed,
        degraded=degraded,
        evidence_only=evidence_only,
        refusal_ok=None,
        retrieval_correctness=retrieval_correctness,
        retrieval_passed=retrieval_passed,
    )


def test_groundedness_does_not_zero_turn_for_one_unmatched_token() -> None:
    answer = "FastAPI is defined in fastapi/applications.py:1 and mentions OpenAPI."
    outputs = {
        "graph_query": {
            "output": {
                "candidates": [
                    {
                        "name": "FastAPI",
                        "file_path": "fastapi/applications.py",
                        "line_start": 1,
                    }
                ]
            }
        }
    }
    score, ungrounded = score_groundedness(answer, outputs)
    assert score is not None
    assert 0.5 < score < 1.0
    assert ungrounded

    turn = TurnSpec(
        query="What is the FastAPI class?",
        expected_entities=["FastAPI"],
        expected_files=["fastapi/applications.py"],
        expected_agents=["graph_query"],
        expected_mode="rules",
        min_agents=1,
    )
    scored = score_turn(
        "s01",
        "simple",
        turn,
        answer=answer,
        metadata={
            "tools_invoked": ["graph_query.find_entity"],
            "routing_mode": "rules",
            "tokens": {"total": 10, "prompt": 8, "completion": 2, "llm_calls": 1},
        },
        agent_outputs=outputs,
        quality=True,
    )
    assert scored.groundedness == score
    assert scored.grounded_passed == (score >= THRESHOLD_GROUNDEDNESS)
    assert scored.synthesis_verdict
    assert scored.retrieval_verdict


def test_retrieval_correctness_independent_of_entity_recall() -> None:
    outputs = {
        "graph_query": {
            "output": {
                "candidates": [
                    {
                        "name": "Depends",
                        "file_path": "tests/test_dependency_duplicates.py",
                        "line_start": 19,
                    }
                ]
            }
        }
    }
    score, missing, retrieved = score_retrieval_correctness(
        ["fastapi/dependencies/utils.py", "fastapi/param_functions.py"],
        outputs,
    )
    assert score == 0.0
    assert "fastapi/dependencies/utils.py" in missing
    assert "tests/test_dependency_duplicates.py" in retrieved
    recall, _missing = score_entity_recall("Depends is used here.", ["Depends"], outputs)
    assert recall == 1.0
    turn = TurnSpec(
        query="How does dependency injection work and show me examples from the codebase",
        expected_entities=["Depends"],
        expected_files=["fastapi/dependencies/utils.py", "fastapi/param_functions.py"],
        expected_agents=["graph_query", "code_analyst"],
        expected_mode="llm",
        min_agents=2,
    )
    scored = score_turn(
        "m01",
        "medium",
        turn,
        answer="Depends is used in tests/test_dependency_duplicates.py:19.",
        metadata={
            "tools_invoked": ["graph_query.find_entity", "code_analyst.explain_implementation"],
            "routing_mode": "llm",
            "tokens": {"total": 10, "prompt": 8, "completion": 2, "llm_calls": 1},
        },
        agent_outputs=outputs,
        quality=True,
    )
    assert scored.entity_recall == 1.0
    assert scored.entity_passed
    assert scored.retrieval_correctness == 0.0
    assert not scored.retrieval_passed
    assert not scored.retrieval_verdict
    assert scored.synthesis_verdict


def test_hard_non_trap_reports_rate_gate_stays_binary() -> None:
    from core.eval.harness import _hard_metrics

    ok = _turn(case_id="s01")
    degraded = _turn(case_id="c04", tier="complex", degraded=True, evidence_only=True)
    metrics = {item.name: item for item in _hard_metrics([ok] * 8 + [degraded])}
    metric = metrics["hard_non_trap_not_degraded"]
    assert metric.value == pytest.approx(8 / 9)
    assert metric.passed is False
    assert metric.n == 9
    assert metric.hits == 8
    assert "8/9" in metric.detail
    assert "c04" in metric.detail


def test_groundedness_metric_gates_mean_not_perfect_turns() -> None:
    from core.eval.harness import _quality_metrics

    low = _turn(case_id="s01", groundedness=0.50, grounded_passed=False)
    high = _turn(case_id="s02", groundedness=0.95, grounded_passed=True)
    metrics = {item.name: item for item in _quality_metrics([low, high])}
    grounded = metrics["groundedness"]
    assert grounded.value == pytest.approx(0.725)
    assert grounded.passed is (0.725 >= THRESHOLD_GROUNDEDNESS)
    assert grounded.n == 2
    assert "mean 0.72 vs gate 0.70" in grounded.detail
    assert "retrieval_correctness" in metrics


def test_layer_verdicts_distinguish_synthesis_timeout() -> None:
    from core.eval.report import format_query_table

    timed_out = _turn(case_id="c04", tier="complex", degraded=True, evidence_only=True)
    table = format_query_table([timed_out])
    assert "Retrieval" in table
    assert "Synthesis" in table
    assert "Grounding" in table
    assert "| PASS | FAIL | PASS |" in table
    assert "Result |" not in table.split("\n")[0]


def test_eval_live_makefile_runs_full_qa_set() -> None:
    makefile = Path(__file__).resolve().parents[2] / "Makefile"
    text = makefile.read_text(encoding="utf-8")
    live_block = text.split("eval-live:", 1)[1].split("\ntokens-report:", 1)[0]
    assert "EVAL_SAMPLE_ONLY" not in live_block
    assert "EVAL_PROVIDER=live" in live_block
    assert "bolt://127.0.0.1:7687" in live_block

