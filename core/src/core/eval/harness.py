"""Load eval JSONL and run ``handle_query`` scoring."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from core.analysis.service import CodeAnalystService, GraphLookup
from core.eval.models import (
    THRESHOLD_EXECUTED_AGENTS,
    THRESHOLD_GROUNDEDNESS,
    THRESHOLD_REFUSAL,
    TIERS,
    EvalScorecard,
    MetricScore,
    QaCase,
    TierScorecard,
    TurnScore,
    TurnSpec,
)
from core.eval.scoring import score_turn
from core.graph.client import GraphClient
from core.llm.offline_provider import OfflineProvider
from core.llm.provider import LLMProvider
from core.memory import ConversationContext
from core.orchestration.service import OrchestratorService
from core.querying.service import GraphQueryService
from core.settings import AnalysisSettings, OrchestratorSettings

ROOT = Path(__file__).resolve().parents[4]
QA_PATH = ROOT / "evals" / "qa.jsonl"
ROUTING_PATH = ROOT / "evals" / "routing.jsonl"

EvalProviderKind = Literal["offline", "live"]


def _as_str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def turn_from_payload(payload: Mapping[str, Any], *, out_of_scope: bool) -> TurnSpec:
    """Build a :class:`TurnSpec` from one JSONL object.

    Args:
        payload: Raw JSON object.
        out_of_scope: Whether the parent case is a trap.

    Returns:
        Parsed turn.
    """
    expected_agents = _as_str_list(payload.get("expected_agents"))
    return TurnSpec(
        query=str(payload["query"]),
        expected_entities=_as_str_list(payload.get("expected_entities")),
        expected_files=_as_str_list(payload.get("expected_files")),
        expected_agents=expected_agents,
        expected_mode=str(payload.get("expected_mode") or "rules"),
        min_agents=int(payload.get("min_agents") or max(1, len(expected_agents))),
        out_of_scope=out_of_scope,
    )


def load_qa_cases(path: Path = QA_PATH) -> list[QaCase]:
    """Load labelled QA rows, including ordered multi-turn lists.

    Args:
        path: JSONL file of labelled cases.

    Returns:
        Parsed cases in file order.
    """
    cases: list[QaCase] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        out_of_scope = bool(payload.get("out_of_scope"))
        turns_raw = payload.get("turns")
        if isinstance(turns_raw, list) and turns_raw:
            turns = tuple(turn_from_payload(item, out_of_scope=out_of_scope) for item in turns_raw)
        else:
            turns = (turn_from_payload(payload, out_of_scope=out_of_scope),)
        cases.append(
            QaCase(
                id=str(payload.get("id") or payload.get("query")),
                tier=str(payload.get("tier") or "simple"),
                turns=turns,
                out_of_scope=out_of_scope,
            )
        )
    return cases


def load_routing_cases(path: Path = ROUTING_PATH) -> list[TurnSpec]:
    """Load the original nine routing golden-set rows.

    Args:
        path: JSONL file of routing cases.

    Returns:
        One :class:`TurnSpec` per routing row.
    """
    cases: list[TurnSpec] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        payload = json.loads(raw)
        cases.append(turn_from_payload(payload, out_of_scope=False))
    return cases


class MemoryClient:
    """In-memory conversation store used by the eval harness."""

    def __init__(self) -> None:
        """Create an empty turn log."""
        self._turns: list[tuple[str, str]] = []

    async def get_context(self, session_id: str, token_budget: int = 3000) -> ConversationContext:
        """Return prior turns for this session.

        Args:
            session_id: Conversation id (ignored; one buffer per client).
            token_budget: Unused token cap.

        Returns:
            Conversation context built from appended turns.
        """
        _ = session_id, token_budget
        from core.memory import ConversationTurn

        turns = [
            ConversationTurn(
                id=index + 1,
                role=role,
                content=content,
                created_at="2026-01-01T00:00:00Z",
                token_estimate=max(1, len(content) // 4),
            )
            for index, (role, content) in enumerate(self._turns)
        ]
        return ConversationContext(recent_turns=turns)

    async def get_cached_response(self, cache_key: str) -> None:
        """Always miss so evals score a fresh executed plan.

        Args:
            cache_key: Unused cache key.
        """
        _ = cache_key
        return None

    async def cache_response(self, cache_key: str, response_json: Any) -> None:
        """No-op cache write.

        Args:
            cache_key: Unused cache key.
            response_json: Unused payload.
        """
        _ = cache_key, response_json

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Record one conversation turn.

        Args:
            session_id: Unused session id.
            role: ``user`` or ``assistant``.
            content: Turn text.
        """
        _ = session_id
        self._turns.append((role, content))


class GraphClientAdapter:
    """Sync graph-query service wrapped as an async orchestrator client."""

    def __init__(self, service: GraphQueryService) -> None:
        """Bind a graph query service.

        Args:
            service: Live graph query service.
        """
        self._service = service

    async def get_statistics(self) -> Any:
        """Return graph statistics."""
        return self._service.get_statistics()

    async def find_entity(self, name: str, entity_type: str | None = None) -> dict[str, Any] | None:
        """Look up an entity by name.

        Args:
            name: Entity name.
            entity_type: Optional label filter.

        Returns:
            First match wrapped so callers still see ``matches``, or ``None``.
        """
        result = self._service.find_entity(name, entity_type)
        if not result.matches:
            return None
        return result.model_dump(mode="json")

    async def get_dependencies(self, name: str) -> dict[str, Any]:
        """Return inbound/outbound dependencies.

        Args:
            name: Qualified name.

        Returns:
            Neighbor payload.
        """
        return self._service.get_dependencies(name).model_dump(mode="json")

    async def get_dependents(self, name: str) -> dict[str, Any]:
        """Return dependents of ``name``.

        Args:
            name: Qualified name.

        Returns:
            Neighbor payload.
        """
        return self._service.get_dependents(name).model_dump(mode="json")

    async def find_related(self, name: str, relationship_type: str) -> dict[str, Any]:
        """Return related nodes of one relationship type.

        Args:
            name: Qualified name.
            relationship_type: Graph relationship type.

        Returns:
            Related-node payload.
        """
        return self._service.find_related(name, relationship_type).model_dump(mode="json")

    async def trace_imports(self, module: str, depth: int = 5) -> dict[str, Any]:
        """Trace import paths.

        Args:
            module: Module name.
            depth: Traversal depth.

        Returns:
            Import-trace payload.
        """
        return self._service.trace_imports(module, depth).model_dump(mode="json")


class CodeAnalystClientAdapter:
    """Code analyst service wrapped as an async orchestrator client."""

    def __init__(self, service: CodeAnalystService) -> None:
        """Bind a code analyst service.

        Args:
            service: Live code analyst service.
        """
        self._service = service

    async def get_code_snippet(self, **kwargs: Any) -> dict[str, Any]:
        """Return a source snippet.

        Args:
            **kwargs: Snippet lookup arguments.

        Returns:
            Snippet payload.
        """
        result = await self._service.get_code_snippet(**kwargs)
        return result.model_dump(mode="json")

    async def explain_implementation(self, qualified_name: str) -> dict[str, Any]:
        """Explain one symbol.

        Args:
            qualified_name: Fully qualified name.

        Returns:
            Explanation payload.
        """
        result = await self._service.explain_implementation(qualified_name)
        return result.model_dump(mode="json")

    async def analyze_function(self, qualified_name: str) -> dict[str, Any]:
        """Analyze one function.

        Args:
            qualified_name: Fully qualified name.

        Returns:
            Function-analysis payload.
        """
        result = await self._service.analyze_function(qualified_name)
        return result.model_dump(mode="json")

    async def analyze_class(self, qualified_name: str) -> dict[str, Any]:
        """Analyze one class.

        Args:
            qualified_name: Fully qualified name.

        Returns:
            Class-analysis payload.
        """
        result = await self._service.analyze_class(qualified_name)
        return result.model_dump(mode="json")

    async def compare_implementations(self, name_a: str, name_b: str) -> dict[str, Any]:
        """Compare two symbols.

        Args:
            name_a: First qualified name.
            name_b: Second qualified name.

        Returns:
            Comparison payload.
        """
        result = await self._service.compare_implementations(name_a, name_b)
        return result.model_dump(mode="json")

    async def find_patterns(self, pattern: str) -> dict[str, Any]:
        """Find instances of a named pattern.

        Args:
            pattern: Pattern name.

        Returns:
            Pattern payload.
        """
        result = await self._service.find_patterns(pattern)
        return result.model_dump(mode="json")


class IndexerClient:
    """Eval indexer that records the call without rewriting the graph."""

    def __init__(self) -> None:
        """Create a no-op indexer."""
        self.calls: list[str | None] = []

    async def index_repository(self, repo_url: str | None = None) -> dict[str, Any]:
        """Pretend to reindex so executed-plan scoring sees the indexer tool.

        Args:
            repo_url: Optional clone URL.

        Returns:
            Skipped index report.
        """
        self.calls.append(repo_url)
        return {"status": "skipped", "detail": "eval harness does not reindex"}


class StubGraphQueryClient:
    """Deterministic graph client for executed-plan routing evals."""

    async def get_statistics(self) -> Any:
        """Return a stable index version."""
        return type("Stats", (), {"index_version": "eval-stub"})()

    async def find_entity(self, name: str, entity_type: str | None = None) -> dict[str, Any]:
        """Return a synthetic hit named after the query term.

        Args:
            name: Lookup term.
            entity_type: Unused label filter.

        Returns:
            Entity payload with a fake FastAPI path.
        """
        _ = entity_type
        return {
            "name": name,
            "qualified_name": f"fastapi.{name}",
            "file_path": "fastapi/applications.py",
            "line_start": 1,
            "line_end": 40,
        }

    async def get_dependencies(self, name: str) -> dict[str, Any]:
        """Return empty dependencies.

        Args:
            name: Qualified name.

        Returns:
            Neighbor payload.
        """
        return {"name": name, "neighbors": []}

    async def get_dependents(self, name: str) -> dict[str, Any]:
        """Return empty dependents.

        Args:
            name: Qualified name.

        Returns:
            Neighbor payload.
        """
        return {"name": name, "neighbors": []}

    async def find_related(self, name: str, relationship_type: str) -> dict[str, Any]:
        """Return empty related nodes.

        Args:
            name: Qualified name.
            relationship_type: Relationship type.

        Returns:
            Related-node payload.
        """
        return {"name": name, "relationship_type": relationship_type, "neighbors": []}

    async def trace_imports(self, module: str, depth: int = 5) -> dict[str, Any]:
        """Return an empty import trace.

        Args:
            module: Module name.
            depth: Traversal depth.

        Returns:
            Import-trace payload.
        """
        _ = depth
        return {"module": module, "paths": []}


class StubCodeAnalystClient:
    """Deterministic code-analyst client for executed-plan routing evals."""

    async def get_code_snippet(self, **kwargs: Any) -> dict[str, Any]:
        """Return a stub snippet.

        Args:
            **kwargs: Snippet lookup arguments.

        Returns:
            Snippet payload.
        """
        path = str(kwargs.get("file_path") or "fastapi/applications.py")
        start = int(kwargs.get("line_start") or 1)
        end = int(kwargs.get("line_end") or 20)
        return {
            "file_path": path,
            "line_start": start,
            "line_end": end,
            "text": f"snippet {path}:{start}-{end}",
            "error": None,
        }

    async def explain_implementation(self, qualified_name: str) -> dict[str, Any]:
        """Return a stub explanation.

        Args:
            qualified_name: Fully qualified name.

        Returns:
            Explanation payload.
        """
        return {"qualified_name": qualified_name, "explanation": f"explained {qualified_name}"}

    async def analyze_function(self, qualified_name: str) -> dict[str, Any]:
        """Return a stub function analysis.

        Args:
            qualified_name: Fully qualified name.

        Returns:
            Function-analysis payload.
        """
        return {"qualified_name": qualified_name, "summary": f"analyzed {qualified_name}"}

    async def analyze_class(self, qualified_name: str) -> dict[str, Any]:
        """Return a stub class analysis.

        Args:
            qualified_name: Fully qualified name.

        Returns:
            Class-analysis payload.
        """
        return {"qualified_name": qualified_name, "summary": f"analyzed class {qualified_name}"}

    async def compare_implementations(self, name_a: str, name_b: str) -> dict[str, Any]:
        """Return a stub comparison.

        Args:
            name_a: First name.
            name_b: Second name.

        Returns:
            Comparison payload.
        """
        return {"name_a": name_a, "name_b": name_b, "summary": f"compared {name_a} and {name_b}"}

    async def find_patterns(self, pattern: str) -> dict[str, Any]:
        """Return a stub pattern result.

        Args:
            pattern: Pattern name.

        Returns:
            Pattern payload.
        """
        return {"pattern": pattern, "instances": []}


def orchestrator_clients(
    *,
    graph_query: Any,
    code_analyst: Any,
    memory: MemoryClient | None = None,
    indexer: IndexerClient | None = None,
) -> Any:
    """Build the specialist client bundle ``handle_query`` expects.

    Args:
        graph_query: Graph query client.
        code_analyst: Code analyst client.
        memory: Optional memory client.
        indexer: Optional indexer client.

    Returns:
        A namespace with the four specialist clients.
    """
    return type(
        "Clients",
        (),
        {
            "memory": memory or MemoryClient(),
            "graph_query": graph_query,
            "code_analyst": code_analyst,
            "indexer": indexer or IndexerClient(),
        },
    )()


def stub_clients(memory: MemoryClient | None = None) -> Any:
    """Return in-process stub specialists for executed-plan checks.

    Args:
        memory: Optional shared memory client.

    Returns:
        Orchestrator client bundle.
    """
    return orchestrator_clients(
        graph_query=StubGraphQueryClient(),
        code_analyst=StubCodeAnalystClient(),
        memory=memory,
    )


def _async_graph_lookup(service: GraphQueryService) -> GraphLookup:
    async def _lookup(cypher: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return service.execute_query(cypher, params).rows

    return _lookup


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_exclude_none(values: Sequence[float | None]) -> tuple[float | None, int, int]:
    scored = [item for item in values if item is not None]
    excluded = len(values) - len(scored)
    if not scored:
        return None, 0, excluded
    return sum(scored) / len(scored), len(scored), excluded


def parse_eval_provider(value: str | None = None) -> EvalProviderKind:
    """Parse ``offline`` / ``live`` from an explicit value or ``EVAL_PROVIDER``.

    Args:
        value: Provider name. ``EVAL_PROVIDER`` when omitted.

    Returns:
        ``offline`` (default) or ``live``.

    Raises:
        ValueError: When the value is not one of the two allowed names.
    """
    raw = (
        (value if value is not None else os.environ.get("EVAL_PROVIDER", "offline")).strip().lower()
    )
    if raw not in {"offline", "live"}:
        raise ValueError(f"EVAL_PROVIDER must be 'offline' or 'live', got {raw!r}")
    return raw  # type: ignore[return-value]


def eval_llm_provider(kind: EvalProviderKind = "offline") -> LLMProvider:
    """Return the LLM backend for an eval run.

    ``offline`` is deterministic and key-free (CI default). ``live`` goes
    through ``core.llm.factory.build_llm_provider`` and refuses to silently
    fall back to the stub, so a live scorecard cannot be a stub run.

    Args:
        kind: ``offline`` or ``live``.

    Returns:
        An ``LLMProvider`` implementation.

    Raises:
        RuntimeError: When ``live`` is requested but no API key is exported.
    """
    if kind == "offline":
        return OfflineProvider()
    from core.llm.factory import build_llm_provider

    provider = build_llm_provider()
    if isinstance(provider, OfflineProvider):
        raise RuntimeError(
            "EVAL_PROVIDER=live requires OPENROUTER_API_KEY exported in the "
            "environment (repo .env files do not load that secret)"
        )
    return provider


def routing_qa_cases(cases: Sequence[QaCase] | None = None) -> list[QaCase]:
    """Return single-turn QA cases that match the nine routing golden queries.

    Args:
        cases: QA cases. Loaded from ``evals/qa.jsonl`` when omitted.

    Returns:
        One case per routing query, in routing-file order.
    """
    loaded = list(cases) if cases is not None else load_qa_cases()
    wanted = [turn.query for turn in load_routing_cases()]
    by_query = {case.query: case for case in loaded if not case.is_multiturn}
    return [by_query[query] for query in wanted if query in by_query]


def build_tier_scorecards(turns: Sequence[TurnScore]) -> list[TierScorecard]:
    """Aggregate per-turn scores into the five-tier scorecard.

    Args:
        turns: Scored eval turns.

    Returns:
        One row per known tier (empty tiers included).
    """
    cards: list[TierScorecard] = []
    for tier in TIERS:
        rows = [item for item in turns if item.tier == tier]
        refusal_rows = [item for item in rows if item.refusal_ok is not None]
        cite_mean, cite_n, cite_excl = _mean_exclude_none(
            [item.citation_precision for item in rows]
        )
        ground_mean, ground_n, ground_excl = _mean_exclude_none(
            [item.groundedness for item in rows]
        )
        cards.append(
            TierScorecard(
                tier=tier,
                n=len(rows),
                passed=sum(1 for item in rows if item.passed),
                executed_agents=_mean([1.0 if item.agents_passed else 0.0 for item in rows]),
                citation_precision=cite_mean,
                groundedness=ground_mean,
                entity_recall=_mean([item.entity_recall for item in rows]),
                retrieval_correctness=_mean([item.retrieval_correctness for item in rows]),
                refusal=(
                    _mean([1.0 if item.refusal_ok else 0.0 for item in refusal_rows])
                    if refusal_rows
                    else None
                ),
                degraded_fails=sum(
                    1
                    for item in rows
                    if item.tier != "trap" and (item.degraded or item.evidence_only)
                ),
                citation_n=cite_n,
                citation_excluded=cite_excl,
                groundedness_n=ground_n,
                groundedness_excluded=ground_excl,
            )
        )
    return cards


def _hard_metrics(turns: Sequence[TurnScore]) -> list[MetricScore]:
    agent_total = len(turns)
    agent_hits = sum(1 for item in turns if item.agents_passed)
    degraded_fails = [
        item.case_id
        for item in turns
        if item.tier != "trap" and (item.degraded or item.evidence_only)
    ]
    agent_value = agent_hits / agent_total if agent_total else 1.0
    non_trap = [item for item in turns if item.tier != "trap"]
    degraded_n = len(non_trap)
    degraded_hits = degraded_n - len(degraded_fails)
    degraded_value = degraded_hits / degraded_n if degraded_n else 1.0
    metrics = [
        MetricScore(
            name="hard_executed_agents",
            value=agent_value,
            passed=agent_value >= THRESHOLD_EXECUTED_AGENTS and agent_hits == agent_total,
            detail=f"{agent_hits}/{agent_total} turns matched expected_agents via tools_invoked",
            n=agent_total,
            hits=agent_hits,
        ),
        MetricScore(
            name="hard_non_trap_not_degraded",
            value=degraded_value,
            passed=not degraded_fails,
            detail=(
                f"{degraded_hits}/{degraded_n} non-trap turns not evidence-only/degraded"
                if not degraded_fails
                else (
                    f"{degraded_hits}/{degraded_n} non-trap turns not evidence-only/degraded; "
                    "degraded=" + ",".join(degraded_fails[:8])
                )
            ),
            n=degraded_n,
            hits=degraded_hits,
        ),
    ]
    trap_rows = [item for item in turns if item.tier == "trap"]
    if trap_rows:
        refusal_hits = sum(1 for item in trap_rows if item.refusal_ok)
        refusal_value = refusal_hits / len(trap_rows)
        metrics.append(
            MetricScore(
                name="refusal_accuracy",
                value=refusal_value,
                passed=refusal_value >= THRESHOLD_REFUSAL,
                detail=f"{refusal_hits}/{len(trap_rows)} traps refused",
                n=len(trap_rows),
                hits=refusal_hits,
            )
        )
    return metrics


def _quality_metrics(turns: Sequence[TurnScore]) -> list[MetricScore]:
    rows = [item for item in turns if item.tier != "trap"]
    if not rows:
        return []

    def _optional_metric(
        name: str,
        values: Sequence[float | None],
        passed_flags: Sequence[bool],
        empty_reason: str,
    ) -> MetricScore:
        scored_flags = [
            flag for value, flag in zip(values, passed_flags, strict=True) if value is not None
        ]
        scored = [value for value in values if value is not None]
        excluded = len(values) - len(scored)
        hits = sum(1 for flag in scored_flags if flag)
        value = _mean(scored)
        detail = f"{hits}/{len(scored)} scored non-trap turns ({excluded} excluded: {empty_reason})"
        return MetricScore(
            name=name,
            value=value,
            passed=len(scored) == 0 or hits == len(scored),
            detail=detail,
            n=len(scored),
            hits=hits,
        )

    def _required_metric(
        name: str, values: Sequence[float], passed_flags: Sequence[bool]
    ) -> MetricScore:
        hits = sum(1 for flag in passed_flags if flag)
        return MetricScore(
            name=name,
            value=_mean(list(values)),
            passed=hits == len(rows),
            detail=f"{hits}/{len(rows)} non-trap turns",
            n=len(rows),
            hits=hits,
        )

    ground_values = [item.groundedness for item in rows]
    ground_flags = [item.grounded_passed for item in rows]
    ground_scored = [value for value in ground_values if value is not None]
    ground_excluded = len(ground_values) - len(ground_scored)
    ground_hits = sum(
        1
        for value, flag in zip(ground_values, ground_flags, strict=True)
        if value is not None and flag
    )
    ground_mean = _mean(ground_scored)
    ground_detail = (
        f"mean {ground_mean:.2f} vs gate {THRESHOLD_GROUNDEDNESS:.2f}; "
        f"{ground_hits}/{len(ground_scored)} turns ≥ gate "
        f"({ground_excluded} excluded: no checkable claims)"
    )

    return [
        _optional_metric(
            "citation_precision",
            [item.citation_precision for item in rows],
            [item.citation_passed for item in rows],
            "no citations",
        ),
        MetricScore(
            name="groundedness",
            value=ground_mean,
            passed=not ground_scored or ground_mean >= THRESHOLD_GROUNDEDNESS,
            detail=ground_detail,
            n=len(ground_scored),
            hits=ground_hits,
        ),
        _required_metric(
            "entity_recall",
            [item.entity_recall for item in rows],
            [item.entity_passed for item in rows],
        ),
        _required_metric(
            "retrieval_correctness",
            [item.retrieval_correctness for item in rows],
            [item.retrieval_passed for item in rows],
        ),
    ]


async def _score_case_turns(
    case: QaCase,
    *,
    orchestrator: OrchestratorService,
    clients_factory: Any,
    graph_client: GraphClient | None,
    repo_root: Path | None,
    quality: bool,
) -> list[TurnScore]:
    memory = MemoryClient()
    clients = clients_factory(memory)
    scores: list[TurnScore] = []
    for index, turn in enumerate(case.turns):
        started = time.perf_counter()
        result = await orchestrator.handle_query(
            turn.query,
            f"eval-{case.id}",
            clients=clients,
            correlation_id=f"eval-{case.id}-{index}",
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        scores.append(
            score_turn(
                case.id,
                case.tier,
                turn,
                answer=result.answer,
                metadata=result.metadata,
                agent_outputs=result.agent_outputs,
                graph_client=graph_client,
                repo_root=repo_root,
                latency_ms=latency_ms,
                quality=quality,
            )
        )
    return scores


async def run_executed_plan_eval(
    turns: Sequence[TurnSpec],
    *,
    case_id_prefix: str = "routing",
    tier: str = "complex",
) -> EvalScorecard:
    """Score executed agents for labelled turns using stub specialists.

    Args:
        turns: Routing golden-set turns.
        case_id_prefix: Prefix for synthetic case ids.
        tier: Tier label applied to each turn.

    Returns:
        Scorecard with executed-plan hard assertions (quality gates skipped).
    """
    orchestrator = OrchestratorService(
        OfflineProvider(),
        settings=OrchestratorSettings.from_env(),
    )
    scores: list[TurnScore] = []
    for index, turn in enumerate(turns):
        case = QaCase(
            id=f"{case_id_prefix}-{index + 1}",
            tier=tier,
            turns=(turn,),
            out_of_scope=turn.out_of_scope,
        )
        scores.extend(
            await _score_case_turns(
                case,
                orchestrator=orchestrator,
                clients_factory=stub_clients,
                graph_client=None,
                repo_root=None,
                quality=False,
            )
        )
    return EvalScorecard(
        turns=scores,
        tiers=build_tier_scorecards(scores),
        metrics=_hard_metrics(scores),
        provider="offline",
    )


async def run_qa_eval(
    cases: Sequence[QaCase] | None = None,
    *,
    graph_client: GraphClient | None = None,
    repo_root: Path | None = None,
    settings: OrchestratorSettings | None = None,
    provider: EvalProviderKind = "offline",
) -> EvalScorecard:
    """Run labelled QA cases through ``handle_query`` and score the executed plan.

    Args:
        cases: QA cases. Loaded from ``evals/qa.jsonl`` when omitted.
        graph_client: Optional open Neo4j client. Created from env when omitted
            and ``NEO4J_URI`` is set.
        repo_root: Indexed repository root.
        settings: Orchestrator settings. Loaded from env when omitted.
        provider: ``offline`` (default, stub answers, CI-safe) or ``live``
            (``core.llm.factory``). Live refuses to fall back to the stub.

    Returns:
        Full scorecard. Quality metrics are skipped when Neo4j is unavailable.
    """
    loaded = list(cases) if cases is not None else load_qa_cases()
    report = EvalScorecard(provider=provider)
    neo4j_uri = os.environ.get("NEO4J_URI")
    resolved_root = repo_root or Path(AnalysisSettings.from_env().repo_root)
    orch_settings = settings or OrchestratorSettings.from_env()
    llm = eval_llm_provider(provider)
    orchestrator = OrchestratorService(
        llm,
        settings=orch_settings,
    )

    if graph_client is None and not neo4j_uri:
        report.lines.append("NEO4J_URI is required for citation, groundedness, and entity metrics")
        turns: list[TurnScore] = []
        for case in loaded:
            turns.extend(
                await _score_case_turns(
                    case,
                    orchestrator=orchestrator,
                    clients_factory=stub_clients,
                    graph_client=None,
                    repo_root=None,
                    quality=False,
                )
            )
        report.turns = turns
        report.tiers = build_tier_scorecards(turns)
        report.metrics = _hard_metrics(turns)
        report.metrics.extend(
            [
                MetricScore("citation_precision", 0.0, False, "skipped: NEO4J_URI unset"),
                MetricScore("groundedness", 0.0, False, "skipped: NEO4J_URI unset"),
                MetricScore("entity_recall", 0.0, False, "skipped: NEO4J_URI unset"),
                MetricScore("retrieval_correctness", 0.0, False, "skipped: NEO4J_URI unset"),
            ]
        )
        return report

    owns_client = graph_client is None
    client = graph_client or GraphClient()
    try:
        if owns_client:
            client.verify_connectivity()
        graph_service = GraphQueryService(client)
        analysis_service = CodeAnalystService(
            llm,
            _async_graph_lookup(graph_service),
            repo_root=resolved_root,
        )

        def _clients(memory: MemoryClient) -> Any:
            return orchestrator_clients(
                graph_query=GraphClientAdapter(graph_service),
                code_analyst=CodeAnalystClientAdapter(analysis_service),
                memory=memory,
            )

        scored: list[TurnScore] = []
        for case in loaded:
            scored.extend(
                await _score_case_turns(
                    case,
                    orchestrator=orchestrator,
                    clients_factory=_clients,
                    graph_client=client,
                    repo_root=resolved_root,
                    quality=True,
                )
            )
        report.turns = scored
        report.tiers = build_tier_scorecards(scored)
        report.metrics = _hard_metrics(scored) + _quality_metrics(scored)
    finally:
        if owns_client:
            client.close()
    return report
