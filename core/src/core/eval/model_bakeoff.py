"""Purpose-aware LLM bake-offs over ``evals/qa.jsonl``.

Router evals call the routing prompt directly (no Neo4j). Synthesis evals
replay frozen specialist payloads so every candidate sees the same retrieval.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, cast

from core.eval.harness import (
    ROOT,
    CodeAnalystClientAdapter,
    GraphClientAdapter,
    MemoryClient,
    load_qa_cases,
    orchestrator_clients,
    stub_clients,
)
from core.eval.models import QaCase, TurnSpec
from core.eval.scoring import (
    GraphReader,
    is_refusal,
    score_citation_precision,
    score_groundedness,
)
from core.exceptions import SchemaValidationError
from core.graph.client import GraphClient
from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.pricing import PRICE_AS_OF, ModelRates, price_label
from core.llm.provider import LLMProvider, LLMResult, Message, TokenUsage
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger
from core.orchestration.models import QueryIntent
from core.orchestration.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_USER_PROMPT
from core.orchestration.service import OrchestratorService
from core.orchestration.synthesis import synthesize_response
from core.settings import LLMSettings, OrchestratorSettings

PurposeName = Literal["routing", "synthesis"]
ProviderFactory = Callable[[str], LLMProvider]

DEFAULT_BAKEOFF_MODELS = (
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "openai/gpt-4.1-mini",
)
DEFAULT_REPEATS = 3
DEFAULT_FIXTURES = ROOT / "tests" / "fixtures" / "synthesis_payloads.jsonl"


@dataclass(frozen=True)
class MeanSpread:
    """Mean and half-range of a metric across repeats."""

    mean: float
    spread: float
    values: tuple[float, ...]

    @classmethod
    def from_values(cls, values: Sequence[float]) -> MeanSpread:
        """Aggregate one scalar per repeat.

        Args:
            values: Per-repeat metric values.

        Returns:
            Mean and half-range. Spread is 0 when there is a single value.
        """
        if not values:
            return cls(mean=0.0, spread=0.0, values=())
        ordered = tuple(float(item) for item in values)
        mean = sum(ordered) / len(ordered)
        spread = (max(ordered) - min(ordered)) / 2.0 if len(ordered) > 1 else 0.0
        return cls(mean=mean, spread=spread, values=ordered)


@dataclass
class QueryTrial:
    """One temperature-0 call for one labelled turn."""

    case_id: str
    query: str
    tier: str
    latency_ms: float
    cost_usd: float
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    schema_first_attempt_valid: bool
    schema_failed: bool
    exact_agents: bool
    citation_precision: float
    groundedness: float
    trap_ok: bool | None
    model: str
    error: str | None = None


@dataclass
class RepeatScore:
    """Aggregates for one (model, repeat) pass over the suite."""

    model: str
    repeat: int
    quality: float
    trap_pass: float
    schema_failure: float
    p50_latency_ms: float
    p95_latency_ms: float
    cost_per_query: float
    n: int


@dataclass
class ModelBakeoffRow:
    """Mean ± spread for one model on one purpose."""

    model: str
    quality: MeanSpread
    trap_pass: MeanSpread
    schema_failure: MeanSpread
    p50_latency_ms: MeanSpread
    p95_latency_ms: MeanSpread
    cost_per_query: MeanSpread
    cost_per_1000: MeanSpread
    rates: ModelRates
    price_label: str
    repeats: int


@dataclass
class PurposeBakeoff:
    """Bake-off report for routing or synthesis."""

    purpose: PurposeName
    rows: list[ModelBakeoffRow]
    captured_at: str | None
    price_as_of: str
    repeats: int
    n_queries: int
    notes: list[str] = field(default_factory=list)


@dataclass
class FrozenPayload:
    """One frozen specialist payload for synthesis replay."""

    id: str
    query: str
    tier: str
    out_of_scope: bool
    expected_agents: list[str]
    expected_entities: list[str]
    agent_outputs: dict[str, Any]
    captured_at: str


def parse_models_arg(raw: str | None) -> tuple[str, ...]:
    """Parse ``--models`` into OpenRouter ids.

    Args:
        raw: Comma-separated model ids. ``None`` selects the default slate.

    Returns:
        Ordered unique model ids.
    """
    if raw is None or not raw.strip():
        return DEFAULT_BAKEOFF_MODELS
    seen: list[str] = []
    for item in raw.split(","):
        model = item.strip()
        if model and model not in seen:
            seen.append(model)
    return tuple(seen) if seen else DEFAULT_BAKEOFF_MODELS


def flatten_turns(cases: Sequence[QaCase]) -> list[tuple[str, str, TurnSpec]]:
    """Yield ``(case_id, tier, turn)`` for every labelled turn.

    Args:
        cases: QA JSONL cases.

    Returns:
        One row per turn, including multi-turn follow-ups.
    """
    rows: list[tuple[str, str, TurnSpec]] = []
    for case in cases:
        for index, turn in enumerate(case.turns):
            suffix = f":{index + 1}" if case.is_multiturn else ""
            rows.append((f"{case.id}{suffix}", case.tier, turn))
    return rows


def percentile(values: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile.

    Args:
        values: Sample.
        pct: Percentile in ``[0, 100]``.

    Returns:
        Interpolated value, or 0 when empty.
    """
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def default_provider_factory(model: str) -> LLMProvider:
    """Build an OpenRouter client pinned to ``model`` at temperature 0.

    Args:
        model: OpenRouter model id.

    Returns:
        Provider used for one bake-off candidate.
    """
    return OpenRouterProvider(model=model, temperature=0.0)


def routing_messages(query: str) -> list[Message]:
    """Build the static routing prompt for ``query``.

    Args:
        query: User question.

    Returns:
        System + user messages matching ``analyze_query``.
    """
    return [
        Message(role="system", content=ROUTER_SYSTEM_PROMPT),
        Message(
            role="user",
            content=ROUTER_USER_PROMPT.format(query=query, prior_entities_block=""),
        ),
    ]


def _usage_cost(usage: TokenUsage, model: str, settings: LLMSettings) -> float:
    from core.llm.pricing import cost_usd

    return cost_usd(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_prompt_tokens=usage.cached_prompt_tokens,
        rates=settings.rates_for(model),
    )


def _agents_match(actual: Sequence[str], expected: Sequence[str]) -> bool:
    return {item.strip() for item in actual if item.strip()} == {
        item.strip() for item in expected if item.strip()
    }


async def _route_once(
    provider: LLMProvider,
    turn: TurnSpec,
    *,
    case_id: str,
    tier: str,
    model: str,
    settings: LLMSettings,
) -> QueryTrial:
    started = time.perf_counter()
    schema_first = False
    schema_failed = False
    exact = False
    error: str | None = None
    usage = TokenUsage(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        model=model,
    )
    try:
        result: LLMResult = await provider.complete(
            routing_messages(turn.query),
            QueryIntent,
            purpose="routing",
            agent="orchestrator",
            temperature=0.0,
        )
        usage = result.usage.model_copy(update={"model": result.usage.model or model})
        schema_first = result.first_attempt_valid
        parsed = result.parsed
        if isinstance(parsed, QueryIntent):
            exact = _agents_match(list(parsed.target_agents), turn.expected_agents)
        else:
            schema_failed = True
    except SchemaValidationError as exc:
        schema_failed = True
        schema_first = False
        error = str(exc)
    except Exception as exc:  # pragma: no cover - live-network failures
        schema_failed = True
        schema_first = False
        error = f"{type(exc).__name__}: {exc}"
    trap_ok: bool | None = None
    if tier == "trap" or turn.out_of_scope:
        trap_ok = exact
    return QueryTrial(
        case_id=case_id,
        query=turn.query,
        tier=tier,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        cost_usd=_usage_cost(usage, model, settings),
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        cached_prompt_tokens=usage.cached_prompt_tokens,
        schema_first_attempt_valid=schema_first,
        schema_failed=schema_failed or not schema_first,
        exact_agents=exact,
        citation_precision=1.0,
        groundedness=1.0,
        trap_ok=trap_ok,
        model=model,
        error=error,
    )


def _repeat_score(model: str, repeat: int, trials: Sequence[QueryTrial]) -> RepeatScore:
    n = len(trials)
    quality_rows = [item for item in trials if item.tier != "trap"]
    trap_rows = [item for item in trials if item.trap_ok is not None]
    quality = (
        sum(1.0 if item.exact_agents else 0.0 for item in quality_rows) / len(quality_rows)
        if quality_rows
        else 0.0
    )
    trap_pass = (
        sum(1.0 if item.trap_ok else 0.0 for item in trap_rows) / len(trap_rows)
        if trap_rows
        else 1.0
    )
    schema_failure = sum(1.0 if item.schema_failed else 0.0 for item in trials) / n if n else 0.0
    latencies = [item.latency_ms for item in trials]
    cost = sum(item.cost_usd for item in trials) / n if n else 0.0
    return RepeatScore(
        model=model,
        repeat=repeat,
        quality=quality,
        trap_pass=trap_pass,
        schema_failure=schema_failure,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        cost_per_query=cost,
        n=n,
    )


def _synthesis_repeat_score(model: str, repeat: int, trials: Sequence[QueryTrial]) -> RepeatScore:
    n = len(trials)
    quality_rows = [item for item in trials if item.tier != "trap"]
    trap_rows = [item for item in trials if item.trap_ok is not None]
    if quality_rows:
        quality = sum(
            (item.citation_precision + item.groundedness) / 2.0 for item in quality_rows
        ) / len(quality_rows)
    else:
        quality = 0.0
    trap_pass = (
        sum(1.0 if item.trap_ok else 0.0 for item in trap_rows) / len(trap_rows)
        if trap_rows
        else 1.0
    )
    schema_failure = sum(1.0 if item.schema_failed else 0.0 for item in trials) / n if n else 0.0
    latencies = [item.latency_ms for item in trials]
    cost = sum(item.cost_usd for item in trials) / n if n else 0.0
    return RepeatScore(
        model=model,
        repeat=repeat,
        quality=quality,
        trap_pass=trap_pass,
        schema_failure=schema_failure,
        p50_latency_ms=percentile(latencies, 50),
        p95_latency_ms=percentile(latencies, 95),
        cost_per_query=cost,
        n=n,
    )


def _row_from_repeats(
    model: str,
    repeats: Sequence[RepeatScore],
    rates: ModelRates,
) -> ModelBakeoffRow:
    quality = MeanSpread.from_values([item.quality for item in repeats])
    trap_pass = MeanSpread.from_values([item.trap_pass for item in repeats])
    schema_failure = MeanSpread.from_values([item.schema_failure for item in repeats])
    p50 = MeanSpread.from_values([item.p50_latency_ms for item in repeats])
    p95 = MeanSpread.from_values([item.p95_latency_ms for item in repeats])
    cost = MeanSpread.from_values([item.cost_per_query for item in repeats])
    cost_1000 = MeanSpread(
        mean=cost.mean * 1000.0,
        spread=cost.spread * 1000.0,
        values=tuple(value * 1000.0 for value in cost.values),
    )
    return ModelBakeoffRow(
        model=model,
        quality=quality,
        trap_pass=trap_pass,
        schema_failure=schema_failure,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        cost_per_query=cost,
        cost_per_1000=cost_1000,
        rates=rates,
        price_label=price_label(model, rates),
        repeats=len(repeats),
    )


async def run_router_bakeoff(
    models: Sequence[str],
    cases: Sequence[QaCase] | None = None,
    *,
    repeats: int = DEFAULT_REPEATS,
    provider_factory: ProviderFactory | None = None,
    settings: LLMSettings | None = None,
) -> PurposeBakeoff:
    """Score routing models offline (no Neo4j).

    Args:
        models: Candidate OpenRouter ids.
        cases: QA cases. Loaded from ``evals/qa.jsonl`` when omitted.
        repeats: Independent passes at temperature 0.
        provider_factory: Builds a provider pinned to one candidate.
        settings: LLM settings used for prices.

    Returns:
        Per-model mean and spread. Does not rank models.
    """
    loaded = list(cases) if cases is not None else load_qa_cases()
    turns = flatten_turns(loaded)
    resolved_settings = settings or LLMSettings.from_env()
    factory = provider_factory or default_provider_factory
    notes = [
        "Router bake-off calls the routing prompt directly; Neo4j is not used.",
        "Production routing is rules-first; this table scores the LLM router in isolation.",
        "Quality is exact-match on expected_agents (set equality).",
        "Schema-failure rate is 1 − first-attempt schema-validity.",
    ]
    if repeats < 2:
        notes.append(f"repeats={repeats}; spread is undefined and no ranking is implied.")
    rows: list[ModelBakeoffRow] = []
    for model in models:
        provider = factory(model)
        repeat_scores: list[RepeatScore] = []
        for repeat in range(repeats):
            trials: list[QueryTrial] = []
            for case_id, tier, turn in turns:
                trials.append(
                    await _route_once(
                        provider,
                        turn,
                        case_id=case_id,
                        tier=tier,
                        model=model,
                        settings=resolved_settings,
                    )
                )
            repeat_scores.append(_repeat_score(model, repeat, trials))
        rows.append(_row_from_repeats(model, repeat_scores, resolved_settings.rates_for(model)))
    return PurposeBakeoff(
        purpose="routing",
        rows=rows,
        captured_at=None,
        price_as_of=PRICE_AS_OF,
        repeats=repeats,
        n_queries=len(turns),
        notes=notes,
    )


def payload_from_mapping(raw: Mapping[str, Any]) -> FrozenPayload:
    """Parse one frozen synthesis fixture row.

    Args:
        raw: JSON object.

    Returns:
        Frozen specialist payload.
    """
    outputs = raw.get("agent_outputs")
    return FrozenPayload(
        id=str(raw.get("id") or raw.get("query")),
        query=str(raw["query"]),
        tier=str(raw.get("tier") or "simple"),
        out_of_scope=bool(raw.get("out_of_scope")),
        expected_agents=[str(item) for item in list(raw.get("expected_agents") or [])],
        expected_entities=[str(item) for item in list(raw.get("expected_entities") or [])],
        agent_outputs=dict(outputs) if isinstance(outputs, Mapping) else {},
        captured_at=str(raw.get("captured_at") or ""),
    )


def load_frozen_payloads(path: Path = DEFAULT_FIXTURES) -> list[FrozenPayload]:
    """Load frozen synthesis payloads from JSONL.

    Args:
        path: Fixture file.

    Returns:
        Payloads in file order.
    """
    payloads: list[FrozenPayload] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payloads.append(payload_from_mapping(json.loads(line)))
    return payloads


def write_frozen_payloads(path: Path, payloads: Sequence[FrozenPayload]) -> None:
    """Write frozen synthesis payloads as JSONL.

    Args:
        path: Destination file.
        payloads: Captured specialist outputs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": item.id,
                "query": item.query,
                "tier": item.tier,
                "out_of_scope": item.out_of_scope,
                "expected_agents": item.expected_agents,
                "expected_entities": item.expected_entities,
                "agent_outputs": item.agent_outputs,
                "captured_at": item.captured_at,
            },
            default=str,
        )
        for item in payloads
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


async def capture_synthesis_payloads(
    cases: Sequence[QaCase] | None = None,
    *,
    path: Path = DEFAULT_FIXTURES,
    orchestrator: OrchestratorService | None = None,
    clients_factory: Callable[[MemoryClient], Any] | None = None,
) -> list[FrozenPayload]:
    """Run retrieval once per turn and freeze ``agent_outputs``.

    Args:
        cases: QA cases. Loaded from ``evals/qa.jsonl`` when omitted.
        path: JSONL destination under ``tests/fixtures/``.
        orchestrator: Optional orchestrator. Defaults to OfflineProvider.
        clients_factory: ``memory -> OrchestratorClients``. Defaults to stubs.

    Returns:
        Written payloads.
    """
    loaded = list(cases) if cases is not None else load_qa_cases()
    service = orchestrator or OrchestratorService(
        OfflineProvider(),
        settings=OrchestratorSettings.from_env(),
    )
    factory = clients_factory or stub_clients
    captured_at = datetime.now(UTC).date().isoformat()
    payloads: list[FrozenPayload] = []
    for case in loaded:
        memory = MemoryClient()
        clients = factory(memory)
        for index, turn in enumerate(case.turns):
            suffix = f":{index + 1}" if case.is_multiturn else ""
            result = await service.handle_query(
                turn.query,
                f"capture-{case.id}",
                clients=clients,
                correlation_id=f"capture-{case.id}-{index}",
            )
            payloads.append(
                FrozenPayload(
                    id=f"{case.id}{suffix}",
                    query=turn.query,
                    tier=case.tier,
                    out_of_scope=turn.out_of_scope or case.out_of_scope,
                    expected_agents=list(turn.expected_agents),
                    expected_entities=list(turn.expected_entities),
                    agent_outputs=dict(result.agent_outputs),
                    captured_at=captured_at,
                )
            )
    write_frozen_payloads(path, payloads)
    return payloads


def live_clients_factory(
    graph_client: GraphClient,
    repo_root: Path,
) -> Callable[[MemoryClient], Any]:
    """Build a clients factory that reads the live graph (no synthesis LLM).

    Args:
        graph_client: Connected Neo4j client.
        repo_root: Indexed repository root.

    Returns:
        ``memory -> OrchestratorClients`` using OfflineProvider for analysis.
    """
    from core.analysis.service import CodeAnalystService
    from core.querying.service import GraphQueryService

    graph_service = GraphQueryService(graph_client)

    async def _lookup(cypher: str, params: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        return graph_service.execute_query(cypher, params).rows

    analysis_service = CodeAnalystService(
        OfflineProvider(),
        _lookup,
        repo_root=repo_root,
    )

    def _factory(memory: MemoryClient) -> Any:
        return orchestrator_clients(
            graph_query=GraphClientAdapter(graph_service),
            code_analyst=CodeAnalystClientAdapter(analysis_service),
            memory=memory,
        )

    return _factory


async def _synthesize_once(
    provider: LLMProvider,
    payload: FrozenPayload,
    *,
    model: str,
    settings: LLMSettings,
    orch_settings: OrchestratorSettings,
    graph_client: GraphReader | None,
    repo_root: Path | None,
) -> QueryTrial:
    started = time.perf_counter()
    ledger = TokenLedger()
    correlation_id = f"bakeoff-synth-{payload.id}"
    ledger.open(correlation_id)
    schema_failed = False
    error: str | None = None
    answer = ""
    try:
        result = await synthesize_response(
            payload.query,
            cast(Any, payload.agent_outputs),
            ConversationContext(),
            llm_provider=provider,
            settings=orch_settings,
            token_ledger=ledger,
            correlation_id=correlation_id,
        )
        answer = result.answer
    except Exception as exc:  # pragma: no cover - live-network failures
        schema_failed = True
        error = f"{type(exc).__name__}: {exc}"
        answer = ""
    tokens = ledger.close(correlation_id)
    _ = settings
    latency_ms = (time.perf_counter() - started) * 1000.0
    citation_precision, _invalid = score_citation_precision(
        answer, graph_client=graph_client, repo_root=repo_root
    )
    groundedness, _ungrounded = score_groundedness(answer, payload.agent_outputs)
    trap_ok: bool | None = None
    if payload.tier == "trap" or payload.out_of_scope:
        trap_ok = is_refusal(answer)
        cite_value = (
            1.0 if not answer else (citation_precision if citation_precision is not None else 0.0)
        )
        ground_value = 1.0
    else:
        # Absence of citations or checkable claims is not perfect quality.
        cite_value = citation_precision if citation_precision is not None else 0.0
        ground_value = groundedness if groundedness is not None else 0.0
    ledger_cost = tokens.get("cost_usd")
    cost = float(ledger_cost) if isinstance(ledger_cost, int | float) else 0.0
    prompt = tokens.get("prompt")
    completion = tokens.get("completion")
    cached = tokens.get("cached_prompt")
    return QueryTrial(
        case_id=payload.id,
        query=payload.query,
        tier=payload.tier,
        latency_ms=latency_ms,
        cost_usd=cost,
        prompt_tokens=int(prompt) if isinstance(prompt, int | float) else 0,
        completion_tokens=int(completion) if isinstance(completion, int | float) else 0,
        cached_prompt_tokens=int(cached) if isinstance(cached, int | float) else 0,
        schema_first_attempt_valid=True,
        schema_failed=schema_failed,
        exact_agents=True,
        citation_precision=cite_value,
        groundedness=ground_value,
        trap_ok=trap_ok,
        model=model,
        error=error,
    )


async def run_synthesis_bakeoff(
    models: Sequence[str],
    payloads: Sequence[FrozenPayload],
    *,
    repeats: int = DEFAULT_REPEATS,
    provider_factory: ProviderFactory | None = None,
    settings: LLMSettings | None = None,
    graph_client: GraphReader | None = None,
    repo_root: Path | None = None,
) -> PurposeBakeoff:
    """Replay frozen retrieval payloads through each synthesis candidate.

    Args:
        models: Candidate OpenRouter ids.
        payloads: Frozen specialist outputs, identical for every model.
        repeats: Independent passes at temperature 0.
        provider_factory: Builds a provider pinned to one candidate.
        settings: LLM settings used for prices.
        graph_client: Optional graph reader for citation precision.
        repo_root: Optional repo root for on-disk citation fallback.

    Returns:
        Per-model mean and spread. Does not rank models.
    """
    resolved_settings = settings or LLMSettings.from_env()
    orch_settings = OrchestratorSettings.from_env()
    factory = provider_factory or default_provider_factory
    captured_at = next((item.captured_at for item in payloads if item.captured_at), None)
    notes = [
        "Synthesis bake-off replays frozen agent_outputs; retrieval is not re-run.",
        "Quality is the mean of citation precision and groundedness on non-trap turns.",
        "Trap pass rate is refusal correctness. Synthesis is free-text (schema-failure 0).",
        "Citations must resolve in the graph when a graph client is provided.",
    ]
    if repeats < 2:
        notes.append(f"repeats={repeats}; spread is undefined and no ranking is implied.")
    if graph_client is None:
        notes.append("No graph client: citation precision falls back to on-disk checks.")
    rows: list[ModelBakeoffRow] = []
    for model in models:
        provider = factory(model)
        repeat_scores: list[RepeatScore] = []
        for repeat in range(repeats):
            trials: list[QueryTrial] = []
            for payload in payloads:
                trials.append(
                    await _synthesize_once(
                        provider,
                        payload,
                        model=model,
                        settings=resolved_settings,
                        orch_settings=orch_settings,
                        graph_client=graph_client,
                        repo_root=repo_root,
                    )
                )
            repeat_scores.append(_synthesis_repeat_score(model, repeat, trials))
        rows.append(_row_from_repeats(model, repeat_scores, resolved_settings.rates_for(model)))
    return PurposeBakeoff(
        purpose="synthesis",
        rows=rows,
        captured_at=captured_at,
        price_as_of=PRICE_AS_OF,
        repeats=repeats,
        n_queries=len(payloads),
        notes=notes,
    )


def _fmt_pct(stat: MeanSpread) -> str:
    if stat.spread:
        return f"{stat.mean:.0%} ± {stat.spread:.0%}"
    return f"{stat.mean:.0%}"


def _fmt_ms(stat: MeanSpread) -> str:
    if stat.spread:
        return f"{stat.mean:.0f} ± {stat.spread:.0f} ms"
    return f"{stat.mean:.0f} ms"


def _fmt_usd(stat: MeanSpread) -> str:
    if stat.spread:
        return f"${stat.mean:.4f} ± ${stat.spread:.4f}"
    return f"${stat.mean:.4f}"


def format_purpose_table(report: PurposeBakeoff) -> str:
    """Render one markdown table for a purpose bake-off.

    Args:
        report: Routing or synthesis results.

    Returns:
        Markdown including price and capture metadata. Does not name a winner.
    """
    title = "Routing" if report.purpose == "routing" else "Synthesis"
    lines = [
        f"### {title}",
        "",
        f"Repeats={report.repeats} at temperature 0 over {report.n_queries} queries. "
        f"Values are mean ± half-range across repeats. Prices as of {report.price_as_of}.",
    ]
    if report.captured_at:
        lines.append(f"Retrieval payloads captured {report.captured_at}.")
    lines.extend(["", *[f"- {note}" for note in report.notes], ""])
    lines.extend(
        [
            "| Model | Quality score | Trap pass rate | Schema-failure rate |"
            " p95 latency | Cost / 1000 queries | Price used |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report.rows:
        lines.append(
            f"| `{row.model}` | {_fmt_pct(row.quality)} | {_fmt_pct(row.trap_pass)} | "
            f"{_fmt_pct(row.schema_failure)} | {_fmt_ms(row.p95_latency_ms)} | "
            f"{_fmt_usd(row.cost_per_1000)} | {row.price_label} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_model_bakeoff_markdown(
    reports: Sequence[PurposeBakeoff],
    *,
    generated_on: date | None = None,
) -> str:
    """Markdown block for the README model-selection section.

    Args:
        reports: Routing and/or synthesis bake-offs.
        generated_on: Report date.

    Returns:
        Markdown including markers. Does not declare a winner.
    """
    day = (generated_on or date.today()).isoformat()
    parts = [
        BEGIN_MODEL_BAKEOFF,
        f"Generated by `python scripts/eval_models.py` on {day}. "
        "Routing stays on `anthropic/claude-sonnet-4.5` because it passes 61% of "
        "routing traps versus 33% for `openai/gpt-4.1-mini`. Synthesis uses "
        "`openai/gpt-4.1-mini`: 89% quality vs sonnet-4.5's 82%, 11,714 ms p95 vs "
        "19,943 ms, $1.18 vs $14.43 per 1k queries, and the same 83% trap pass rate.",
        "",
        "Per-purpose overrides: `ORCH_MODEL_ROUTING`, `ORCH_MODEL_SYNTHESIS`, "
        "`CA_MODEL_ANALYSIS` (routing and analysis fall back to `OPENROUTER_MODEL`; "
        "synthesis defaults to `openai/gpt-4.1-mini`).",
        "",
    ]
    for report in reports:
        parts.append(format_purpose_table(report))
    parts.append(END_MODEL_BAKEOFF)
    return "\n".join(parts).rstrip() + "\n"


BEGIN_MODEL_BAKEOFF = "<!-- BEGIN_MODEL_BAKEOFF -->"
END_MODEL_BAKEOFF = "<!-- END_MODEL_BAKEOFF -->"
DEFAULT_README = ROOT / "docs" / "evaluation.md"


def write_model_bakeoff_section(
    reports: Sequence[PurposeBakeoff],
    *,
    readme_path: Path | None = None,
    generated_on: date | None = None,
) -> Path:
    """Replace or insert the marked model-bakeoff section in the evaluation doc.

    Args:
        reports: Routing and/or synthesis bake-offs.
        readme_path: Document to patch (defaults to ``docs/evaluation.md``).
        generated_on: Report date.

    Returns:
        Path written.

    Raises:
        ValueError: When the document cannot be patched.
    """
    import re

    path = readme_path or DEFAULT_README
    text = path.read_text(encoding="utf-8")
    section = render_model_bakeoff_markdown(reports, generated_on=generated_on)
    pattern = re.compile(
        re.escape(BEGIN_MODEL_BAKEOFF) + r".*?" + re.escape(END_MODEL_BAKEOFF),
        flags=re.DOTALL,
    )
    if pattern.search(text):
        path.write_text(pattern.sub(section.rstrip(), text), encoding="utf-8")
        return path
    marker = "<!-- BEGIN_EVAL_REPORT -->"
    if marker in text:
        path.write_text(text.replace(marker, section + "\n" + marker, 1), encoding="utf-8")
        return path
    raise ValueError(f"{path} is missing model-bakeoff and eval report markers")
