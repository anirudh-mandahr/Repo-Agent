"""Compare eval quality with request budgets on/off and choose truncation order."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.eval.harness import load_qa_cases, run_qa_eval
from core.eval.model_bakeoff import percentile
from core.eval.models import (
    BudgetComparison,
    EvalScorecard,
    QaCase,
    TruncationOrderScore,
)
from core.eval.scoring import GraphReader, score_citation_precision, score_groundedness
from core.llm.offline_provider import OfflineProvider
from core.orchestration.prompt_budget import (
    DEFAULT_TRUNCATION_ORDER,
    TruncationOrder,
    apply_prompt_budget,
)
from core.settings import OrchestratorSettings

_ORDERS: tuple[TruncationOrder, TruncationOrder] = (
    "snippets_then_lists",
    "lists_then_snippets",
)


def quality_dropped(enabled: EvalScorecard, disabled: EvalScorecard) -> bool:
    """True when any tier lost citation precision or groundedness with budgets on.

    Args:
        enabled: Scorecard with request budgets enabled.
        disabled: Scorecard with request budgets disabled.

    Returns:
        Whether a per-tier quality metric dropped. Tiers where a metric was
        excluded (``None``) are skipped — empty output is not treated as 1.00.
    """
    by_disabled = {item.tier: item for item in disabled.tiers}
    for row in enabled.tiers:
        baseline = by_disabled.get(row.tier)
        if baseline is None or baseline.n == 0:
            continue
        if (
            row.citation_precision is not None
            and baseline.citation_precision is not None
            and row.citation_precision + 1e-9 < baseline.citation_precision
        ):
            return True
        if (
            row.groundedness is not None
            and baseline.groundedness is not None
            and row.groundedness + 1e-9 < baseline.groundedness
        ):
            return True
    return False


def choose_truncation_order(scores: Sequence[TruncationOrderScore]) -> TruncationOrder:
    """Pick the truncation order with the higher combined quality.

    Args:
        scores: Per-order citation precision and groundedness.

    Returns:
        The winning order. Ties keep every tied candidate; the first in
        ``_ORDERS`` that matches a tied winner is returned so the choice is
        determined by the measured scores rather than an untested assumption.
        Unmeasurable dimensions (``None``) are omitted from the combination
        rather than treated as 0% or 1.00.
    """
    if not scores:
        return DEFAULT_TRUNCATION_ORDER

    def _combo(item: TruncationOrderScore) -> float:
        parts = [
            value for value in (item.citation_precision, item.groundedness) if value is not None
        ]
        return sum(parts) / len(parts) if parts else 0.0

    best = max(_combo(item) for item in scores)
    tied = {item.order for item in scores if abs(_combo(item) - best) < 1e-9}
    for order in _ORDERS:
        if order in tied:
            return order
    return DEFAULT_TRUNCATION_ORDER


async def score_truncation_orders(
    payload: dict[str, Any],
    *,
    query: str,
    budget: int = 400,
    graph_client: GraphReader | None = None,
    repo_root: Path | None = None,
) -> list[TruncationOrderScore]:
    """Score both truncation orders on one oversized specialist payload.

    Citations in the synthesis Sources footer come from the (trimmed) payload.
    ``score_citation_precision`` without a graph or repo root cannot resolve
    those ``file:line`` coordinates and would report 0% next to a groundedness
    figure — that is not a measurement. Callers must pass ``repo_root`` and/or
    ``graph_client`` so citations are validated the same way as the QA scorer.
    When neither backend is available, citation precision is ``None``
    (excluded) rather than 0%.

    Args:
        payload: Agent outputs that exceed ``budget``.
        query: User question used to measure the prompt.
        budget: Tight synthesis prompt token budget.
        graph_client: Optional Neo4j reader used to resolve citations.
        repo_root: Optional indexed repository root for on-disk citation checks.

    Returns:
        One score per truncation order.
    """
    scores: list[TruncationOrderScore] = []
    for order in _ORDERS:
        trimmed, _truncation = apply_prompt_budget(
            payload,
            query=query,
            session_context_block="",
            budget=budget,
            order=order,
        )
        from core.memory import ConversationContext
        from core.orchestration.synthesis import synthesize_response

        result = await synthesize_response(
            query,
            trimmed,  # type: ignore[arg-type]
            ConversationContext(),
            llm_provider=OfflineProvider(),
            settings=OrchestratorSettings(
                synthesis_prompt_token_budget=10_000_000,
                prompt_truncation_order=order,
            ),
        )
        citation, _invalid = score_citation_precision(
            result.answer, graph_client=graph_client, repo_root=repo_root
        )
        grounded, _ungrounded = score_groundedness(result.answer, trimmed)
        scores.append(
            TruncationOrderScore(
                order=order,
                citation_precision=citation,
                groundedness=grounded,
            )
        )
    return scores


async def run_budget_comparison(
    cases: Sequence[QaCase] | None = None,
    *,
    enabled: EvalScorecard | None = None,
    truncation_payload: dict[str, Any] | None = None,
    truncation_query: str = "Compare FastAPI and APIRouter implementations in the codebase",
    graph_client: GraphReader | None = None,
    repo_root: Path | None = None,
) -> BudgetComparison:
    """Run the QA scorer with request budgets enabled and disabled.

    Args:
        cases: QA cases. Loaded from ``evals/qa.jsonl`` when omitted.
        enabled: Optional already-run scorecard for the budgets-on setting.
        truncation_payload: Optional oversized payload for order selection.
        truncation_query: Query used when scoring truncation orders.
        graph_client: Optional graph reader for truncation citation checks.
        repo_root: Optional repo root for truncation citation checks. When
            omitted, ``AnalysisSettings.repo_root`` is used if that path exists.

    Returns:
        Both scorecards, latency percentiles, and the chosen truncation order.
    """
    from core.settings import AnalysisSettings

    loaded = list(cases) if cases is not None else load_qa_cases()
    if enabled is None:
        enabled = await run_qa_eval(
            loaded,
            settings=OrchestratorSettings(request_budgets_enabled=True),
            provider="offline",
        )
    disabled = await run_qa_eval(
        loaded,
        settings=OrchestratorSettings(request_budgets_enabled=False),
        provider="offline",
    )
    latencies = [float(item.latency_ms) for item in enabled.turns]
    resolved_root = repo_root
    if resolved_root is None:
        candidate = Path(AnalysisSettings.from_env().repo_root)
        if candidate.exists():
            resolved_root = candidate
    orders = await score_truncation_orders(
        truncation_payload or _default_truncation_payload(),
        query=truncation_query,
        graph_client=graph_client,
        repo_root=resolved_root,
    )
    chosen = choose_truncation_order(orders)
    return BudgetComparison(
        enabled=enabled,
        disabled=disabled,
        truncation_orders=orders,
        chosen_truncation_order=chosen,
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        no_quality_drop=not quality_dropped(enabled, disabled),
    )


def _default_truncation_payload() -> dict[str, Any]:
    huge = "class FastAPI:\n    " + ("x" * 40_000)
    return {
        "graph_query": {
            "ok": True,
            "output": {
                "queried_entities": ["FastAPI"],
                "entities": [
                    {
                        "qualified_name": "fastapi.applications.FastAPI",
                        "file_path": "fastapi/applications.py",
                        "line_start": 10,
                        "line_end": 40,
                    }
                ],
                "candidates": [
                    {"qualified_name": f"Entity{i}", "file_path": f"mod{i}.py"} for i in range(40)
                ],
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
                        "text": huge,
                        "error": None,
                    }
                ],
            },
        },
    }
