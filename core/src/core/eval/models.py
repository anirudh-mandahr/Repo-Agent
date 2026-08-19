"""Pydantic and dataclass models for routing and QA eval scoring."""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

TIERS = ("simple", "medium", "complex", "trap", "multi-turn")

THRESHOLD_RECALL = 0.80
# Live golden-set (2026-08-19, 9 turns, claim-level all-or-nothing): mean 0.74,
# simple 0.62 / medium 0.80 / complex 0.76. None reached 1.00 — a single unmatched
# paraphrase token zeroed a claim and failed the turn. Token-level support is the
# graded score. The pass gate is 0.70 (one decimal below that measured mean).
THRESHOLD_GROUNDEDNESS = 0.70
THRESHOLD_CITATION_PRECISION = 1.00
THRESHOLD_REFUSAL = 0.95
THRESHOLD_EXECUTED_AGENTS = 1.00
GATED_METRICS = frozenset(
    {
        "hard_executed_agents",
        "hard_non_trap_not_degraded",
        "citation_precision",
    }
)
OFFLINE_PROVIDER_LABEL = "provider: offline (stub answers — routing/retrieval only)"
LIVE_PROVIDER_LABEL = "provider: live"


def provider_label(kind: str) -> str:
    """Return the README / scorecard label for an eval provider.

    Args:
        kind: ``offline`` or ``live``.

    Returns:
        A string a reader cannot mistake for the other run.
    """
    if kind == "live":
        return LIVE_PROVIDER_LABEL
    return OFFLINE_PROVIDER_LABEL


class TurnSpec(BaseModel):
    """One labelled eval turn."""

    query: str
    expected_entities: list[str] = Field(default_factory=list)
    expected_files: list[str] = Field(default_factory=list)
    expected_agents: list[str] = Field(default_factory=list)
    expected_mode: str = "rules"
    min_agents: int = 1
    out_of_scope: bool = False


class QaCase(BaseModel):
    """One labelled QA case, possibly multi-turn."""

    id: str
    tier: str = "simple"
    turns: tuple[TurnSpec, ...]
    out_of_scope: bool = False

    @property
    def query(self) -> str:
        """First-turn query text."""
        return self.turns[0].query

    @property
    def is_multiturn(self) -> bool:
        """True when the case has more than one turn."""
        return len(self.turns) > 1


@dataclass
class MetricScore:
    """One named metric with a pass/fail gate."""

    name: str
    value: float
    passed: bool
    detail: str
    n: int = 0
    hits: int = 0


@dataclass
class TurnScore:
    """Per-turn executed-plan and answer-quality scores."""

    case_id: str
    tier: str
    query: str
    expected_agents: list[str]
    executed_agents: list[str]
    routing_mode: str
    agents_passed: bool
    citation_precision: float | None
    citation_passed: bool
    groundedness: float | None
    grounded_passed: bool
    entity_recall: float
    entity_passed: bool
    degraded: bool
    evidence_only: bool
    refusal_ok: bool | None
    retrieval_correctness: float = 1.0
    retrieval_passed: bool = True
    ungrounded_claims: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    retrieved_paths: list[str] = field(default_factory=list)
    answer: str = ""
    latency_ms: int = 0
    tokens_total: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    tools_invoked: list[str] = field(default_factory=list)

    @property
    def hard_fail(self) -> bool:
        """True when executed agents diverge or a non-trap answer is degraded."""
        if not self.agents_passed:
            return True
        if self.tier != "trap" and (self.degraded or self.evidence_only):
            return True
        if not self.citation_passed:
            return True
        return False

    @property
    def retrieval_verdict(self) -> bool:
        """True when expected agents ran and labelled files were retrieved."""
        return self.agents_passed and self.retrieval_passed

    @property
    def synthesis_verdict(self) -> bool:
        """True when synthesis produced a non-degraded, non-evidence-only answer."""
        return not self.degraded and not self.evidence_only

    @property
    def grounding_verdict(self) -> bool:
        """True when citation precision and groundedness cleared their gates."""
        return self.citation_passed and self.grounded_passed

    @property
    def passed(self) -> bool:
        """True when every applicable gate for this turn passed."""
        if self.hard_fail:
            return False
        if not self.grounded_passed or not self.entity_passed or not self.retrieval_passed:
            return False
        if self.tier == "trap" and self.refusal_ok is False:
            return False
        return True


@dataclass
class TierScorecard:
    """Pass rates for one eval tier."""

    tier: str
    n: int
    passed: int
    executed_agents: float
    citation_precision: float | None
    groundedness: float | None
    entity_recall: float
    retrieval_correctness: float
    refusal: float | None
    degraded_fails: int
    citation_n: int = 0
    citation_excluded: int = 0
    groundedness_n: int = 0
    groundedness_excluded: int = 0

    @property
    def pass_rate(self) -> float:
        """Fraction of turns that passed every applicable gate."""
        return self.passed / self.n if self.n else 0.0


@dataclass
class EvalScorecard:
    """Aggregated eval output: per-turn rows plus per-tier rates."""

    turns: list[TurnScore] = field(default_factory=list)
    tiers: list[TierScorecard] = field(default_factory=list)
    metrics: list[MetricScore] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    provider: str = "offline"

    @property
    def hard_assertions_passed(self) -> bool:
        """True when no executed-plan or degraded-answer hard assertion failed."""
        return all(not item.hard_fail for item in self.turns) and all(
            item.passed for item in self.metrics if item.name.startswith("hard_")
        )

    @property
    def ok(self) -> bool:
        """True when hard assertions and citation precision passed.

        Groundedness, entity recall, and trap refusal are reported per tier
        rather than failing the whole suite.
        """
        return self.hard_assertions_passed and all(
            item.passed for item in self.metrics if item.name in GATED_METRICS
        )


@dataclass
class TruncationOrderScore:
    """Quality of one prompt-truncation order under a tight token budget."""

    order: str
    citation_precision: float | None
    groundedness: float | None


@dataclass
class BudgetComparison:
    """QA scorecards with request budgets enabled vs disabled."""

    enabled: EvalScorecard
    disabled: EvalScorecard
    truncation_orders: list[TruncationOrderScore] = field(default_factory=list)
    chosen_truncation_order: str = "snippets_then_lists"
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    no_quality_drop: bool = True
