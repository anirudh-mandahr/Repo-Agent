"""Outer request budget spanning routing, specialist calls, and synthesis."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from core.observability.ledger import TokenLedger
from core.settings import OrchestratorSettings

BudgetCeiling = Literal["deadline", "tokens", "cost"]

_UNLIMITED_DEADLINE_S = 86_400.0
# Unmeasured engineering defaults, not bake-off results. The bake-off recorded
# p95 latency per model, not a latency-versus-prompt-size slope. A fitted slope
# from per-case bake-off latency and prompt tokens would validate both values.
REFERENCE_SYNTHESIS_PROMPT_TOKENS = 4_000
SYNTHESIS_EXTRA_SECONDS_PER_TOKEN = 0.002


def scaled_synthesis_reserve_s(
    prompt_tokens: int,
    *,
    base_reserve_s: float,
) -> float:
    """Wall-clock seconds to reserve for synthesis given estimated prompt tokens.

    The configured reserve is the floor (typical prompt). Extra tokens beyond
    :data:`REFERENCE_SYNTHESIS_PROMPT_TOKENS` add a linear prefill allowance.

    Args:
        prompt_tokens: Estimated system+user synthesis prompt tokens.
        base_reserve_s: Configured ``ORCH_SYNTHESIS_RESERVE_S`` floor.

    Returns:
        Non-negative seconds. Never below ``base_reserve_s`` when that floor
        is positive.
    """
    floor = max(0.0, base_reserve_s)
    extra_tokens = max(0, int(prompt_tokens) - REFERENCE_SYNTHESIS_PROMPT_TOKENS)
    return floor + extra_tokens * SYNTHESIS_EXTRA_SECONDS_PER_TOKEN


@dataclass
class RequestBudget:
    """Wall-clock deadline plus token and cost ceilings for one ``handle_query``.

    Distinct from the per-prompt synthesis token budget. Spend is read from the
    token ledger's ``cost_usd`` / ``total`` rather than recomputed here.
    """

    deadline_monotonic: float
    token_ceiling: int | None
    cost_usd_max: float | None
    safety_margin_s: float
    min_synthesis_timeout_s: float
    max_synthesis_timeout_s: float
    synthesis_reserve_s: float = 0.0
    exhausted: BudgetCeiling | None = None
    _base_synthesis_reserve_s: float = 0.0
    _clock: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Keep the configured reserve floor when callers only set the effective value."""
        if self._base_synthesis_reserve_s <= 0.0 and self.synthesis_reserve_s > 0.0:
            self._base_synthesis_reserve_s = self.synthesis_reserve_s

    @classmethod
    def from_settings(
        cls,
        settings: OrchestratorSettings,
        *,
        now: float | None = None,
        clock: object | None = None,
    ) -> RequestBudget:
        """Build a budget from orchestrator settings.

        Args:
            settings: Timeouts and request-level ceilings.
            now: Optional monotonic timestamp for tests.
            clock: Optional monotonic clock.

        Returns:
            A budget whose deadline starts at ``now``.
        """
        tick = clock if callable(clock) else time.monotonic
        started = now if now is not None else tick()
        enabled = settings.request_budgets_enabled
        deadline_s = (
            settings.request_deadline_s if enabled else _UNLIMITED_DEADLINE_S
        )
        token_ceiling = (
            settings.request_token_budget
            if enabled and settings.request_token_budget > 0
            else None
        )
        cost_max = (
            settings.request_cost_usd_max
            if enabled and settings.request_cost_usd_max > 0
            else None
        )
        reserve = max(0.0, settings.synthesis_reserve_s)
        return cls(
            deadline_monotonic=started + max(0.0, deadline_s),
            token_ceiling=token_ceiling,
            cost_usd_max=cost_max,
            safety_margin_s=max(0.0, settings.synthesis_safety_margin_s),
            min_synthesis_timeout_s=max(0.0, settings.synthesis_min_timeout_s),
            max_synthesis_timeout_s=max(0.0, settings.synthesis_timeout_s),
            synthesis_reserve_s=reserve,
            _base_synthesis_reserve_s=reserve,
            _clock=tick,
        )

    def remaining_s(self, now: float | None = None) -> float:
        """Seconds left before the wall-clock deadline.

        Args:
            now: Optional monotonic timestamp.

        Returns:
            Non-negative remaining seconds.
        """
        current = now if now is not None else self._clock()
        return max(0.0, self.deadline_monotonic - current)

    def specialist_remaining_s(self, now: float | None = None) -> float:
        """Seconds the plan phase may still spend on specialist calls.

        Args:
            now: Optional monotonic timestamp.

        Returns:
            Remaining wall-clock minus the synthesis reserve, floored at 0.
        """
        return max(0.0, self.remaining_s(now) - self.synthesis_reserve_s)

    def apply_prompt_reserve(self, prompt_tokens: int) -> float:
        """Raise the synthesis reserve to match estimated prompt size.

        Never shrinks below the configured ``ORCH_SYNTHESIS_RESERVE_S`` floor.
        A zero floor (tests that disable the reserve) is left unchanged so
        settings validation stays the source of the constant.

        Args:
            prompt_tokens: Estimated synthesis prompt tokens after compacting
                and the per-prompt token budget.

        Returns:
            The reserve in force after applying ``prompt_tokens``.
        """
        if self._base_synthesis_reserve_s <= 0.0:
            return self.synthesis_reserve_s
        scaled = scaled_synthesis_reserve_s(
            prompt_tokens,
            base_reserve_s=self._base_synthesis_reserve_s,
        )
        self.synthesis_reserve_s = max(self._base_synthesis_reserve_s, scaled)
        return self.synthesis_reserve_s

    def synthesis_timeout_s(
        self,
        now: float | None = None,
        *,
        prompt_tokens: int | None = None,
    ) -> float:
        """Derive the synthesis LLM timeout from remaining wall-clock.

        Remaining request time minus a safety margin, capped by the configured
        typical-prompt maximum. When ``prompt_tokens`` (or a previously applied
        prompt reserve) exceeds a typical prompt, the cap is raised so bulky
        evidence can use leftover request time instead of timing out at 20s.

        ``0`` means skip the LLM and use evidence fallback.

        Args:
            now: Optional monotonic timestamp.
            prompt_tokens: Optional estimated synthesis prompt tokens.

        Returns:
            Seconds to wait for synthesis, or ``0`` when no room remains.
        """
        usable = self.remaining_s(now) - self.safety_margin_s
        if usable <= 0:
            return 0.0
        cap = self.max_synthesis_timeout_s
        if prompt_tokens is not None:
            scaled = scaled_synthesis_reserve_s(
                prompt_tokens,
                base_reserve_s=self.max_synthesis_timeout_s,
            )
            cap = max(cap, scaled)
        elif self.synthesis_reserve_s > self._base_synthesis_reserve_s:
            cap = max(cap, self.synthesis_reserve_s)
        return min(cap, usable)

    def check(
        self,
        ledger: TokenLedger | None,
        correlation_id: str,
        *,
        now: float | None = None,
    ) -> BudgetCeiling | None:
        """Return the ceiling that is already exhausted, if any.

        Args:
            ledger: Request token ledger. Spend is read from ``snapshot``.
            correlation_id: Ledger key.
            now: Optional monotonic timestamp.

        Returns:
            ``deadline``, ``tokens``, or ``cost`` when a ceiling has tripped.
        """
        if self.exhausted is not None:
            return self.exhausted
        if self.remaining_s(now) <= 0:
            self.exhausted = "deadline"
            return self.exhausted
        if ledger is None:
            return None
        snapshot = ledger.snapshot(correlation_id)
        raw_total = snapshot.get("total") or 0
        raw_cost = snapshot.get("cost_usd") or 0.0
        total = int(raw_total) if isinstance(raw_total, (int, float)) else 0
        cost = float(raw_cost) if isinstance(raw_cost, (int, float)) else 0.0
        if self.token_ceiling is not None and total >= self.token_ceiling:
            self.exhausted = "tokens"
            return self.exhausted
        if self.cost_usd_max is not None and cost >= self.cost_usd_max:
            self.exhausted = "cost"
            return self.exhausted
        return None

    def allow_new_call(
        self,
        ledger: TokenLedger | None,
        correlation_id: str,
        *,
        now: float | None = None,
    ) -> bool:
        """True when another specialist or LLM call may be issued.

        Args:
            ledger: Request token ledger.
            correlation_id: Ledger key.
            now: Optional monotonic timestamp.

        Returns:
            ``False`` when a ceiling has already been hit or only the
            synthesis reserve remains.
        """
        if self.check(ledger, correlation_id, now=now) is not None:
            return False
        return self.specialist_remaining_s(now) > 0
