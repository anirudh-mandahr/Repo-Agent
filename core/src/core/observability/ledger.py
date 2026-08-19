"""In-process token ledger keyed by correlation_id."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from core.llm.pricing import cost_usd
from core.llm.provider import LLMPurpose, TokenUsage
from core.settings import LLMSettings


def empty_token_totals() -> dict[str, Any]:
    """Return a zeroed token-totals mapping for cache hits and new requests.

    Returns:
        Ledger close() shape with empty purpose and model buckets.
    """
    return {
        "total": 0,
        "prompt": 0,
        "completion": 0,
        "cached_prompt": 0,
        "uncached_prompt": 0,
        "llm_calls": 0,
        "cost_usd": 0.0,
        "by_purpose": {},
        "by_model": {},
    }


def _bucket() -> dict[str, Any]:
    return {
        "prompt": 0,
        "completion": 0,
        "total": 0,
        "cached_prompt": 0,
        "uncached_prompt": 0,
        "cost_usd": 0.0,
        "llm_calls": 0,
        "model": None,
    }


@dataclass
class _LedgerEntry:
    prompt: int = 0
    completion: int = 0
    cached_prompt: int = 0
    uncached_prompt: int = 0
    total: int = 0
    llm_calls: int = 0
    cost_usd: float = 0.0
    by_purpose: dict[str, dict[str, Any]] = field(
        default_factory=lambda: defaultdict(_bucket)
    )
    by_model: dict[str, dict[str, Any]] = field(
        default_factory=lambda: defaultdict(_bucket)
    )


class TokenLedger:
    """Accumulate token usage during one `handle_query` request."""

    def __init__(self) -> None:
        """Create an empty in-process ledger."""
        self._entries: dict[str, _LedgerEntry] = {}

    def open(self, correlation_id: str) -> None:
        """Start accumulating usage for ``correlation_id``.

        Args:
            correlation_id: Request id to key the ledger entry.
        """
        self._entries[correlation_id] = _LedgerEntry()

    def record(self, correlation_id: str, purpose: LLMPurpose, usage: TokenUsage) -> None:
        """Add one LLM call's tokens to the open request.

        Args:
            correlation_id: Request id.
            purpose: Ledger bucket such as routing or synthesis.
            usage: Token counts from the provider, including the resolved model.
        """
        entry = self._entries.setdefault(correlation_id, _LedgerEntry())
        settings = LLMSettings.from_env()
        model = usage.model or settings.resolve_model(purpose)
        cached = usage.cached_prompt_tokens
        uncached = usage.uncached_prompt_tokens
        call_cost = cost_usd(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_prompt_tokens=cached,
            rates=settings.rates_for(model),
        )
        entry.prompt += usage.prompt_tokens
        entry.completion += usage.completion_tokens
        entry.cached_prompt += cached
        entry.uncached_prompt += uncached
        entry.total += usage.total_tokens
        entry.llm_calls += 1
        entry.cost_usd += call_cost
        _add_to_bucket(
            entry.by_purpose[purpose],
            usage=usage,
            cached=cached,
            uncached=uncached,
            call_cost=call_cost,
            model=model,
        )
        _add_to_bucket(
            entry.by_model[model],
            usage=usage,
            cached=cached,
            uncached=uncached,
            call_cost=call_cost,
            model=model,
        )
        from core.observability.metrics import record_llm_usage

        record_llm_usage(
            purpose=purpose,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            cached_prompt_tokens=cached,
            model=model,
        )

    def record_payload(
        self,
        correlation_id: str,
        payload: object,
        *,
        purpose: LLMPurpose = "analysis",
    ) -> bool:
        """Record usage embedded in a specialist MCP payload, if present.

        Args:
            correlation_id: Request id opened earlier.
            payload: Tool result mapping or pydantic model that may carry
                ``usage``.
            purpose: Ledger bucket. Code analyst spend uses ``analysis``.

        Returns:
            ``True`` when a usage object was recorded.
        """
        usage = usage_from_payload(payload)
        if usage is None:
            return False
        self.record(correlation_id, purpose, usage)
        return True

    def snapshot(self, correlation_id: str) -> dict[str, object]:
        """Return current totals without closing the request entry.

        Args:
            correlation_id: Request id opened earlier.

        Returns:
            The same shape as :meth:`close` including ``cost_usd``.
        """
        entry = self._entries.get(correlation_id)
        if entry is None:
            return empty_token_totals()
        return _entry_totals(entry)

    def close(self, correlation_id: str) -> dict[str, object]:
        """Pop the request entry and return aggregated token totals.

        Args:
            correlation_id: Request id opened earlier.

        Returns:
            Totals including per-purpose and per-model buckets.
        """
        entry = self._entries.pop(correlation_id, _LedgerEntry())
        return _entry_totals(entry)


def _entry_totals(entry: _LedgerEntry) -> dict[str, object]:
    return {
        "total": entry.total,
        "prompt": entry.prompt,
        "completion": entry.completion,
        "cached_prompt": entry.cached_prompt,
        "uncached_prompt": entry.uncached_prompt,
        "llm_calls": entry.llm_calls,
        "cost_usd": round(entry.cost_usd, 8),
        "by_purpose": {key: dict(value) for key, value in entry.by_purpose.items()},
        "by_model": {key: dict(value) for key, value in entry.by_model.items()},
    }


def usage_from_payload(payload: object) -> TokenUsage | None:
    """Extract ``TokenUsage`` from a code_analyst MCP payload.

    Args:
        payload: Tool result mapping, pydantic model, or unrelated value.

    Returns:
        Usage when the payload includes a usable ``usage`` object; otherwise
        ``None``.
    """
    raw: object | None
    if isinstance(payload, Mapping):
        raw = payload.get("usage")
    else:
        try:
            raw = payload.usage  # type: ignore[attr-defined]
        except AttributeError:
            return None
    if raw is None:
        return None
    if isinstance(raw, TokenUsage):
        usage = raw
    else:
        try:
            usage = TokenUsage.model_validate(raw)
        except (TypeError, ValueError):
            return None
    if usage.total_tokens <= 0 and usage.prompt_tokens <= 0:
        return None
    return usage


def record_payload_usage(
    ledger: TokenLedger | None,
    correlation_id: str,
    payload: object,
    *,
    purpose: LLMPurpose = "analysis",
) -> bool:
    """Record specialist-payload usage onto ``ledger`` when present.

    Args:
        ledger: Request token ledger, or ``None`` when accounting is disabled.
        correlation_id: Request id.
        payload: Tool result that may embed ``usage``.
        purpose: Ledger bucket. Code analyst spend uses ``analysis``.

    Returns:
        ``True`` when usage was recorded.
    """
    if ledger is None:
        return False
    return ledger.record_payload(correlation_id, payload, purpose=purpose)


def _add_to_bucket(
    bucket: dict[str, Any],
    *,
    usage: TokenUsage,
    cached: int,
    uncached: int,
    call_cost: float,
    model: str,
) -> None:
    bucket["prompt"] += usage.prompt_tokens
    bucket["completion"] += usage.completion_tokens
    bucket["total"] += usage.total_tokens
    bucket["cached_prompt"] += cached
    bucket["uncached_prompt"] += uncached
    bucket["cost_usd"] = round(float(bucket["cost_usd"]) + call_cost, 8)
    bucket["llm_calls"] = int(bucket["llm_calls"]) + 1
    bucket["model"] = model
