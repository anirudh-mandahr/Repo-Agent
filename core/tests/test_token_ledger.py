from __future__ import annotations

import asyncio

from core.llm.provider import TokenUsage
from core.observability.ledger import TokenLedger


def test_token_ledger_sums_two_calls_per_correlation_id() -> None:
    ledger = TokenLedger()
    ledger.open("corr-1")
    ledger.record(
        "corr-1",
        "routing",
        TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )
    ledger.record(
        "corr-1",
        "synthesis",
        TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
    )

    closed = ledger.close("corr-1")

    assert closed["total"] == 180
    assert closed["prompt"] == 150
    assert closed["completion"] == 30
    assert closed["llm_calls"] == 2
    assert closed["by_purpose"]["routing"]["total"] == 120
    assert closed["by_purpose"]["synthesis"]["total"] == 60
    assert "by_model" in closed
    assert closed["cached_prompt"] == 0


def test_token_ledger_does_not_cross_contaminate_concurrent_ids() -> None:
    ledger = TokenLedger()
    ledger.open("corr-a")
    ledger.open("corr-b")

    async def _record(correlation_id: str, purpose: str, total: int) -> None:
        ledger.record(
            correlation_id,
            purpose,  # type: ignore[arg-type]
            TokenUsage(prompt_tokens=total - 1, completion_tokens=1, total_tokens=total),
        )

    async def _main() -> None:
        await asyncio.gather(
            _record("corr-a", "routing", 120),
            _record("corr-b", "synthesis", 60),
        )

    asyncio.run(_main())

    closed_a = ledger.close("corr-a")
    closed_b = ledger.close("corr-b")

    assert closed_a["total"] == 120
    assert closed_b["total"] == 60
    assert "routing" in closed_a["by_purpose"]
    assert "routing" not in closed_b["by_purpose"]


def test_token_ledger_records_model_and_cached_prompt_separately() -> None:
    ledger = TokenLedger()
    ledger.open("corr-cache")
    ledger.record(
        "corr-cache",
        "routing",
        TokenUsage(
            prompt_tokens=100,
            completion_tokens=10,
            total_tokens=110,
            model="anthropic/claude-haiku-4.5",
            cached_prompt_tokens=80,
        ),
    )
    closed = ledger.close("corr-cache")
    assert closed["cached_prompt"] == 80
    assert closed["uncached_prompt"] == 20
    assert closed["by_purpose"]["routing"]["model"] == "anthropic/claude-haiku-4.5"
    assert closed["by_model"]["anthropic/claude-haiku-4.5"]["cached_prompt"] == 80
    assert float(closed["cost_usd"]) > 0
    haiku_cost = float(closed["by_model"]["anthropic/claude-haiku-4.5"]["cost_usd"])
    sonnet_usage = TokenUsage(
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        model="anthropic/claude-sonnet-4.5",
        cached_prompt_tokens=80,
    )
    other = TokenLedger()
    other.open("corr-sonnet")
    other.record("corr-sonnet", "routing", sonnet_usage)
    sonnet_cost = float(other.close("corr-sonnet")["cost_usd"])
    assert haiku_cost < sonnet_cost


def test_token_ledger_records_analysis_purpose_from_payload() -> None:
    from core.observability.ledger import usage_from_payload

    ledger = TokenLedger()
    ledger.open("corr-analysis")
    usage = TokenUsage(
        prompt_tokens=200,
        completion_tokens=40,
        total_tokens=240,
        model="anthropic/claude-sonnet-4.5",
    )
    assert ledger.record_payload(
        "corr-analysis",
        {"qualified_name": "fastapi.FastAPI", "explanation": "ok", "usage": usage},
        purpose="analysis",
    )
    closed = ledger.close("corr-analysis")
    assert closed["total"] == 240
    assert closed["llm_calls"] == 1
    assert closed["by_purpose"]["analysis"]["total"] == 240
    assert closed["by_purpose"]["analysis"]["llm_calls"] == 1
    assert float(closed["cost_usd"]) > 0
    assert usage_from_payload({"explanation": "ok"}) is None
    zero = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    assert usage_from_payload({"usage": zero}) is None
    assert usage_from_payload(object()) is None
