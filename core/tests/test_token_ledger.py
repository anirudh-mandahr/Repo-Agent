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
