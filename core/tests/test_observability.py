"""Observability helpers: cache counters, token cost, and trace carriers."""

from __future__ import annotations

from core.llm.provider import TokenUsage
from core.observability.ledger import TokenLedger
from core.observability.metrics import render_metrics
from core.observability.tracing import inject_trace_carrier, start_span


def test_token_ledger_increments_prometheus_counters() -> None:
    ledger = TokenLedger()
    ledger.open("corr-metrics")
    ledger.record(
        "corr-metrics",
        "routing",
        TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )
    ledger.close("corr-metrics")
    text = render_metrics().decode("utf-8")
    assert "repochat_llm_tokens_total" in text
    assert "repochat_llm_cost_usd_total" in text
    assert "repochat_llm_calls_total" in text


def test_budget_and_synthesis_metrics_are_declared() -> None:
    from core.observability.metrics import (
        record_budget_exhausted,
        record_evidence_only,
        record_synthesis_latency,
        record_ttft,
    )

    record_synthesis_latency(0.2)
    record_ttft(0.05)
    record_budget_exhausted("tokens")
    record_evidence_only()
    text = render_metrics().decode("utf-8")
    assert "repochat_synthesis_duration_seconds" in text
    assert "repochat_time_to_first_token_seconds" in text
    assert "repochat_budget_exhausted_total" in text
    assert "repochat_evidence_only_total" in text


def test_inject_trace_carrier_includes_correlation_id() -> None:
    carrier = inject_trace_carrier("corr-trace-1")
    assert carrier["correlation_id"] == "corr-trace-1"
    with start_span("unit", correlation_id="corr-trace-1"):
        nested = inject_trace_carrier("corr-trace-1")
    assert nested["correlation_id"] == "corr-trace-1"
