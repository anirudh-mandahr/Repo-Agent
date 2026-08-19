"""Prometheus counters and histograms shared by the gateway and MCP agents."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from core.llm.pricing import cost_usd
from core.settings import LLMSettings

REQUESTS = Counter(
    "repochat_requests_total",
    "Request count by agent, tool or route, and status.",
    ["agent", "tool", "status"],
)
LATENCY = Histogram(
    "repochat_request_duration_seconds",
    "Request latency in seconds.",
    ["agent", "tool"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)
SYNTHESIS_LATENCY = Histogram(
    "repochat_synthesis_duration_seconds",
    "Synthesis LLM call latency in seconds.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0),
)
TTFT = Histogram(
    "repochat_time_to_first_token_seconds",
    "Time from synthesis start to first streamed token.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)
BUDGET_EXHAUSTED = Counter(
    "repochat_budget_exhausted_total",
    "Request budget ceilings that stopped further specialist calls.",
    ["ceiling"],
)
EVIDENCE_ONLY = Counter(
    "repochat_evidence_only_total",
    "Synthesis turns that fell back to retrieved evidence.",
)
ERRORS = Counter(
    "repochat_errors_total",
    "Error count by agent.",
    ["agent"],
)
CACHE_LOOKUPS = Counter(
    "repochat_cache_lookups_total",
    "Response-cache lookups by hit or miss.",
    ["result"],
)
LLM_TOKENS = Counter(
    "repochat_llm_tokens_total",
    "LLM token counts by purpose, direction, and model.",
    ["purpose", "direction", "model"],
)
LLM_COST = Counter(
    "repochat_llm_cost_usd_total",
    "Estimated LLM cost in USD sourced from the token ledger.",
    ["model"],
)
LLM_CALLS = Counter(
    "repochat_llm_calls_total",
    "LLM completion calls by purpose.",
    ["purpose", "model"],
)

METRICS_CONTENT_TYPE = CONTENT_TYPE_LATEST

PROMPT_USD_PER_MILLION = 3.0
COMPLETION_USD_PER_MILLION = 15.0


def estimate_cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    model: str | None = None,
) -> float:
    """Return a USD cost for one LLM call.

    Args:
        prompt_tokens: Prompt-side tokens (cached + uncached).
        completion_tokens: Completion-side tokens.
        cached_prompt_tokens: Prompt tokens served from the provider cache.
        model: Optional model id used to select per-1M rates.

    Returns:
        Estimated USD using configured per-model rates, falling back to the
        global Claude Sonnet 4.5 defaults.
    """
    settings = LLMSettings.from_env()
    resolved_model = model or settings.model
    return cost_usd(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        rates=settings.rates_for(resolved_model),
    )


@contextmanager
def observe_request(agent: str, tool: str) -> Iterator[None]:
    """Count one request and record its latency and error status.

    Args:
        agent: Agent or ``gateway`` label.
        tool: MCP tool name or HTTP path.

    Yields:
        Control to the wrapped call.
    """
    started = perf_counter()
    status = "ok"
    try:
        yield
    except Exception:
        status = "error"
        ERRORS.labels(agent=agent).inc()
        raise
    finally:
        REQUESTS.labels(agent=agent, tool=tool, status=status).inc()
        LATENCY.labels(agent=agent, tool=tool).observe(perf_counter() - started)


def record_http_result(agent: str, tool: str, *, status_code: int, duration_s: float) -> None:
    """Record one finished HTTP/ASGI request.

    Args:
        agent: Agent or ``gateway`` label.
        tool: Route path.
        status_code: HTTP status.
        duration_s: Wall time in seconds.
    """
    status = "ok" if status_code < 400 else "error"
    if status_code >= 400:
        ERRORS.labels(agent=agent).inc()
    REQUESTS.labels(agent=agent, tool=tool, status=status).inc()
    LATENCY.labels(agent=agent, tool=tool).observe(duration_s)


def record_cache_lookup(*, hit: bool) -> None:
    """Increment the cache hit/miss counter.

    Args:
        hit: Whether the lookup returned a cached response.
    """
    CACHE_LOOKUPS.labels(result="hit" if hit else "miss").inc()


def record_synthesis_latency(duration_s: float) -> None:
    """Record one synthesis call's wall time.

    Args:
        duration_s: Seconds spent in the synthesis LLM call (or fallback).
    """
    SYNTHESIS_LATENCY.observe(max(0.0, duration_s))


def record_ttft(duration_s: float) -> None:
    """Record time-to-first-token for one synthesis call.

    Args:
        duration_s: Seconds from synthesis start to the first streamed token.
    """
    TTFT.observe(max(0.0, duration_s))


def record_budget_exhausted(ceiling: str) -> None:
    """Increment the budget-exhausted counter for one ceiling.

    Args:
        ceiling: ``deadline``, ``tokens``, or ``cost``.
    """
    BUDGET_EXHAUSTED.labels(ceiling=ceiling).inc()


def record_evidence_only() -> None:
    """Increment the evidence-only synthesis counter."""
    EVIDENCE_ONLY.inc()


def record_llm_usage(
    *,
    purpose: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    model: str | None = None,
) -> None:
    """Increment token, call, and cost counters from one ledger record.

    Args:
        purpose: Ledger bucket such as ``routing`` or ``synthesis``.
        prompt_tokens: Prompt-side tokens.
        completion_tokens: Completion-side tokens.
        cached_prompt_tokens: Prompt tokens served from the provider cache.
        model: Resolved model id for this call.
    """
    model_label = model or "unknown"
    uncached = max(0, prompt_tokens - cached_prompt_tokens)
    LLM_TOKENS.labels(purpose=purpose, direction="prompt", model=model_label).inc(
        prompt_tokens
    )
    LLM_TOKENS.labels(purpose=purpose, direction="uncached_prompt", model=model_label).inc(
        uncached
    )
    LLM_TOKENS.labels(purpose=purpose, direction="cached_prompt", model=model_label).inc(
        cached_prompt_tokens
    )
    LLM_TOKENS.labels(purpose=purpose, direction="completion", model=model_label).inc(
        completion_tokens
    )
    LLM_CALLS.labels(purpose=purpose, model=model_label).inc()
    LLM_COST.labels(model=model_label).inc(
        estimate_cost_usd(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_prompt_tokens=cached_prompt_tokens,
            model=model_label if model else None,
        )
    )


def render_metrics() -> bytes:
    """Render the default Prometheus text exposition.

    Returns:
        Encoded ``text/plain`` payload.
    """
    return generate_latest()
