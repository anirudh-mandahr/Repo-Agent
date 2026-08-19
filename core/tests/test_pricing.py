"""Per-model price catalog and overlay JSON."""

from __future__ import annotations

import pytest

from core.llm.pricing import (
    PRICE_AS_OF,
    SYNTHESIS_P95_MS,
    ModelRates,
    cost_usd,
    measured_synthesis_p95_s,
    parse_prices_json,
    price_label,
    resolve_rates,
)
from core.settings import LLMSettings


def test_parse_prices_json_blank_and_overlay() -> None:
    assert parse_prices_json("") == {}
    overlay = parse_prices_json(
        '{"demo/model": {"prompt": 1.0, "completion": 2.0, "cached_prompt": 0.1}}'
    )
    assert overlay["demo/model"].prompt_usd_per_million == 1.0
    assert overlay["demo/model"].completion_usd_per_million == 2.0
    assert overlay["demo/model"].cached_prompt_usd_per_million == 0.1


def test_parse_prices_json_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_prices_json("{")
    with pytest.raises(ValueError, match="JSON object"):
        parse_prices_json("[1]")


def test_resolve_rates_prefers_overlay_then_catalog() -> None:
    overlay = {"anthropic/claude-sonnet-4.5": ModelRates(prompt_usd_per_million=9.0)}
    rates = resolve_rates("anthropic/claude-sonnet-4.5", overlay=overlay)
    assert rates.prompt_usd_per_million == 9.0
    catalog = resolve_rates("anthropic/claude-haiku-4.5")
    assert catalog.prompt_usd_per_million == 0.50
    fallback = resolve_rates("unknown/model", fallback=ModelRates(prompt_usd_per_million=4.0))
    assert fallback.prompt_usd_per_million == 4.0


def test_cost_usd_bills_cached_tokens_cheaper() -> None:
    rates = ModelRates(
        prompt_usd_per_million=3.0,
        completion_usd_per_million=15.0,
        cached_prompt_usd_per_million=0.30,
    )
    uncached = cost_usd(prompt_tokens=1_000_000, completion_tokens=0, rates=rates)
    cached = cost_usd(
        prompt_tokens=1_000_000,
        completion_tokens=0,
        cached_prompt_tokens=1_000_000,
        rates=rates,
    )
    assert uncached == 3.0
    assert cached == 0.30


def test_settings_rates_for_uses_overlay(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "LLM_MODEL_PRICES_JSON",
        '{"custom/model": {"prompt": 1.25, "completion": 10.0, "cached_prompt": 0.2}}',
    )
    settings = LLMSettings.from_env()
    rates = settings.rates_for("custom/model")
    assert rates.prompt_usd_per_million == 1.25
    label = price_label("custom/model", rates)
    assert "custom/model" in label
    assert PRICE_AS_OF in label


def test_measured_synthesis_p95_matches_bakeoff_catalog() -> None:
    assert measured_synthesis_p95_s("openai/gpt-4.1-mini") == pytest.approx(11.714)
    assert measured_synthesis_p95_s("anthropic/claude-sonnet-4.5") == pytest.approx(19.943)
    assert measured_synthesis_p95_s("unknown/model") is None
    assert SYNTHESIS_P95_MS["openai/gpt-4.1-mini"] < SYNTHESIS_P95_MS["anthropic/claude-sonnet-4.5"]
