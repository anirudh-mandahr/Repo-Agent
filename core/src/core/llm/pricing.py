"""Per-model token prices used by the ledger, metrics, and model bake-off."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field, field_validator

PRICE_AS_OF = "2026-08-18"
DEFAULT_PROMPT_USD_PER_MILLION = 3.0
DEFAULT_COMPLETION_USD_PER_MILLION = 15.0
DEFAULT_CACHED_PROMPT_USD_PER_MILLION = 0.30


class ModelRates(BaseModel):
    """USD per 1M tokens for one model id."""

    prompt_usd_per_million: float = Field(default=DEFAULT_PROMPT_USD_PER_MILLION)
    completion_usd_per_million: float = Field(default=DEFAULT_COMPLETION_USD_PER_MILLION)
    cached_prompt_usd_per_million: float = Field(default=DEFAULT_CACHED_PROMPT_USD_PER_MILLION)

    @field_validator(
        "prompt_usd_per_million",
        "completion_usd_per_million",
        "cached_prompt_usd_per_million",
    )
    @classmethod
    def _non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("token prices must be non-negative")
        return value


# Bake-off synthesis p95 (ms) from `scripts/eval_models.py` on 2026-08-19.
# OrchestratorSettings requires synthesis_reserve_s - safety_margin >= these values.
SYNTHESIS_P95_MS: dict[str, int] = {
    "anthropic/claude-sonnet-4.5": 19943,
    "anthropic/claude-haiku-4.5": 9068,
    "openai/gpt-4.1-mini": 11714,
}


def measured_synthesis_p95_s(model: str) -> float | None:
    """Return baked-off synthesis p95 in seconds for ``model``, if known.

    Args:
        model: OpenRouter model id.

    Returns:
        p95 seconds, or ``None`` when the model was not in the bake-off.
    """
    raw = SYNTHESIS_P95_MS.get(model)
    if raw is None:
        return None
    return raw / 1000.0


# OpenRouter standard (non-batch) list prices. Overlay JSON wins when set.
CATALOG: dict[str, ModelRates] = {
    "anthropic/claude-sonnet-4.5": ModelRates(
        prompt_usd_per_million=3.0,
        completion_usd_per_million=15.0,
        cached_prompt_usd_per_million=0.30,
    ),
    "anthropic/claude-haiku-4.5": ModelRates(
        prompt_usd_per_million=0.50,
        completion_usd_per_million=2.50,
        cached_prompt_usd_per_million=0.05,
    ),
    "openai/gpt-4.1-mini": ModelRates(
        prompt_usd_per_million=0.40,
        completion_usd_per_million=1.60,
        cached_prompt_usd_per_million=0.10,
    ),
    "google/gemini-2.5-flash": ModelRates(
        prompt_usd_per_million=0.30,
        completion_usd_per_million=2.50,
        cached_prompt_usd_per_million=0.03,
    ),
}


def parse_prices_json(raw: str) -> dict[str, ModelRates]:
    """Parse an optional ``LLM_MODEL_PRICES_JSON`` overlay.

    Args:
        raw: JSON object mapping model id to rate fields.

    Returns:
        Parsed overlay. Empty when ``raw`` is blank.

    Raises:
        ValueError: When the payload is not a JSON object of rate mappings.
    """
    text = raw.strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM_MODEL_PRICES_JSON is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("LLM_MODEL_PRICES_JSON must be a JSON object")
    overlay: dict[str, ModelRates] = {}
    for model, value in payload.items():
        overlay[str(model)] = _rates_from_mapping(value)
    return overlay


def _rates_from_mapping(value: Any) -> ModelRates:
    if isinstance(value, ModelRates):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("each model price entry must be an object")
    data = dict(value)
    if "prompt" in data and "prompt_usd_per_million" not in data:
        data["prompt_usd_per_million"] = data.pop("prompt")
    if "completion" in data and "completion_usd_per_million" not in data:
        data["completion_usd_per_million"] = data.pop("completion")
    if "cached_prompt" in data and "cached_prompt_usd_per_million" not in data:
        data["cached_prompt_usd_per_million"] = data.pop("cached_prompt")
    return ModelRates.model_validate(data)


def resolve_rates(
    model: str,
    *,
    overlay: Mapping[str, ModelRates] | None = None,
    fallback: ModelRates | None = None,
) -> ModelRates:
    """Return overlay, then catalog, then fallback rates for ``model``.

    Args:
        model: OpenRouter model id.
        overlay: Optional env-configured per-model rates.
        fallback: Global default rates when the model is unknown.

    Returns:
        Resolved :class:`ModelRates`.
    """
    if overlay and model in overlay:
        return overlay[model]
    if model in CATALOG:
        return CATALOG[model]
    return fallback or ModelRates()


def cost_usd(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_prompt_tokens: int = 0,
    rates: ModelRates | None = None,
) -> float:
    """USD cost for one call using per-1M input/output/cached rates.

    Cached input tokens are billed at ``cached_prompt_usd_per_million``; the
    remainder of ``prompt_tokens`` is billed at the uncached input rate.

    Args:
        prompt_tokens: Total prompt-side tokens (cached + uncached).
        completion_tokens: Completion-side tokens.
        cached_prompt_tokens: Prompt tokens served from provider cache.
        rates: Per-1M rates. Defaults to Claude Sonnet 4.5 catalog rates.

    Returns:
        Estimated USD.
    """
    resolved = rates or ModelRates()
    cached = max(0, cached_prompt_tokens)
    uncached = max(0, prompt_tokens - cached)
    return (
        (uncached / 1_000_000) * resolved.prompt_usd_per_million
        + (cached / 1_000_000) * resolved.cached_prompt_usd_per_million
        + (completion_tokens / 1_000_000) * resolved.completion_usd_per_million
    )


def price_label(model: str, rates: ModelRates) -> str:
    """Format the price that will be used for ``model``.

    Args:
        model: OpenRouter model id.
        rates: Resolved rates.

    Returns:
        Report line including the catalog date.
    """
    return (
        f"{model}: ${rates.prompt_usd_per_million:.2f} in / "
        f"${rates.completion_usd_per_million:.2f} out / "
        f"${rates.cached_prompt_usd_per_million:.3f} cached per 1M "
        f"(as of {PRICE_AS_OF})"
    )
