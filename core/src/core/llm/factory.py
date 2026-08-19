"""Choose OpenRouter or the deterministic offline provider from env."""

from __future__ import annotations

from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.provider import LLMProvider
from core.logging import get_logger
from core.settings import LLMSettings


def build_llm_provider() -> LLMProvider:
    """Return OpenRouter when an API key is set, otherwise OfflineProvider.

    Returns:
        A production or offline LLM provider. Tests should inject StubProvider.
    """
    settings = LLMSettings.from_env()
    if settings.api_key:
        return OpenRouterProvider.from_env()
    get_logger("llm").warning("llm.offline_provider", reason="OPENROUTER_API_KEY unset")
    return OfflineProvider()
