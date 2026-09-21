"""Choose OpenRouter, offline, or a TypeSafe-routed composite provider from env."""

from __future__ import annotations

from core.exceptions import ConfigurationError
from core.llm.jev_provider import JevRoutingProvider
from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.provider import LLMProvider
from core.llm.purpose_router import PurposeRoutedProvider, is_typesafe_model
from core.logging import get_logger
from core.settings import LLMSettings, TypeSafeSettings


def build_llm_provider() -> LLMProvider:
    """Return the production LLM provider selected by env configuration.

    ``ORCH_MODEL_ROUTING`` ordinarily just picks a model *within* whichever
    provider this function would already return -- it cannot select a
    different provider. When it names a TypeSafe id (``typesafe/jev-latest``
    or any ``typesafe/`` id) and a TypeSafe key is configured, this builds a
    :class:`~core.llm.purpose_router.PurposeRoutedProvider` that sends the
    ``routing`` purpose to :class:`~core.llm.jev_provider.JevRoutingProvider`
    while every other purpose (synthesis, analysis, summarization) keeps
    using the provider that would otherwise have been returned.

    For any deployment that leaves ``ORCH_MODEL_ROUTING`` unset or pointed at
    an OpenRouter model, this is a no-op: behaviour is exactly OpenRouter when
    ``OPENROUTER_API_KEY`` is set, else :class:`OfflineProvider`, as before.

    Returns:
        A production or offline LLM provider. Tests should inject StubProvider.

    Raises:
        ConfigurationError: When ``ORCH_MODEL_ROUTING`` names a TypeSafe model
            but no TypeSafe API key is configured. This fails loudly rather
            than silently falling back to the default provider, since a
            silent fallback would make an eval claim about Jev untrue in
            production.
    """
    llm_settings = LLMSettings.from_env()
    default_provider: LLMProvider
    if llm_settings.api_key:
        default_provider = OpenRouterProvider.from_env()
    else:
        get_logger("llm").warning("llm.offline_provider", reason="OPENROUTER_API_KEY unset")
        default_provider = OfflineProvider()

    routing_model = llm_settings.resolve_model("routing")
    if not is_typesafe_model(routing_model):
        return default_provider

    typesafe_settings = TypeSafeSettings.from_env()
    if typesafe_settings.api_key is None:
        raise ConfigurationError(
            f"ORCH_MODEL_ROUTING={routing_model} requires a TypeSafe key: set "
            "TYPESAFE_API_KEY in the environment, then retry"
        )
    routing_provider: LLMProvider = JevRoutingProvider(
        api_key=typesafe_settings.api_key.get_secret_value(),
        model=routing_model,
        base_url=typesafe_settings.base_url,
        threshold=typesafe_settings.agent_threshold,
    )
    return PurposeRoutedProvider(
        default=default_provider,
        overrides={"routing": routing_provider},
    )
