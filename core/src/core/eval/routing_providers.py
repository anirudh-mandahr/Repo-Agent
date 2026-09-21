"""Provider factory for the routing bake-off.

``run_router_bakeoff`` treats a model as an opaque id and builds it through an
injectable factory, so a routing backend that is not an OpenRouter chat model
plugs in here rather than in the bake-off itself.
"""

from __future__ import annotations

from typing import Final

from core.eval.model_bakeoff import default_provider_factory
from core.exceptions import ConfigurationError
from core.llm.jev_provider import JEV_MODEL_ID, JevRoutingProvider
from core.llm.provider import LLMProvider
from core.settings import TypeSafeSettings

TYPESAFE_PREFIX: Final = "typesafe/"


def is_typesafe_model(model: str) -> bool:
    """True when ``model`` is served by TypeSafe rather than OpenRouter.

    Args:
        model: Bake-off model id.

    Returns:
        bool.
    """
    return model == JEV_MODEL_ID or model.startswith(TYPESAFE_PREFIX)


def routing_provider_factory(
    model: str,
    *,
    settings: TypeSafeSettings | None = None,
) -> LLMProvider:
    """Build the routing provider for ``model``.

    TypeSafe ids go to :class:`JevRoutingProvider`; everything else falls
    through to the bake-off's OpenRouter default.

    Args:
        model: Bake-off model id.
        settings: Optional settings snapshot. Loaded from env/.env when omitted.

    Returns:
        A provider satisfying :class:`~core.llm.provider.LLMProvider`.

    Raises:
        ConfigurationError: When a TypeSafe model is requested without a key.
    """
    if not is_typesafe_model(model):
        return default_provider_factory(model)
    resolved = settings or TypeSafeSettings.from_env()
    if resolved.api_key is None:
        raise ConfigurationError(
            f"{model} requires a TypeSafe key: set TYPESAFE_API_KEY in the "
            "environment or in .env, then retry"
        )
    return JevRoutingProvider(
        api_key=resolved.api_key.get_secret_value(),
        model=model,
        base_url=resolved.base_url,
        threshold=resolved.agent_threshold,
    )


def typesafe_key_present() -> bool:
    """Whether a TypeSafe key is resolvable from the environment or ``.env``.

    Returns:
        bool.
    """
    return TypeSafeSettings.from_env().api_key is not None
