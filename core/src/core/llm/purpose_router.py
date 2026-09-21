"""Composite ``LLMProvider`` that dispatches by purpose to different backends.

``LLMSettings.resolve_model`` and ``OpenRouterProvider.model_for`` only ever
select a *model* within one provider. Serving one purpose (routing) from a
different *provider* than the rest (synthesis, analysis, summarization)
needs one more level of indirection: something that still satisfies
``LLMProvider`` itself, so the rest of the orchestrator does not need to know
that more than one backend is involved.

``is_typesafe_model`` also lives here, not in ``core.eval``, because
``core.llm.factory`` needs it to decide whether to build a
:class:`~core.llm.jev_provider.JevRoutingProvider` for the routing purpose,
and ``core.llm`` must not import from ``core.eval`` (the reverse is fine, and
is exactly how ``core.eval.routing_providers`` reuses this function).
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel

from core.llm.jev_provider import JEV_MODEL_ID
from core.llm.provider import LLMProvider, LLMPurpose, LLMResult, Message

TYPESAFE_PREFIX: Final = "typesafe/"


def is_typesafe_model(model: str) -> bool:
    """True when ``model`` is served by TypeSafe rather than OpenRouter.

    Args:
        model: A resolved model id, e.g. ``typesafe/jev-latest`` or an
            OpenRouter slug such as ``anthropic/claude-sonnet-4.5``.

    Returns:
        bool.
    """
    return model == JEV_MODEL_ID or model.startswith(TYPESAFE_PREFIX)


class PurposeRoutedProvider:
    """Dispatches ``complete()`` to a different provider per ``purpose``.

    Total by construction: any purpose not present in ``overrides`` falls
    through to ``default``, so every ``LLMPurpose`` always resolves to some
    provider.
    """

    def __init__(
        self,
        *,
        default: LLMProvider,
        overrides: dict[LLMPurpose, LLMProvider],
    ) -> None:
        """Create a purpose-dispatching provider.

        Args:
            default: Provider used for any purpose not in ``overrides``.
            overrides: Purpose-specific providers, e.g. ``{"routing": jev}``.
        """
        self._default = default
        self._overrides = dict(overrides)

    def provider_for(self, purpose: LLMPurpose) -> LLMProvider:
        """Return the provider that ``purpose`` would be dispatched to.

        Args:
            purpose: The purpose to resolve.

        Returns:
            The provider mapped in ``overrides``, or ``default`` when absent.
        """
        return self._overrides.get(purpose, self._default)

    async def complete(
        self,
        messages: list[Message],
        response_model: type[BaseModel] | None = None,
        *,
        purpose: LLMPurpose,
        agent: str = "llm",
        max_tokens: int = 1024,
        temperature: float | None = None,
    ) -> LLMResult:
        """Delegate to whichever provider is mapped to ``purpose``.

        Args:
            messages: Chat messages.
            response_model: Optional structured schema.
            purpose: Selects which underlying provider handles the call.
            agent: Calling agent name, passed through unchanged.
            max_tokens: Passed through unchanged.
            temperature: Passed through unchanged.

        Returns:
            The ``LLMResult`` returned by the delegated provider.
        """
        provider = self.provider_for(purpose)
        return await provider.complete(
            messages,
            response_model,
            purpose=purpose,
            agent=agent,
            max_tokens=max_tokens,
            temperature=temperature,
        )
