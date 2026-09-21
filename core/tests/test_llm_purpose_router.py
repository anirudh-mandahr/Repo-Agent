"""PurposeRoutedProvider dispatch, plus build_llm_provider's TypeSafe wiring."""

from __future__ import annotations

import pytest

from core.exceptions import ConfigurationError
from core.llm.jev_provider import JEV_MODEL_ID, JevRoutingProvider
from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import OpenRouterProvider
from core.llm.provider import LLMProvider, Message
from core.llm.purpose_router import PurposeRoutedProvider, is_typesafe_model
from core.llm.stub import StubProvider

_MESSAGES = [Message(role="user", content="hi")]


# --- PurposeRoutedProvider ---------------------------------------------------


async def test_routing_goes_to_the_override_and_synthesis_to_the_default() -> None:
    """Each purpose must reach the provider it was mapped to, not the other one."""
    default = StubProvider(["synthesis text"])
    routing = StubProvider(["routing text"])
    composite = PurposeRoutedProvider(default=default, overrides={"routing": routing})

    await composite.complete(_MESSAGES, purpose="routing", agent="orchestrator")
    await composite.complete(_MESSAGES, purpose="synthesis", agent="orchestrator")

    assert len(routing.calls) == 1
    assert routing.calls[0].purpose == "routing"
    assert len(default.calls) == 1
    assert default.calls[0].purpose == "synthesis"


async def test_every_purpose_resolves_to_a_provider() -> None:
    """No override is set for analysis/summarization; both must fall through."""
    default = StubProvider(["a", "b"])
    routing = StubProvider(["routing"])
    composite = PurposeRoutedProvider(default=default, overrides={"routing": routing})

    await composite.complete(_MESSAGES, purpose="analysis", agent="code_analyst")
    await composite.complete(_MESSAGES, purpose="summarization", agent="memory")

    assert len(default.calls) == 2
    assert routing.calls == []


def test_provider_for_reports_the_resolved_backend() -> None:
    default = StubProvider()
    routing = StubProvider()
    composite = PurposeRoutedProvider(default=default, overrides={"routing": routing})

    assert composite.provider_for("routing") is routing
    assert composite.provider_for("synthesis") is default
    assert composite.provider_for("analysis") is default
    assert composite.provider_for("summarization") is default


def test_composite_satisfies_the_llm_provider_protocol() -> None:
    composite = PurposeRoutedProvider(default=StubProvider(), overrides={})
    assert isinstance(composite, LLMProvider)


def test_is_typesafe_model() -> None:
    assert is_typesafe_model(JEV_MODEL_ID)
    assert is_typesafe_model("typesafe/some-other-id")
    assert not is_typesafe_model("anthropic/claude-sonnet-4.5")


# --- build_llm_provider -------------------------------------------------------


def _clear_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "OPENROUTER_API_KEY",
        "ANTHROPIC_API_KEY",
        "ORCH_MODEL_ROUTING",
        "TYPESAFE_API_KEY",
        "JEV_API_KEY",
        "JEV-API-KEY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_default_config_is_byte_for_byte_todays_behaviour_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ORCH_MODEL_ROUTING, no OpenRouter key: OfflineProvider, unchanged."""
    from core.llm.factory import build_llm_provider

    _clear_llm_env(monkeypatch)
    provider = build_llm_provider()
    assert isinstance(provider, OfflineProvider)
    assert not isinstance(provider, PurposeRoutedProvider)


def test_default_config_is_byte_for_byte_todays_behaviour_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ORCH_MODEL_ROUTING, an OpenRouter key set: plain OpenRouterProvider."""
    from core.llm.factory import build_llm_provider

    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    provider = build_llm_provider()
    assert isinstance(provider, OpenRouterProvider)
    assert not isinstance(provider, PurposeRoutedProvider)


def test_typesafe_routing_model_without_key_raises_configuration_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.llm.factory import build_llm_provider

    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("ORCH_MODEL_ROUTING", JEV_MODEL_ID)
    with pytest.raises(ConfigurationError, match="TYPESAFE_API_KEY"):
        build_llm_provider()


def test_typesafe_routing_model_with_key_builds_composite_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Routing is TypeSafe; synthesis/analysis/summarization stay on OpenRouter."""
    from core.llm.factory import build_llm_provider

    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("ORCH_MODEL_ROUTING", JEV_MODEL_ID)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")

    provider = build_llm_provider()

    assert isinstance(provider, PurposeRoutedProvider)
    routing_provider = provider.provider_for("routing")
    assert isinstance(routing_provider, JevRoutingProvider)
    for purpose in ("synthesis", "analysis", "summarization"):
        assert isinstance(provider.provider_for(purpose), OpenRouterProvider)


def test_typesafe_routing_model_with_key_and_no_openrouter_key_uses_offline_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The non-routing default still follows the existing key-present rule."""
    from core.llm.factory import build_llm_provider

    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ORCH_MODEL_ROUTING", JEV_MODEL_ID)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")

    provider = build_llm_provider()

    assert isinstance(provider, PurposeRoutedProvider)
    assert isinstance(provider.provider_for("routing"), JevRoutingProvider)
    assert isinstance(provider.provider_for("synthesis"), OfflineProvider)


def test_typesafe_key_alias_jev_api_key_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    from core.llm.factory import build_llm_provider

    _clear_llm_env(monkeypatch)
    monkeypatch.setenv("ORCH_MODEL_ROUTING", JEV_MODEL_ID)
    monkeypatch.setenv("JEV_API_KEY", "ts-test")

    provider = build_llm_provider()
    assert isinstance(provider, PurposeRoutedProvider)
    routing_provider = provider.provider_for("routing")
    assert isinstance(routing_provider, JevRoutingProvider)
