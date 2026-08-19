"""StubProvider records calls and returns canned responses. No network."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from core.analysis.models import ImplementationExplanation
from core.llm import LLMProvider, LLMResult, Message, SchemaValidationError, StubProvider
from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import STRUCTURED_TOOL_NAME, OpenRouterProvider
from core.llm.provider import parse_structured
from core.settings import DEFAULT_OPENROUTER_MODEL, DEFAULT_SYNTHESIS_MODEL, LLMSettings


class _Box(BaseModel):
    value: str


async def test_stub_provider_returns_canned_text() -> None:
    provider: LLMProvider = StubProvider(["hello"])
    result = await provider.complete([Message(role="user", content="hi")], purpose="analysis")
    assert isinstance(result, LLMResult)
    assert result.text == "hello"
    assert result.parsed is None
    assert result.usage.total_tokens == 120


async def test_stub_provider_records_messages_and_response_model() -> None:
    provider = StubProvider()
    provider.enqueue(_Box(value="ok").model_dump())
    messages = [Message(role="user", content="structured")]
    result = await provider.complete(messages, _Box, purpose="routing")
    assert isinstance(result.parsed, _Box)
    assert result.parsed.value == "ok"
    assert len(provider.calls) == 1
    assert provider.calls[0].messages == messages
    assert provider.calls[0].response_model is _Box
    assert provider.calls[0].purpose == "routing"


async def test_stub_provider_retries_once_then_raises() -> None:
    provider = StubProvider(["not-json", "{"])
    with pytest.raises(SchemaValidationError):
        await provider.complete([Message(role="user", content="go")], _Box, purpose="routing")
    assert len(provider.calls) == 2


async def test_stub_provider_marks_first_attempt_invalid_after_retry() -> None:
    provider = StubProvider(["not-json", _Box(value="ok").model_dump()])
    result = await provider.complete(
        [Message(role="user", content="go")], _Box, purpose="routing"
    )
    assert isinstance(result.parsed, _Box)
    assert result.first_attempt_valid is False
    assert result.schema_attempts == 2


def test_parse_structured_accepts_json_and_mapping() -> None:
    parsed = parse_structured('{"value": "x"}', _Box, agent="test")
    assert parsed.value == "x"
    parsed_map = parse_structured({"value": "y"}, _Box, agent="test")
    assert parsed_map.value == "y"


def test_parse_structured_raises_on_invalid_json() -> None:
    with pytest.raises(SchemaValidationError, match="invalid JSON"):
        parse_structured("not-json", _Box, agent="test")


def test_openrouter_provider_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "test/provider-model")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.test/api/v1")
    provider = OpenRouterProvider.from_env()
    assert provider._api_key == "openrouter-test-key"
    assert provider._model == "test/provider-model"
    assert provider._base_url == "https://openrouter.test/api/v1"


def test_purpose_models_fall_back_to_openrouter_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    monkeypatch.delenv("ORCH_MODEL_ROUTING", raising=False)
    monkeypatch.delenv("ORCH_MODEL_SYNTHESIS", raising=False)
    monkeypatch.delenv("CA_MODEL_ANALYSIS", raising=False)
    settings = LLMSettings.from_env()
    assert settings.model == DEFAULT_OPENROUTER_MODEL
    assert settings.routing_model is None
    assert settings.resolve_model("routing") == DEFAULT_OPENROUTER_MODEL
    assert settings.resolve_model("synthesis") == DEFAULT_SYNTHESIS_MODEL
    assert settings.resolve_model("analysis") == DEFAULT_OPENROUTER_MODEL


def test_purpose_models_override_global(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    monkeypatch.setenv("ORCH_MODEL_ROUTING", "anthropic/claude-haiku-4.5")
    monkeypatch.setenv("ORCH_MODEL_SYNTHESIS", "openai/gpt-4.1-mini")
    monkeypatch.setenv("CA_MODEL_ANALYSIS", "google/gemini-2.5-flash")
    settings = LLMSettings.from_env()
    assert settings.model == DEFAULT_OPENROUTER_MODEL
    assert settings.resolve_model("routing") == "anthropic/claude-haiku-4.5"
    assert settings.resolve_model("synthesis") == "openai/gpt-4.1-mini"
    assert settings.resolve_model("analysis") == "google/gemini-2.5-flash"
    assert settings.resolve_model("summarization") == DEFAULT_OPENROUTER_MODEL


def test_blank_purpose_model_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    monkeypatch.setenv("ORCH_MODEL_ROUTING", "  ")
    settings = LLMSettings.from_env()
    assert settings.routing_model is None
    assert settings.resolve_model("routing") == DEFAULT_OPENROUTER_MODEL


class _FakeCompletions:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.payloads.append(kwargs)
        function = type("Function", (), {"arguments": '{"value": "ok"}'})()
        tool_call = type("ToolCall", (), {"function": function})()
        message = type("Message", (), {"tool_calls": [tool_call], "content": None})()
        choice = type("Choice", (), {"message": message})()
        details = type("Details", (), {"cached_tokens": 40})()
        usage = type(
            "Usage",
            (),
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": details,
            },
        )()
        return type("Response", (), {"choices": [choice], "usage": usage})()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


async def test_offline_provider_explains_without_network() -> None:
    provider: LLMProvider = OfflineProvider()
    result = await provider.complete(
        [
            Message(role="system", content="explain"),
            Message(
                role="user",
                content=(
                    "Explain this class implementation.\n"
                    "Qualified name: fastapi.applications.FastAPI\n"
                ),
            ),
        ],
        ImplementationExplanation,
        purpose="analysis",
    )
    assert isinstance(result.parsed, ImplementationExplanation)
    assert result.parsed.qualified_name == "fastapi.applications.FastAPI"
    assert result.parsed.explanation
    assert result.parsed.error is None


async def test_openrouter_provider_structured_tool_call_without_network() -> None:
    client = _FakeClient()
    provider = OpenRouterProvider(
        client=client, model="test/provider-model"  # type: ignore[arg-type]
    )
    result = await provider.complete([Message(role="user", content="go")], _Box, purpose="routing")
    assert isinstance(result.parsed, _Box)
    assert result.parsed.value == "ok"
    payload = client.chat.completions.payloads[0]
    assert payload["model"] == "test/provider-model"
    assert payload["tool_choice"] == {
        "type": "function",
        "function": {"name": STRUCTURED_TOOL_NAME},
    }
    assert result.usage.total_tokens == 120
    assert result.usage.model == "test/provider-model"
    assert result.usage.cached_prompt_tokens == 40
    assert result.usage.uncached_prompt_tokens == 60


async def test_openrouter_selects_model_by_purpose(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.5")
    monkeypatch.setenv("ORCH_MODEL_ROUTING", "anthropic/route-model")
    monkeypatch.setenv("ORCH_MODEL_SYNTHESIS", "openai/synth-model")
    monkeypatch.setenv("CA_MODEL_ANALYSIS", "google/analysis-model")
    client = _FakeClient()
    provider = OpenRouterProvider(client=client)  # type: ignore[arg-type]
    await provider.complete(
        [Message(role="system", content="static"), Message(role="user", content="go")],
        _Box,
        purpose="routing",
    )
    await provider.complete([Message(role="user", content="go")], purpose="synthesis")
    await provider.complete([Message(role="user", content="go")], purpose="analysis")
    models = [payload["model"] for payload in client.chat.completions.payloads]
    assert models == ["anthropic/route-model", "openai/synth-model", "google/analysis-model"]
    routing_messages = client.chat.completions.payloads[0]["messages"]
    assert isinstance(routing_messages, list)
    system = routing_messages[0]
    assert isinstance(system, dict)
    content = system["content"]
    assert isinstance(content, list)
    assert content[0]["cache_control"] == {"type": "ephemeral"}


async def test_openrouter_stream_yields_deltas() -> None:
    class _StreamChunk:
        def __init__(self, text: str | None, usage: object | None = None) -> None:
            delta = type("Delta", (), {"content": text})()
            choice = type("Choice", (), {"delta": delta})()
            self.choices = [choice]
            self.usage = usage

    class _Stream:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        async def create(self, **kwargs: object) -> object:
            self.payloads.append(kwargs)

            async def _gen() -> object:
                yield _StreamChunk("Hel")
                yield _StreamChunk("lo")
                usage = type(
                    "Usage",
                    (),
                    {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
                )()
                yield _StreamChunk(None, usage)

            return _gen()

    class _Client:
        def __init__(self) -> None:
            self.chat = type("Chat", (), {"completions": _Stream()})()

    client = _Client()
    provider = OpenRouterProvider(client=client, model="anthropic/claude-sonnet-4.5")  # type: ignore[arg-type]
    parts: list[str] = []
    usage = None
    async for delta, chunk_usage in provider.stream(
        [Message(role="user", content="hi")],
        purpose="synthesis",
    ):
        if delta:
            parts.append(delta)
        if chunk_usage is not None:
            usage = chunk_usage
    assert "".join(parts) == "Hello"
    assert usage is not None
    assert usage.total_tokens == 5
    assert client.chat.completions.payloads[0]["stream"] is True


async def test_openrouter_skips_cache_control_for_non_anthropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini")
    client = _FakeClient()
    provider = OpenRouterProvider(client=client)  # type: ignore[arg-type]
    await provider.complete(
        [Message(role="system", content="static"), Message(role="user", content="go")],
        purpose="synthesis",
    )
    messages = client.chat.completions.payloads[0]["messages"]
    assert isinstance(messages, list)
    system = messages[0]
    assert isinstance(system, dict)
    assert system["content"] == "static"


async def test_openrouter_respects_temperature() -> None:
    client = _FakeClient()
    provider = OpenRouterProvider(
        client=client, model="anthropic/claude-sonnet-4.5", temperature=0.0  # type: ignore[arg-type]
    )
    await provider.complete([Message(role="user", content="go")], purpose="routing")
    assert client.chat.completions.payloads[0]["temperature"] == 0.0
