"""StubProvider records calls and returns canned responses. No network."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from core.analysis.models import ImplementationExplanation
from core.llm import LLMProvider, LLMResult, Message, SchemaValidationError, StubProvider
from core.llm.offline_provider import OfflineProvider
from core.llm.openrouter_provider import STRUCTURED_TOOL_NAME, OpenRouterProvider
from core.llm.provider import parse_structured


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


class _FakeCompletions:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    async def create(self, **kwargs: object) -> object:
        self.payloads.append(kwargs)
        function = type("Function", (), {"arguments": '{"value": "ok"}'})()
        tool_call = type("ToolCall", (), {"function": function})()
        message = type("Message", (), {"tool_calls": [tool_call], "content": None})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


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
    assert result.usage.total_tokens > 0
