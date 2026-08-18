"""CodeAnalystService with StubProvider. No network."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from core.analysis.lookups import CLASS_CONTEXT, ENTITY_LOCATION, FUNCTION_CONTEXT
from core.analysis.models import (
    ClassAnalysis,
    FunctionAnalysis,
    ImplementationComparison,
    ImplementationExplanation,
    PatternAnalysis,
    PatternInstance,
)
from core.analysis.prompts import ANALYZE_FUNCTION_PROMPT, COMPARE_IMPLEMENTATIONS_PROMPT
from core.analysis.service import CodeAnalystService
from core.analysis.snippets import PathTraversalError, get_snippet
from core.exceptions import SchemaValidationError
from core.llm.stub import StubProvider
from core.querying.patterns import PATTERN_TEMPLATES, SUPPORTED_PATTERNS

FIXTURES = Path(__file__).parent / "fixtures"

HELPER_CONTEXT: dict[str, Any] = {
    "qualified_name": "sample_module.helper",
    "name": "helper",
    "file_path": "sample_module.py",
    "line_start": 13,
    "line_end": 14,
    "parameters": [{"name": "value", "annotation": "int", "default": None, "position": 0}],
    "decorators": [],
    "module": "sample_module",
    "class_name": None,
    "dependents": ["sample_module.ping", "sample_module.Worker.run"],
}

PING_CONTEXT: dict[str, Any] = {
    "qualified_name": "sample_module.ping",
    "name": "ping",
    "file_path": "sample_module.py",
    "line_start": 36,
    "line_end": 38,
    "parameters": [],
    "decorators": ["app.get"],
    "module": "sample_module",
    "class_name": None,
    "dependents": [],
}

WORKER_CONTEXT: dict[str, Any] = {
    "qualified_name": "sample_module.Worker",
    "name": "Worker",
    "file_path": "sample_module.py",
    "line_start": 17,
    "line_end": 27,
    "bases": ["Base"],
    "methods": ["sample_module.Worker.run", "sample_module.Worker.label"],
    "inherited_from": ["sample_module.Base"],
    "decorators": [],
    "module": "sample_module",
}


class FakeLookup:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.by_name: dict[str, list[dict[str, Any]]] = {
            "sample_module.helper": [HELPER_CONTEXT],
            "sample_module.ping": [PING_CONTEXT],
            "sample_module.Worker": [WORKER_CONTEXT],
        }
        self.pattern_rows: list[dict[str, Any]] = []

    async def __call__(
        self,
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        mapping = dict(params or {})
        self.calls.append((cypher, mapping))
        name = mapping.get("qualified_name")
        query = cypher.strip()
        if query == FUNCTION_CONTEXT.strip():
            if name in {"sample_module.helper", "sample_module.ping"}:
                return list(self.by_name[str(name)])
            return []
        if query == CLASS_CONTEXT.strip():
            if name == "sample_module.Worker":
                return list(self.by_name[str(name)])
            return []
        if query == ENTITY_LOCATION.strip() and isinstance(name, str) and name in self.by_name:
            return list(self.by_name[name])
        for template in PATTERN_TEMPLATES.values():
            if query == template.strip():
                return list(self.pattern_rows)
        return []


def _service(provider: StubProvider, lookup: FakeLookup | None = None) -> CodeAnalystService:
    return CodeAnalystService(provider, lookup or FakeLookup(), repo_root=FIXTURES)


def test_analyze_function_prompt_template_exposes_snippet_and_dependents() -> None:
    assert "{snippet}" in ANALYZE_FUNCTION_PROMPT
    assert "{dependents}" in ANALYZE_FUNCTION_PROMPT
    assert "{snippet_a}" in COMPARE_IMPLEMENTATIONS_PROMPT


async def test_analyze_function_prompt_contains_snippet_and_dependents() -> None:
    provider = StubProvider()
    provider.enqueue(
        FunctionAnalysis(
            qualified_name="sample_module.helper",
            summary="Identity helper.",
            purpose="Return the given integer.",
            parameters=["value: int"],
            dependents=["sample_module.ping", "sample_module.Worker.run"],
        ).model_dump()
    )
    result = await _service(provider).analyze_function("sample_module.helper")
    assert result.error is None
    assert result.summary == "Identity helper."
    assert len(provider.calls) == 1
    prompt = "\n".join(message.content for message in provider.calls[0].messages)
    assert "def helper" in prompt
    assert "return value" in prompt
    assert "sample_module.ping" in prompt
    assert "sample_module.Worker.run" in prompt
    assert "Dependents:" in prompt


async def test_compare_implementations_retries_once_then_raises() -> None:
    provider = StubProvider(["not-json", "{"])
    with pytest.raises(SchemaValidationError):
        await _service(provider).compare_implementations(
            "sample_module.helper",
            "sample_module.ping",
        )
    assert len(provider.calls) == 2
    assert provider.calls[0].response_model is ImplementationComparison
    assert provider.calls[1].response_model is ImplementationComparison


async def test_get_code_snippet_makes_no_llm_call_and_rejects_traversal() -> None:
    provider = StubProvider()
    lookup = FakeLookup()
    service = _service(provider, lookup)

    snippet = await service.get_code_snippet(
        file_path="sample_module.py",
        line_start=13,
        line_end=14,
    )
    assert snippet.error is None
    assert "def helper" in snippet.text
    assert provider.calls == []
    assert lookup.calls == []

    traversed = await service.get_code_snippet(
        file_path="../secret.py",
        line_start=1,
        line_end=1,
    )
    assert traversed.error is not None
    assert "outside" in traversed.error
    assert provider.calls == []

    with pytest.raises(PathTraversalError):
        get_snippet("../secret.py", 1, 1, repo_root=FIXTURES)


async def test_find_patterns_unknown_returns_structured_error_without_llm() -> None:
    provider = StubProvider()
    lookup = FakeLookup()
    result = await _service(provider, lookup).find_patterns("singleton")
    assert result.error is not None
    assert "singleton" in result.error
    for name in SUPPORTED_PATTERNS:
        assert name in result.error
    assert result.supported_patterns == list(SUPPORTED_PATTERNS)
    assert provider.calls == []
    assert lookup.calls == []


async def test_analyze_class_and_explain_implementation() -> None:
    provider = StubProvider()
    provider.enqueue(
        ClassAnalysis(
            qualified_name="sample_module.Worker",
            summary="Worker class",
            purpose="Does work",
            methods=["sample_module.Worker.run"],
            bases=["Base"],
        ).model_dump(),
        ImplementationExplanation(
            qualified_name="sample_module.helper",
            explanation="Returns the input unchanged.",
        ).model_dump(),
    )
    service = _service(provider)
    class_result = await service.analyze_class("sample_module.Worker")
    assert class_result.summary == "Worker class"
    class_prompt = "\n".join(message.content for message in provider.calls[0].messages)
    assert "class Worker" in class_prompt
    assert "sample_module.Worker.run" in class_prompt

    explained = await service.explain_implementation("sample_module.helper")
    assert "unchanged" in explained.explanation
    explain_prompt = "\n".join(message.content for message in provider.calls[1].messages)
    assert "def helper" in explain_prompt


async def test_explain_implementation_accepts_class() -> None:
    provider = StubProvider()
    provider.enqueue(
        ImplementationExplanation(
            qualified_name="sample_module.Worker",
            explanation="Worker coordinates helper calls.",
        ).model_dump()
    )
    explained = await _service(provider).explain_implementation("sample_module.Worker")
    assert explained.error is None
    assert "Worker" in explained.explanation
    prompt = "\n".join(message.content for message in provider.calls[0].messages)
    assert "class Worker" in prompt
    assert "sample_module.Worker.run" in prompt


async def test_find_patterns_uses_cypher_template_then_llm() -> None:
    provider = StubProvider()
    lookup = FakeLookup()
    lookup.pattern_rows = [
        {
            "qualified_name": "sample_module.ping",
            "file_path": "sample_module.py",
            "line_start": 36,
            "line_end": 38,
        }
    ]
    provider.enqueue(
        PatternAnalysis(
            pattern="decorator",
            instances=[
                PatternInstance(
                    qualified_name="sample_module.ping",
                    explanation="FastAPI route decorator",
                )
            ],
            supported_patterns=list(SUPPORTED_PATTERNS),
        ).model_dump()
    )
    result = await _service(provider, lookup).find_patterns("decorator")
    assert result.instances[0].qualified_name == "sample_module.ping"
    assert lookup.calls[0][0] == PATTERN_TEMPLATES["decorator"]
    prompt = "\n".join(message.content for message in provider.calls[0].messages)
    assert "sample_module.ping" in prompt


async def test_get_code_snippet_by_qualified_name() -> None:
    provider = StubProvider()
    snippet = await _service(provider).get_code_snippet(qualified_name="sample_module.helper")
    assert snippet.error is None
    assert "def helper" in snippet.text
    assert provider.calls == []
