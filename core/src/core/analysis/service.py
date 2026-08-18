"""Code Analyst service: graph lookup + snippet retrieval + structured LLM analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from core.analysis.lookups import CLASS_CONTEXT, ENTITY_LOCATION, FUNCTION_CONTEXT
from core.analysis.models import (
    ClassAnalysis,
    FunctionAnalysis,
    ImplementationComparison,
    ImplementationExplanation,
    PatternAnalysis,
    PatternInstance,
    SnippetResult,
)
from core.analysis.prompts import (
    ANALYZE_CLASS_PROMPT,
    ANALYZE_CLASS_SYSTEM,
    ANALYZE_FUNCTION_PROMPT,
    ANALYZE_FUNCTION_SYSTEM,
    COMPARE_IMPLEMENTATIONS_PROMPT,
    COMPARE_IMPLEMENTATIONS_SYSTEM,
    EXPLAIN_CLASS_PROMPT,
    EXPLAIN_CLASS_SYSTEM,
    EXPLAIN_IMPLEMENTATION_PROMPT,
    EXPLAIN_IMPLEMENTATION_SYSTEM,
    FIND_PATTERNS_PROMPT,
    FIND_PATTERNS_SYSTEM,
)
from core.analysis.snippets import PathTraversalError, get_snippet, repo_root_from_env
from core.exceptions import SchemaValidationError
from core.llm.provider import LLMProvider, LLMResult, Message
from core.logging import get_logger
from core.querying.patterns import SUPPORTED_PATTERNS, pattern_cypher

log = get_logger(__name__)

MAX_PROMPT_SNIPPET_LINES = 120
MAX_PROMPT_LIST_ITEMS = 40


class GraphLookup(Protocol):
    """Async callable that runs read-only Cypher (Graph Query agent or a test fake)."""

    async def __call__(
        self,
        cypher: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...


class CodeAnalystService:
    """Six analysis tools. LLM calls go through ``provider``; graph via ``graph_lookup``."""

    def __init__(
        self,
        provider: LLMProvider,
        graph_lookup: GraphLookup,
        *,
        repo_root: Path | str | None = None,
        agent: str = "code_analyst",
    ) -> None:
        self._provider = provider
        self._graph_lookup = graph_lookup
        self._repo_root = Path(repo_root) if repo_root is not None else repo_root_from_env()
        self._agent = agent

    async def analyze_function(self, qualified_name: str) -> FunctionAnalysis:
        """Fetch function metadata + snippet, then return a structured LLM analysis."""
        context = await self._function_context(qualified_name)
        if context is None:
            return FunctionAnalysis(
                qualified_name=qualified_name,
                error=f"Entity not found: {qualified_name}",
            )
        snippet = self._snippet_from_row(context)
        if snippet.error:
            return FunctionAnalysis(qualified_name=qualified_name, error=snippet.error)
        prompt = render_prompt(
            ANALYZE_FUNCTION_PROMPT,
            {
                "qualified_name": qualified_name,
                "module": _str(context.get("module")),
                "class_name": _str(context.get("class_name")),
                "parameters": _format_parameters(context.get("parameters")),
                "decorators": _bullets(_str_list(context.get("decorators"))),
                "dependents": _bullets(_str_list(context.get("dependents"))),
                "snippet": snippet.text,
            },
        )
        result = await self._complete(
            ANALYZE_FUNCTION_SYSTEM,
            prompt,
            FunctionAnalysis,
        )
        return _require_parsed(result, FunctionAnalysis, agent=self._agent)

    async def analyze_class(self, qualified_name: str) -> ClassAnalysis:
        """Fetch class metadata + snippet, then return a structured LLM analysis."""
        rows = await self._graph_lookup(CLASS_CONTEXT, {"qualified_name": qualified_name})
        if not rows:
            return ClassAnalysis(
                qualified_name=qualified_name,
                error=f"Entity not found: {qualified_name}",
            )
        context = rows[0]
        snippet = self._snippet_from_row(context)
        if snippet.error:
            return ClassAnalysis(qualified_name=qualified_name, error=snippet.error)
        bases = _str_list(context.get("bases")) or _str_list(context.get("inherited_from"))
        prompt = render_prompt(
            ANALYZE_CLASS_PROMPT,
            {
                "qualified_name": qualified_name,
                "module": _str(context.get("module")),
                "bases": _bullets(bases),
                "methods": _bullets(_str_list(context.get("methods"))),
                "decorators": _bullets(_str_list(context.get("decorators"))),
                "snippet": snippet.text,
            },
        )
        result = await self._complete(ANALYZE_CLASS_SYSTEM, prompt, ClassAnalysis)
        return _require_parsed(result, ClassAnalysis, agent=self._agent)

    async def find_patterns(self, pattern: str) -> PatternAnalysis:
        """Find decorator / dependency_injection / factory instances, then explain them."""
        cypher = pattern_cypher(pattern)
        if cypher is None:
            supported = list(SUPPORTED_PATTERNS)
            error = (
                f"Unknown pattern {pattern!r}. "
                f"Supported patterns: {', '.join(supported)}"
            )
            log.warning("analysis.unknown_pattern", pattern=pattern, supported=supported)
            return PatternAnalysis(
                pattern=pattern,
                error=error,
                supported_patterns=supported,
            )
        rows = await self._graph_lookup(cypher, {})
        instances = [
            PatternInstance(
                qualified_name=_str(row.get("qualified_name")),
                file_path=_str(row.get("file_path")),
                line_start=_opt_int(row.get("line_start")),
                line_end=_opt_int(row.get("line_end")),
            )
            for row in rows
            if _str(row.get("qualified_name"))
        ]
        if not instances:
            return PatternAnalysis(
                pattern=pattern,
                instances=[],
                supported_patterns=list(SUPPORTED_PATTERNS),
            )
        prompt = render_prompt(
            FIND_PATTERNS_PROMPT,
            {
                "pattern": pattern,
                "instances": _bullets([item.qualified_name for item in instances]),
            },
        )
        result = await self._complete(FIND_PATTERNS_SYSTEM, prompt, PatternAnalysis)
        parsed = _require_parsed(result, PatternAnalysis, agent=self._agent)
        if not parsed.instances:
            parsed = parsed.model_copy(update={"instances": instances, "pattern": pattern})
        if not parsed.supported_patterns:
            parsed = parsed.model_copy(update={"supported_patterns": list(SUPPORTED_PATTERNS)})
        return parsed

    async def get_code_snippet(
        self,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        context: int = 5,
    ) -> SnippetResult:
        """Retrieve numbered source. No LLM call."""
        path = file_path
        start = line_start
        end = line_end
        if path is None or start is None or end is None:
            if not qualified_name:
                return SnippetResult(
                    error="Provide qualified_name or file_path with line_start and line_end"
                )
            rows = await self._graph_lookup(
                ENTITY_LOCATION,
                {"qualified_name": qualified_name},
            )
            if not rows:
                return SnippetResult(error=f"Entity not found: {qualified_name}")
            row = rows[0]
            path = path or _str(row.get("file_path"))
            start = start if start is not None else _opt_int(row.get("line_start"))
            end = end if end is not None else _opt_int(row.get("line_end"))
        if not path or start is None or end is None:
            return SnippetResult(error="Missing file_path or line range")
        return self._read_snippet(path, start, end, context=context)

    async def explain_implementation(self, qualified_name: str) -> ImplementationExplanation:
        """Explain a function, method, or class using graph context and source."""
        context = await self._function_context(qualified_name)
        if context is not None:
            snippet = self._snippet_from_row(context)
            if snippet.error:
                return ImplementationExplanation(
                    qualified_name=qualified_name,
                    error=snippet.error,
                )
            prompt = render_prompt(
                EXPLAIN_IMPLEMENTATION_PROMPT,
                {
                    "qualified_name": qualified_name,
                    "module": _str(context.get("module")),
                    "class_name": _str(context.get("class_name")),
                    "parameters": _format_parameters(context.get("parameters")),
                    "decorators": _bullets(_str_list(context.get("decorators"))),
                    "dependents": _bullets(_limited(_str_list(context.get("dependents")))),
                    "snippet": _clip_text(snippet.text),
                },
            )
            result = await self._complete(
                EXPLAIN_IMPLEMENTATION_SYSTEM,
                prompt,
                ImplementationExplanation,
            )
            return _require_parsed(
                result,
                ImplementationExplanation,
                agent=self._agent,
            )

        rows = await self._graph_lookup(CLASS_CONTEXT, {"qualified_name": qualified_name})
        if not rows:
            return ImplementationExplanation(
                qualified_name=qualified_name,
                error=f"Entity not found: {qualified_name}",
            )
        class_context = rows[0]
        snippet = self._snippet_from_row(class_context)
        if snippet.error:
            return ImplementationExplanation(
                qualified_name=qualified_name,
                error=snippet.error,
            )
        bases = _str_list(class_context.get("bases")) or _str_list(
            class_context.get("inherited_from")
        )
        prompt = render_prompt(
            EXPLAIN_CLASS_PROMPT,
            {
                "qualified_name": qualified_name,
                "module": _str(class_context.get("module")),
                "bases": _bullets(bases),
                "methods": _bullets(_limited(_str_list(class_context.get("methods")))),
                "decorators": _bullets(_str_list(class_context.get("decorators"))),
                "snippet": _clip_text(snippet.text),
            },
        )
        result = await self._complete(
            EXPLAIN_CLASS_SYSTEM,
            prompt,
            ImplementationExplanation,
        )
        return _require_parsed(result, ImplementationExplanation, agent=self._agent)

    async def compare_implementations(self, name_a: str, name_b: str) -> ImplementationComparison:
        """Two snippets + graph context side by side, structured comparison output."""
        context_a = await self._function_context(name_a)
        context_b = await self._function_context(name_b)
        missing: list[str] = []
        if context_a is None:
            missing.append(name_a)
        if context_b is None:
            missing.append(name_b)
        if context_a is None or context_b is None:
            return ImplementationComparison(
                name_a=name_a,
                name_b=name_b,
                error=f"Entity not found: {', '.join(missing)}",
            )
        snippet_a = self._snippet_from_row(context_a)
        snippet_b = self._snippet_from_row(context_b)
        for snippet in (snippet_a, snippet_b):
            if snippet.error:
                return ImplementationComparison(
                    name_a=name_a,
                    name_b=name_b,
                    error=snippet.error,
                )
        prompt = render_prompt(
            COMPARE_IMPLEMENTATIONS_PROMPT,
            {
                "name_a": name_a,
                "module_a": _str(context_a.get("module")),
                "class_a": _str(context_a.get("class_name")),
                "parameters_a": _format_parameters(context_a.get("parameters")),
                "decorators_a": _bullets(_str_list(context_a.get("decorators"))),
                "dependents_a": _bullets(_str_list(context_a.get("dependents"))),
                "snippet_a": snippet_a.text,
                "name_b": name_b,
                "module_b": _str(context_b.get("module")),
                "class_b": _str(context_b.get("class_name")),
                "parameters_b": _format_parameters(context_b.get("parameters")),
                "decorators_b": _bullets(_str_list(context_b.get("decorators"))),
                "dependents_b": _bullets(_str_list(context_b.get("dependents"))),
                "snippet_b": snippet_b.text,
            },
        )
        result = await self._complete(
            COMPARE_IMPLEMENTATIONS_SYSTEM,
            prompt,
            ImplementationComparison,
        )
        return _require_parsed(result, ImplementationComparison, agent=self._agent)

    async def _function_context(self, qualified_name: str) -> dict[str, Any] | None:
        rows = await self._graph_lookup(
            FUNCTION_CONTEXT,
            {"qualified_name": qualified_name},
        )
        return rows[0] if rows else None

    def _snippet_from_row(self, row: Mapping[str, Any], context: int = 5) -> SnippetResult:
        path = _str(row.get("file_path"))
        start = _opt_int(row.get("line_start"))
        end = _opt_int(row.get("line_end"))
        if not path or start is None or end is None:
            return SnippetResult(error="Missing file_path or line range")
        return self._read_snippet(path, start, end, context=context)

    def _read_snippet(
        self,
        file_path: str,
        line_start: int,
        line_end: int,
        context: int = 5,
    ) -> SnippetResult:
        try:
            text = get_snippet(
                file_path,
                line_start,
                line_end,
                context=context,
                repo_root=self._repo_root,
            )
        except PathTraversalError as exc:
            return SnippetResult(
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                error=str(exc),
            )
        except OSError as exc:
            log.warning("snippet.read_failed", file_path=file_path, error=str(exc))
            return SnippetResult(
                file_path=file_path,
                line_start=line_start,
                line_end=line_end,
                error=str(exc),
            )
        return SnippetResult(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            text=text,
        )

    async def _complete(
        self,
        system: str,
        prompt: str,
        response_model: type[BaseModel],
    ) -> LLMResult:
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=prompt),
        ]
        return await self._provider.complete(
            messages,
            response_model,
            purpose="analysis",
            agent=self._agent,
        )


def render_prompt(template: str, mapping: Mapping[str, str]) -> str:
    """Replace ``{name}`` placeholders without interpreting braces in snippet text."""
    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered


def _require_parsed[TModel: BaseModel](
    result: LLMResult,
    model: type[TModel],
    *,
    agent: str,
) -> TModel:
    parsed = result.parsed
    if isinstance(parsed, model):
        return parsed
    if parsed is not None:
        try:
            return model.model_validate(parsed.model_dump())
        except ValidationError as exc:
            raise SchemaValidationError(agent=agent, message=str(exc)) from exc
    raise SchemaValidationError(
        agent=agent,
        message=f"provider returned no {model.__name__} instance",
    )


def _str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    return text if text not in {"None", "null"} else ""


def _opt_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value and value not in {"None", "null"} else []
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        items: list[str] = []
        for item in value:
            text = _str(item)
            if text:
                items.append(text)
        return items
    text = _str(value)
    return [text] if text else []


def _bullets(items: Sequence[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _limited(items: Sequence[str], limit: int = MAX_PROMPT_LIST_ITEMS) -> list[str]:
    values = list(items)
    if len(values) <= limit:
        return values
    extra = len(values) - limit
    return [*values[:limit], f"... ({extra} more omitted)"]


def _clip_text(text: str, max_lines: int = MAX_PROMPT_SNIPPET_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    omitted = len(lines) - max_lines
    return "\n".join([*lines[:max_lines], f"... ({omitted} more lines omitted)"])


def _format_parameters(raw: Any) -> str:
    if raw is None:
        return "(none)"
    if isinstance(raw, str):
        return raw if raw else "(none)"
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray, str)):
        return _str(raw) or "(none)"
    lines: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            name = _str(item.get("name"))
            if not name:
                continue
            annotation = _str(item.get("annotation"))
            default = item.get("default")
            text = name
            if annotation:
                text += f": {annotation}"
            if default is not None and _str(default):
                text += f" = {default}"
            lines.append(text)
        else:
            text = _str(item)
            if text:
                lines.append(text)
    return _bullets(lines)
