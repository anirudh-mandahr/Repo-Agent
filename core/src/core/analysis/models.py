"""Pydantic models for Code Analyst tool inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema

from core.llm.provider import TokenUsage


class FunctionAnalysis(BaseModel):
    """Structured analysis of a function or method."""

    qualified_name: str
    summary: str = ""
    purpose: str = ""
    parameters: list[str] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    module: str | None = None
    class_name: str | None = None
    error: str | None = None
    usage: SkipJsonSchema[TokenUsage | None] = None

    @field_validator("usage", mode="before")
    @classmethod
    def _coerce_usage(cls, value: object) -> TokenUsage | None:
        return coerce_analyst_usage(value)


class ClassAnalysis(BaseModel):
    """Structured analysis of a class."""

    qualified_name: str
    summary: str = ""
    purpose: str = ""
    methods: list[str] = Field(default_factory=list)
    bases: list[str] = Field(default_factory=list)
    decorators: list[str] = Field(default_factory=list)
    module: str | None = None
    error: str | None = None
    usage: SkipJsonSchema[TokenUsage | None] = None

    @field_validator("usage", mode="before")
    @classmethod
    def _coerce_usage(cls, value: object) -> TokenUsage | None:
        return coerce_analyst_usage(value)


class PatternInstance(BaseModel):
    """One graph hit for a named structural pattern.

    ``subject`` is the pattern's own name where it has one -- for the decorator
    pattern it is the decorator applied (``property``, ``dataclass``), while
    ``qualified_name`` is the entity it was applied to.
    """

    qualified_name: str
    subject: str = ""
    file_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    explanation: str = ""


class PatternAnalysis(BaseModel):
    """LLM explanation of pattern instances, or a structured unknown-pattern error."""

    pattern: str
    summary: str = ""
    instances: list[PatternInstance] = Field(default_factory=list)
    error: str | None = None
    supported_patterns: list[str] = Field(default_factory=list)
    usage: SkipJsonSchema[TokenUsage | None] = None

    @field_validator("usage", mode="before")
    @classmethod
    def _coerce_usage(cls, value: object) -> TokenUsage | None:
        return coerce_analyst_usage(value)


class SnippetResult(BaseModel):
    """Numbered source text retrieved without an LLM call."""

    file_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    text: str = ""
    error: str | None = None


class ImplementationExplanation(BaseModel):
    """Explanation-focused write-up of a function or method."""

    qualified_name: str
    explanation: str = ""
    error: str | None = None
    usage: SkipJsonSchema[TokenUsage | None] = None

    @field_validator("usage", mode="before")
    @classmethod
    def _coerce_usage(cls, value: object) -> TokenUsage | None:
        return coerce_analyst_usage(value)


class ImplementationComparison(BaseModel):
    """Side-by-side comparison of two implementations."""

    name_a: str
    name_b: str
    summary: str = ""
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    error: str | None = None
    usage: SkipJsonSchema[TokenUsage | None] = None

    @field_validator("usage", mode="before")
    @classmethod
    def _coerce_usage(cls, value: object) -> TokenUsage | None:
        return coerce_analyst_usage(value)


def coerce_analyst_usage(value: object) -> TokenUsage | None:
    """Accept a ``TokenUsage`` payload and drop malformed LLM extras.

    Args:
        value: Inbound usage object, mapping, or junk from a structured LLM.

    Returns:
        A validated :class:`TokenUsage`, or ``None`` when the value is absent
        or not usable.
    """
    if value is None:
        return None
    if isinstance(value, TokenUsage):
        return value
    try:
        return TokenUsage.model_validate(value)
    except (TypeError, ValueError):
        return None
