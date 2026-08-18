"""Pydantic models for Code Analyst tool inputs and outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


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


class PatternInstance(BaseModel):
    """One graph hit for a named structural pattern."""

    qualified_name: str
    file_path: str = ""
    line_start: int | None = None
    line_end: int | None = None
    explanation: str = ""


class PatternAnalysis(BaseModel):
    """LLM explanation of pattern instances, or a structured unknown-pattern error."""

    pattern: str
    instances: list[PatternInstance] = Field(default_factory=list)
    error: str | None = None
    supported_patterns: list[str] = Field(default_factory=list)


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


class ImplementationComparison(BaseModel):
    """Side-by-side comparison of two implementations."""

    name_a: str
    name_b: str
    summary: str = ""
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    error: str | None = None
