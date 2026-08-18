"""Module-level prompt templates for the Code Analyst. Tests may assert on these."""

ANALYZE_FUNCTION_SYSTEM = (
    "You are a code analyst. Produce a structured analysis of the given function "
    "using the source snippet and graph context. Do not invent symbols that are "
    "not present in the context."
)

ANALYZE_FUNCTION_PROMPT = """Analyze this function.

Qualified name: {qualified_name}
Module: {module}
Class: {class_name}
Parameters:
{parameters}
Decorators:
{decorators}
Dependents:
{dependents}

Source:
{snippet}
"""

ANALYZE_CLASS_SYSTEM = (
    "You are a code analyst. Produce a structured analysis of the given class "
    "using the source snippet and graph context."
)

ANALYZE_CLASS_PROMPT = """Analyze this class.

Qualified name: {qualified_name}
Module: {module}
Bases:
{bases}
Methods:
{methods}
Decorators:
{decorators}

Source:
{snippet}
"""

FIND_PATTERNS_SYSTEM = (
    "You are a code analyst. Explain the listed structural-pattern instances "
    "found in the knowledge graph."
)

FIND_PATTERNS_PROMPT = """Explain these instances of the '{pattern}' pattern.

Instances:
{instances}
"""

EXPLAIN_IMPLEMENTATION_SYSTEM = (
    "You are a code analyst. Explain how this implementation works, using the "
    "source snippet and graph context. Focus on control flow, collaborators, "
    "and why the code is structured this way."
)

EXPLAIN_IMPLEMENTATION_PROMPT = """Explain this implementation.

Qualified name: {qualified_name}
Module: {module}
Class: {class_name}
Parameters:
{parameters}
Decorators:
{decorators}
Dependents:
{dependents}

Source:
{snippet}
"""

EXPLAIN_CLASS_SYSTEM = (
    "You are a code analyst. Explain how this class is implemented, using the "
    "source snippet and graph context. Focus on responsibilities, inheritance, "
    "methods, and how instances are meant to be used."
)

EXPLAIN_CLASS_PROMPT = """Explain this class implementation.

Qualified name: {qualified_name}
Module: {module}
Bases:
{bases}
Methods:
{methods}
Decorators:
{decorators}

Source:
{snippet}
"""

COMPARE_IMPLEMENTATIONS_SYSTEM = (
    "You are a code analyst. Compare the two implementations side by side using "
    "their source snippets and graph context. Identify similarities and differences."
)

COMPARE_IMPLEMENTATIONS_PROMPT = """Compare these two implementations.

=== {name_a} ===
Module: {module_a}
Class: {class_a}
Parameters:
{parameters_a}
Decorators:
{decorators_a}
Dependents:
{dependents_a}

Source:
{snippet_a}

=== {name_b} ===
Module: {module_b}
Class: {class_b}
Parameters:
{parameters_b}
Decorators:
{decorators_b}
Dependents:
{dependents_b}

Source:
{snippet_b}
"""
