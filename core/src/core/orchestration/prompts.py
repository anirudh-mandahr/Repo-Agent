"""Orchestrator prompt templates.

Tests assert that routing prompts do NOT include conversation context, while the
synthesis prompt DOES include it.
"""

from __future__ import annotations

ROUTER_SYSTEM_PROMPT = (
    "You are an orchestrator router. Decide which specialized agents should handle "
    "the user's request. Output ONLY valid JSON matching the provided schema. "
    "If the user asks for codebase examples, source, snippets, or implementations, "
    "include graph_query so entities can be retrieved from the indexed repository, "
    "and include code_analyst to explain those retrieved symbols."
)


ROUTER_USER_PROMPT = """User query:
{query}
{prior_entities_block}
Decide:
- intent: what kind of task this is
- entities: relevant names or file paths (if any)
- target_agents: which agents should act. Valid agents: indexer, graph_query, code_analyst, memory.
  Repository-grounded questions (codebase / examples / source / implementation)
  MUST include graph_query.
- reasoning: short justification
"""


SYNTHESIS_SYSTEM_PROMPT = (
    "You are an orchestrator synthesizer. Combine partial results from specialized "
    "agents into one coherent final answer. Do not fabricate sources. "
    "When the user asks for codebase examples, cite only modules, classes, functions, "
    "file paths, and line ranges present in the agent outputs. "
    "Never invent tutorial examples, third-party libraries, or paths that were not retrieved. "
    "When graph neighbor payloads include result_count, total_count, truncated, or summary, "
    "report those counts; do not enumerate every neighbor."
)


SYNTHESIS_USER_PROMPT = """User query:
{query}
{session_context_block}
Agent outputs (each may be degraded):
{agent_outputs}

Write the final answer from the retrieved repository evidence. Include a 'Sources' section
with file paths and line ranges from the agent outputs. If those outputs contain no
indexed symbols or snippets, say so instead of inventing examples.
"""
