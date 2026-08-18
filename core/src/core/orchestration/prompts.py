"""Orchestrator prompt templates.

Tests assert that routing prompts do NOT include conversation context, while the
synthesis prompt DOES include it.
"""

from __future__ import annotations

ROUTER_SYSTEM_PROMPT = (
    "You are an orchestrator router. Decide which specialized agents should handle "
    "the user's request. Output ONLY valid JSON matching the provided schema."
)


ROUTER_USER_PROMPT = """User query:
{query}

Decide:
- intent: what kind of task this is
- entities: relevant names or file paths (if any)
- target_agents: which agents should act
- reasoning: short justification
"""


SYNTHESIS_SYSTEM_PROMPT = (
    "You are an orchestrator synthesizer. Combine partial results from specialized "
    "agents into one coherent final answer. Do not fabricate sources."
)


SYNTHESIS_USER_PROMPT = """User query:
{query}
{session_context_block}
Agent outputs (each may be degraded):
{agent_outputs}

Write the final answer. Include a 'Sources' section with best-effort file paths and
line ranges when available.
"""

