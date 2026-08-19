"""Bounded evidence-driven refinement: execute, evaluate, expand terms/agents, repeat."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence

from core.logging import get_logger
from core.memory import ConversationContext
from core.observability.ledger import TokenLedger
from core.settings import OrchestratorSettings

from .budget import RequestBudget
from .evidence import evaluate_evidence, merge_agent_outputs, refine_plan
from .executor import run_plan
from .models import AgentName, AgentOutput, ExecutionPlan, PlanIteration
from .prompt_budget import estimated_synthesis_prompt_tokens

log = get_logger(__name__)


def _tools_invoked(
    plan: ExecutionPlan, agent_outputs: Mapping[AgentName, AgentOutput]
) -> list[str]:
    tools: list[str] = []
    for agent in plan.agents:
        output = agent_outputs.get(agent)
        invoked = getattr(output, "tools_invoked", None) if output is not None else None
        if invoked:
            tools.extend(list(invoked))
    return tools


async def run_refinement_loop(
    plan: ExecutionPlan,
    *,
    query: str,
    context: ConversationContext | None,
    clients: object,
    settings: OrchestratorSettings,
    correlation_id: str,
    deadline_monotonic: float | None = None,
    budget: RequestBudget | None = None,
    token_ledger: TokenLedger | None = None,
) -> tuple[dict[AgentName, AgentOutput], list[PlanIteration]]:
    """Execute ``plan`` and heuristically refine while evidence is insufficient.

    Follow-up rounds expand keywords/entities and missing agents. This is not
    an LLM replan.

    Args:
        plan: Initial execution plan.
        query: User question.
        context: Conversation context.
        clients: Specialist MCP clients.
        settings: Orchestrator settings including iteration/deadline caps.
        correlation_id: Request correlation id.
        deadline_monotonic: Optional ``time.monotonic()`` cutoff for all rounds.
        budget: Optional outer request budget. Stops new rounds when exhausted.
        token_ledger: Ledger whose ``cost_usd`` the budget reads.

    Returns:
        Combined agent outputs and one :class:`PlanIteration` per executed round.
    """
    max_iterations = max(1, settings.max_plan_iterations)
    plan_cutoff = time.monotonic() + settings.plan_deadline_s
    if deadline_monotonic is not None:
        plan_cutoff = min(plan_cutoff, deadline_monotonic)
    if budget is not None:
        reserved_deadline = budget.deadline_monotonic - budget.synthesis_reserve_s
        plan_cutoff = min(plan_cutoff, reserved_deadline)
    deadline = plan_cutoff

    combined: dict[AgentName, AgentOutput] = {}
    iterations: list[PlanIteration] = []
    current = plan
    assessment = None

    for round_index in range(max_iterations):
        if budget is not None and not budget.allow_new_call(token_ledger, correlation_id):
            log.info(
                "orchestrator.budget_exhausted",
                ceiling=budget.exhausted,
                iteration=round_index + 1,
            )
            break
        if round_index > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log.info(
                    "orchestrator.plan_deadline",
                    iteration=round_index + 1,
                    max_iterations=max_iterations,
                )
                break
            if assessment is None:
                break
            follow_up = refine_plan(query, current, assessment)
            if follow_up is None:
                break
            current = follow_up

        round_outputs = await run_plan(
            current,
            query=query,
            context=context,
            clients=clients,  # type: ignore[arg-type]
            settings=settings,
            correlation_id=correlation_id,
            budget=budget,
            token_ledger=token_ledger,
        )
        combined = (
            merge_agent_outputs(combined, round_outputs) if combined else dict(round_outputs)
        )
        assessment = evaluate_evidence(query, current, combined)
        if budget is not None:
            budget.apply_prompt_reserve(
                estimated_synthesis_prompt_tokens(
                    query,
                    combined,
                    token_budget=settings.synthesis_prompt_token_budget,
                    order=settings.prompt_truncation_order,
                )
            )
        search_terms: Sequence[str]
        if current.search_terms:
            search_terms = current.search_terms
        else:
            search_terms = current.intent.entities
        tools = _tools_invoked(current, round_outputs)
        iteration = PlanIteration(
            iteration=current.iteration,
            routing_mode=current.routing_mode,
            agents=list(current.agents),
            tools_invoked=tools,
            search_terms=list(search_terms),
            sufficient=assessment.sufficient,
            reason=assessment.reason,
            refinement=current.refinement_reason or None,
        )
        iterations.append(iteration)
        log.info(
            "orchestrator.plan_iteration",
            iteration=iteration.iteration,
            sufficient=iteration.sufficient,
            reason=iteration.reason,
            agents=iteration.agents,
            tools_invoked=tools,
        )
        if assessment.sufficient:
            break

    return combined, iterations


__all__ = ["run_refinement_loop"]