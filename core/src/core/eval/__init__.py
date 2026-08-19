"""Eval scoring for executed plans and answer quality."""

from core.eval.budget_compare import run_budget_comparison
from core.eval.harness import (
    eval_llm_provider,
    load_qa_cases,
    load_routing_cases,
    parse_eval_provider,
    routing_qa_cases,
    run_executed_plan_eval,
    run_qa_eval,
)
from core.eval.model_bakeoff import (
    format_purpose_table,
    parse_models_arg,
    run_router_bakeoff,
    run_synthesis_bakeoff,
)
from core.eval.models import (
    BudgetComparison,
    EvalScorecard,
    MetricScore,
    QaCase,
    TurnScore,
    TurnSpec,
)
from core.eval.report import (
    format_budget_comparison,
    format_scorecard_text,
    render_readme_section,
    write_readme_section,
    write_summary_section,
)
from core.eval.scoring import (
    agents_from_tools_invoked,
    executed_agents_ok,
    router_only_agents,
    score_retrieval_correctness,
    score_turn,
)

__all__ = [
    "BudgetComparison",
    "EvalScorecard",
    "MetricScore",
    "QaCase",
    "TurnScore",
    "TurnSpec",
    "agents_from_tools_invoked",
    "eval_llm_provider",
    "executed_agents_ok",
    "format_budget_comparison",
    "format_purpose_table",
    "format_scorecard_text",
    "load_qa_cases",
    "load_routing_cases",
    "parse_eval_provider",
    "parse_models_arg",
    "render_readme_section",
    "router_only_agents",
    "routing_qa_cases",
    "run_budget_comparison",
    "run_executed_plan_eval",
    "run_qa_eval",
    "run_router_bakeoff",
    "run_synthesis_bakeoff",
    "score_retrieval_correctness",
    "score_turn",
    "write_readme_section",
    "write_summary_section",
]
