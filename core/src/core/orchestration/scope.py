"""Out-of-repository trap phrases that must refuse without FastAPI hallucination."""

from __future__ import annotations

import re

# Trap evals (evals/traps.jsonl / evals/qa.jsonl t01–t06). "logging" is a real
# FastAPI symbol, so the PostgreSQL WAL phrasing must be listed explicitly;
# incidental fulltext/exact hits on logger.py must not count as in-scope.
_OUT_OF_SCOPE_RE = re.compile(
    r"\b(?:django|rails|activerecord|kubernetes|tensorflow|gradienttape|"
    r"postgresql|postgres|useeffect)\b|"
    r"write[-\s]ahead\s+log(?:ging)?|"
    r"\breact(?:'s)?\b",
    re.IGNORECASE,
)


def is_out_of_scope(query: str) -> bool:
    """True when ``query`` asks about a system outside the indexed FastAPI repo.

    Args:
        query: User question.

    Returns:
        Whether the orchestrator should refuse rather than answer from incidental hits.
    """
    return bool(_OUT_OF_SCOPE_RE.search(query or ""))
