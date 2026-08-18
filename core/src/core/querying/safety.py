"""Reject write Cypher before it reaches a read-only Neo4j transaction."""

from __future__ import annotations

import re

WRITE_CLAUSES: tuple[tuple[str, str], ...] = (
    ("CREATE", r"\bCREATE\b"),
    ("MERGE", r"\bMERGE\b"),
    ("DELETE", r"\bDELETE\b"),
    ("DETACH", r"\bDETACH\b"),
    ("SET", r"\bSET\b"),
    ("REMOVE", r"\bREMOVE\b"),
    ("DROP", r"\bDROP\b"),
    ("CALL db.*", r"\bCALL\s+DB\."),
    ("LOAD CSV", r"\bLOAD\s+CSV\b"),
)

_CLAUSE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (clause, re.compile(pattern)) for clause, pattern in WRITE_CLAUSES
)

_LIMIT_RE = re.compile(r"\bLIMIT\b")


class QueryRejected(Exception):
    """Raised when ``guard_readonly`` finds a write clause."""

    def __init__(self, clause: str) -> None:
        self.clause = clause
        super().__init__(f"rejected write clause: {clause}")


def strip_comments_and_strings(cypher: str) -> str:
    """Remove comments and mask string/identifier literals, preserving layout."""
    out: list[str] = []
    i = 0
    length = len(cypher)
    while i < length:
        char = cypher[i]
        nxt = cypher[i + 1] if i + 1 < length else ""
        if char == "/" and nxt == "/":
            i += 2
            while i < length and cypher[i] != "\n":
                i += 1
            continue
        if char == "/" and nxt == "*":
            i += 2
            while i + 1 < length and not (cypher[i] == "*" and cypher[i + 1] == "/"):
                i += 1
            i = min(i + 2, length)
            continue
        if char in {"'", '"', "`"}:
            quote = char
            i += 1
            while i < length:
                current = cypher[i]
                if current == "\\" and i + 1 < length:
                    i += 2
                    continue
                if current == quote:
                    if i + 1 < length and cypher[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                i += 1
            out.append(" ")
            continue
        out.append(char)
        i += 1
    return "".join(out)


def normalize_cypher(cypher: str) -> str:
    """Comment-stripped, string-masked, uppercase copy used for keyword scans."""
    return strip_comments_and_strings(cypher).upper()


def guard_readonly(cypher: str) -> str:
    """Reject Cypher that contains a write clause. Returns ``cypher`` unchanged."""
    normalized = normalize_cypher(cypher)
    earliest: tuple[int, str] | None = None
    for clause, pattern in _CLAUSE_PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        start = match.start()
        if earliest is None or start < earliest[0]:
            earliest = (start, clause)
    if earliest is not None:
        raise QueryRejected(earliest[1])
    return cypher


def has_limit(cypher: str) -> bool:
    """Return True when a LIMIT clause is present outside comments and strings."""
    return _LIMIT_RE.search(normalize_cypher(cypher)) is not None
