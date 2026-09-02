# Regression diagnosis and fix (2026-08-20)

Three tree states are compared. All live runs are 58 turns over `evals/qa.jsonl`,
routing `anthropic/claude-sonnet-4.5`, synthesis `openai/gpt-4.1-mini`.

| State | What it is |
| --- | --- |
| **A — baseline** | The previous session's live run, $1.8904 (`baseline-before/`) |
| **B — new fixes** | + entity-resolution / correlation-id / MCP-timeout fixes (`after-fixes/`) |
| **C — final** | + the two fixes made in this session (`final-live/`) |

## Headline: the regression was real, and worse than the scorecard suggested

Live A -> B:

| Metric | A (baseline) | B (new fixes) | Delta |
| --- | ---: | ---: | ---: |
| Tokens | 967,385 | 867,434 | **-99,951 (-10%)** |
| Cost | $1.8904 | $2.3268 | **+$0.4364 (+23%)** |
| Mean latency | 13,955 ms | 15,124 ms | +1,169 ms |
| Turns passed | 50/58 | 48/58 | **-2** |

Offline (stub provider, same tree, no LLM variance — this is the clean signal):

| Metric | B | C | 
| --- | ---: | ---: |
| retrieval_correctness | **0.865** | **0.981** |
| entity_recall | 0.962 | 0.981 |
| Failing turns | **9** | **2** |

State B's offline retrieval of 0.865 is well below the 0.96 the old README recorded.
The regression was not marginal.

## Final result (C)

| Metric | A baseline | B your fixes | C final |
| --- | ---: | ---: | ---: |
| Turns passed | 50/58 | 48/58 | **54/58** |
| `hard_assertions` | FAIL | FAIL | **PASS** |
| citation_precision | 0.97 FAIL | 1.00 FAIL | **1.00 PASS** |
| retrieval_correctness | 0.97 | **0.87** | **0.99** |
| entity_recall | 0.98 | 0.97 | **0.99** |
| groundedness | 0.96 | 0.94 | 0.93 |
| Tokens | 967,385 | 867,434 | **856,140** |
| Cost | $1.8904 | $2.3268 | **$2.1440** |
| Mean latency | 13,955 ms | 15,124 ms | 14,564 ms |

C is the first run of the three where the live eval passes its hard assertions.
Retrieval and entity recall now exceed the original baseline, and citation precision
reaches a true 1.00. Cost remains ~13% above baseline ($2.14 vs $1.89): the residual
increase is the model-mix shift described above, not evidence volume — tokens are 11%
*below* baseline.

Per tier (C): simple 100%, medium 92%, complex 92%, trap 100% (refusal 100%),
multi-turn 88%.

Turns fixed by these changes: `s10`, `c10`, `m10`, `mt06`, `mt06` t2 (re-export
fallback), `mt03` t2 (snake_case carry), `c06` (citation normalization).

Remaining 4 failures: `c09` (entity, known limitation), `mt04` t2 (retrieval, known
limitation), `m04` and `mt01` t2 (groundedness 0.64 / 0.42 against the 0.70 gate —
LLM-quality variance, not retrieval).

## What I got wrong first

My initial hypothesis was that the phantom-import fix would cut the expensive turns.
**It did not.** The five most expensive turns all got *more* expensive in B:

| Case | A tokens | B tokens | A cost | B cost |
| --- | ---: | ---: | ---: | ---: |
| c04 | 119,295 | 128,003 | $0.341 | $0.373 |
| c08 | 119,187 | 128,201 | $0.340 | $0.375 |
| m03 | 72,037 | 76,038 | $0.202 | $0.229 |
| m05 | 69,607 | 74,323 | $0.199 | $0.225 |
| m06 | 62,751 | 68,344 | $0.170 | $0.202 |

The -10% token change came from elsewhere: cheap simple/multi-turn cases dropped hard
(s03 10,078 -> 1,484; mt02 10,041 -> 1,527; s08 4,214 -> 760). Meanwhile a handful of
turns became far more expensive per token, because they shifted onto the expensive
routing model rather than cheap synthesis — c12 went $0.008 -> $0.076 on only 1.8x the
tokens (roughly $0.6/M -> $3.2/M effective). **That model-mix shift, not token volume,
is why cost rose while tokens fell.**

## Root cause of the quality regression: re-exports of un-indexed symbols

`FIND_IMPORTED_NAME` was changed to stop synthesizing coordinates and instead join to
the real `Class`/`Function`/`Method` node. That was correct for `fastapi.FastAPI` and it
is what fixed the entity-resolution contract tests.

But some names have **no node in the graph at all**, because they are defined in
starlette, which is not indexed. Measured against the live graph:

| Name | Real nodes | Importing modules | FastAPI-local importers |
| --- | ---: | ---: | ---: |
| FastAPI | 1 | 569 | 1 |
| APIRouter | 1 | 39 | 1 |
| Depends | 2 | 125 | 2 |
| get_openapi | 1 | 3 | 1 |
| **WebSocket** | **0** | 18 | 5 |
| **JSONResponse** | **0** | 30 | 5 |
| **CORSMiddleware** | **0** | 2 | 1 |
| **Request** | **0** | 37 | 10 |

For those zero-node names the new `MATCH (target)` matched nothing, so `find_entity`
returned **no rows** and the file vanished from retrieval. Concretely, "What is
WebSocket?" retrieved:

- A: `fastapi/websockets.py`, `fastapi/routing.py`, `fastapi/exception_handlers.py`, `fastapi/dependencies/utils.py`, `fastapi/__init__.py`
- B: `fastapi/applications.py`, `fastapi/routing.py`, `docs_src/...`

The old fabricated-coordinate behaviour was accidentally carrying correct *file*
attribution for re-exported names. Removing it fixed the coordinates and lost the files.
That cost 6 turn-failures in B: s10, m04, m10, c10, mt06, mt06 t2.

## Fixes applied in this session

### 1. Re-export fallback in `FIND_IMPORTED_NAME` (`core/src/core/querying/templates.py`)

Decide **globally**, not per row: if any row resolved to a real node, return only real
nodes; otherwise fall back to the importing `Module` nodes, preferring FastAPI-local
ones. The fallback returns real `Module` nodes with real `file_path`s — nothing is
fabricated, so the contract tests still hold.

A per-row `coalesce(target, m)` would have been wrong: `FastAPI` has 569 importers and
only one resolves, so per-row fallback would have reinstated the 568-hit explosion. The
global guard is what keeps both properties at once. Verified against the graph:

- `FastAPI` -> 1 `Class` node (not 569 modules)
- `WebSocket` -> 5 real `Module` nodes including `fastapi/websockets.py`
- `JSONResponse` -> includes `fastapi/responses.py`; `CORSMiddleware` -> `fastapi/middleware/cors.py`

### 2. snake_case identifiers in entity carry (`core/src/core/orchestration/router.py`)

The identifier regex matched CamelCase and dotted paths but never a bare snake_case
name, so `"Where is get_openapi?"` extracted nothing and the follow-up turn
`"Explain how that is implemented"` had no antecedent to carry (mt03 t2 scored 0.00
entity recall / 0.00 retrieval, offline as well as live). Added one alternation
requiring at least one underscore, so prose words cannot be read as identifiers.
Checked against all 73 eval queries: adds exactly `get_openapi` in 7 places, zero
false positives.

### 3. Citation path normalization (`core/src/core/orchestration/synthesis.py`)

Every citation-precision failure was the same shape — `fastapi.param_functions.py:2283`
instead of `fastapi/param_functions.py:2283`. These are **not hallucinations**: the line
numbers are all valid (`param_functions.py` has 2,460 lines, `params.py` 754,
`routing.py` 6,447, `applications.py` 4,774). The model rendered the coordinate from the
qualified name. `_normalize_dotted_paths` rewrites a dotted coordinate to the slash path
**only when that path actually appears in the agent evidence**, so it can never invent a
citation.

## Deliberately not fixed

`c09` ("Explain how routing, OpenAPI, and dependencies connect across the codebase",
missing `APIRouter` / `fastapi/routing.py`) and `mt04` t2 ("Show me examples of that",
missing `fastapi/dependencies/utils.py`) still fail offline. Both could be forced green
by adding entries to `_CONCEPT_ENTITIES` mapping "routing" -> `APIRouter` and
`Depends` -> `dependencies/utils.py`. That is fitting the mapping table to the eval set,
so they are left failing and documented as known limitations instead.

## Guardrail status

The 9 sample queries in `evals/routing.jsonl` are **9/9 PASS** after every change,
with all expected agents matched and retrieval PASS on each. Offline suite: 496 passed
(490 + 6 new tests). Integration + live: 14 passed.
