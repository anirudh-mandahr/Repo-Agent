# Design decisions and trade-offs

**Framework-free core with thin MCP adapters.** Parsing, Cypher, routing, LLM logic, and the Code Analyst's Graph Query client live in `core/`; agent packages only wire FastMCP transport. The analyst lookup uses the same pooled, circuit-breaker-protected MCP session as the orchestrator. Rejected embedding that logic inside agent packages, which would couple business rules to MCP and hinder unit testing.

**Streamable HTTP per agent container.** Each agent is an independent Docker service on an internal port. Rejected a single-process stdio MCP multiplex that would prevent independent scaling and health checks. Host publish is gateway `:8000` only.

**MCP health probes instead of TCP sockets.** Docker healthchecks call each agent's MCP `health` tool: Neo4j for `graph_query`, SQLite for `memory`, `/repo` plus graph_query reachability for `code_analyst`, downstream aggregation for `orchestrator`. Rejected bare TCP port opens that report healthy while dependencies are down.

**Shared-secret MCP mesh.** Compose sets `GATEWAY_API_KEY` (default `dev-gateway-key`) and reuses it as `MCP_SHARED_SECRET` so agents are not callable without the key even on the Docker network.

**Restricted Neo4j access.** Only indexer (writes) and graph_query (reads) open Neo4j sessions. Rejected letting every agent connect directly, which would blur write/read boundaries.

**Batched Neo4j writes with `UNWIND`.** Bulk upserts amortize round-trips during indexing. Rejected per-row individual Cypher transactions.

**File-granularity incremental indexing.** Content hashes skip unchanged files on re-index (`mode=incremental`). `mode=full` on `POST /api/index` (and the indexer tool) re-parses every file. Rejected treating the public `mode` parameter as documentation-only.

**Single specialist set, not multi-phase plans.** `ExecutionPlan.agents` is a flat list. The executor already sequences `code_analyst` behind `graph_query` when the analyst lacks coordinates. A second plan phase would duplicate that wait, and the router never emitted more than one phase, so nested `phases` was decoration.

**Heuristic evidence-driven refinement, not LLM replanning.** After a round, keyword/entity expansion and missing-agent suggestions produce a follow-up plan. No second routing LLM call.

**Plan starvation under a scaled synthesis reserve.** Settings validation only checks `plan_deadline_s + synthesis_reserve_s <= request_deadline_s` on the configured constants. At runtime a ~12k-token prompt scales the reserve to roughly 36s, so `plan_remaining_s` clamps to zero and heuristic refinement stops early. Retrieval is deliberately starved in favour of synthesizing evidence already gathered.

**Pluggable embeddings, hash by default.** The third `find_entity` tier is vector search behind `GQ_EMBEDDINGS_ENABLED` (default off; the flag is authoritative). `EMBEDDING_BACKEND` defaults to `hash` (`HashingEmbeddingProvider`, 256-dim bag-of-words) so indexing stays offline and free; `openrouter` calls a real model over the existing `OPENROUTER_API_KEY` and requests 256 dimensions so the Neo4j vector indexes need no migration. An API key alone does not switch the backend. Indexer and graph_query share one factory; the indexer records `embedding_fingerprint` on `:Meta` and graph_query disables the tier on mismatch. Score floors differ because Neo4j reports cosine as `(1 + cos) / 2` (0.6 hash, 0.7 model). Rejected a local ONNX/sentence-transformers runtime (new image weight, GPU optional) and rejected changing the index dimension without an explicit drop/recreate.

**Poll the indexer for job status, do not hold the MCP call open.** `POST /api/index` still returns a job id immediately, but the background task used to await `index_repository` under `GATEWAY_REQUEST_TIMEOUT_S` (10s). A full FastAPI index outlives that, so the gateway marked the job `failed` while the indexer container finished successfully. The gateway now fires the dispatch and follows `get_index_status`. Job records remain an in-process dict (lost on restart). Rejected raising the MCP timeout to “long enough” (unbounded, ties up pool slots) and rejected adding Redis solely to fix the timeout/status mismatch.

**Re-exported names resolve to the importing module, but only as a fallback.** `find_entity` joins import edges to the real `Class`/`Function`/`Method` node rather than synthesizing a coordinate, so `fastapi.FastAPI` resolves to `fastapi.applications.FastAPI` instead of a node that does not exist. Names re-exported from starlette (`WebSocket`, `JSONResponse`, `CORSMiddleware`, `Request`) have no node at all, because starlette is outside the indexed tree; a strict join drops them entirely and loses `fastapi/websockets.py`. The lookup therefore decides once per query rather than per row: if any import edge resolved to a real node, only real nodes are returned; otherwise it falls back to the importing `Module` nodes, preferring FastAPI-local ones. Rejected a per-row `coalesce(target, module)`, which would reinstate one hit per importer — `FastAPI` is imported by 569 modules and only one edge resolves. The fallback returns real `Module` nodes with real paths, so no fabricated coordinate re-enters the graph.

**Module citations use the header span.** Module nodes store the docstring or leading-import range, not `1–<file length>`. Citations omit line ranges for Module hits. Class and function line numbers are unchanged.

**Citations are normalized to repo paths before scoring.** Synthesis models sometimes render a coordinate from the qualified name (`fastapi.param_functions.py:2283`) instead of the `file_path` the evidence carried. The line numbers are correct; only the separator is wrong, and citation validation resolves the path against evidence, so the turn fails. `_normalize_dotted_paths` rewrites a dotted coordinate to its slash form **only when that path appears in the agent evidence**, so the step can never invent a citation it did not retrieve. Rejected relaxing the citation gate below 1.00, which would have hidden a real formatting defect.

**Heuristic multi-turn coreference.** Pronoun detection plus entity carry from prior user turns and the folded session summary. Not model-based coreference.

**First-class graph nodes for parameters, decorators, imports, docstrings.** Queries traverse typed nodes and relationships. Rejected storing these only as scalar properties on code nodes.

**Shared decorator nodes.** `:Decorator` nodes are excluded from file-subtree deletes so reused decorators survive reindex. Rejected deleting decorators with the owning file subtree.

**Guarded read-only Cypher.** User-facing `execute_query` rejects writes and injects `LIMIT`. Rejected raw user Cypher with no guard or cap.

**Best-effort `CALLS` resolution by name.** Call sites are linked without full type inference. Rejected waiting for perfect resolution before emitting any `CALLS` edges.

**LLMProvider protocol + StubProvider.** All LLM calls go through one interface; tests use deterministic stubs. Rejected real API calls in unit tests.

**Auditable guarded queries.** `execute_query` returns `cypher_executed` and `params` so callers can verify LIMIT injection. Rejected logging-only audit trails.

**Offline-green test contract.** `make test` excludes `integration` and `live` markers and runs without Docker, Neo4j, or API keys. Rejected relying solely on runtime skipif checks without marker-based exclusion.

**On-demand context, not static prompt files.** Repository knowledge lives in the Neo4j graph; conversational state lives in the Memory agent. Both are queried per request rather than loaded into every prompt upfront, so context is paid for only when needed.

**Response cache scoped to the session, not the whole deployment.** The key is `orchestrator:v1:{index_version}:{session_id}:{normalized_query}:{entity_fingerprint}` (`_cache_key` in `core/orchestration/service.py`). Re-indexing bumps `index_version` and invalidates stale entries automatically; the `session_id` component means an identical question asked in a different session is answered fresh rather than served another session's answer, and the resolved-entity fingerprint keeps a follow-up turn that resolved a pronoun to a different antecedent from colliding with the earlier turn. Rejected a deployment-wide `query + index_version` key, which is cheaper but leaks one session's synthesized answer into another's transcript.

**Degradation policy (HTTP status is exact).** Specialist timeouts and errors during retrieval still produce an answer: HTTP 200 with `degraded: true` and an incomplete-retrieval prefix when graph_query or code_analyst failed. Synthesis LLM timeout or error also returns HTTP 200: the orchestrator renders retrieved entity hits, dependents, and snippets as markdown (`evidence_only: true`, `degraded_reason` is the exception class name such as `TimeoutError`). Successful retrieval is never discarded because synthesis failed. `AgentUnavailableError` and `CircuitBreakerOpenError` (orchestrator MCP unreachable or a specialist breaker open) are HTTP 503. `GraphLookupError` is HTTP 503. `RoutingError` and `SchemaValidationError` are HTTP 422. Unhandled errors in the gateway itself are HTTP 500. A leaked `SynthesisError` that escapes `handle_query` is still mapped to HTTP 503; the synthesizer path itself no longer raises that error. These error bodies are declared on `/api/chat` in OpenAPI as `GatewayErrorBody`.

*Fixed 2026-08-21.* The degraded-200 half of this policy held when a specialist answered with an error or timed out at the MCP layer, but not when a specialist's **container was stopped**: `open_streamable_http_session` guarded its setup with `except Exception`, and the connect failure arrives as a `CancelledError` — a `BaseException` — because the MCP client issues `initialize` from a child task inside an `anyio` task group whose scope is cancelled on failure. The handler was skipped, the exit stack was never unwound in the entering task, and the escaping cancellation killed the orchestrator's request task without writing a response; the gateway then mapped its own `chat_timeout_s` ceiling to HTTP 503. The `RuntimeError: Attempted to exit cancel scope in a different task than it was entered in` was the orphaned stack being closed later from another task — a symptom, not the cause. The setup path now catches `BaseException`, unwinds in the entering task, and surfaces the underlying `ConnectError` as a transient `ConnectionError`, so a stopped specialist returns HTTP 200 with `degraded: true` (measured 21.1s cold, 19.0s warm). Genuine caller cancellation still propagates. The recorded walkthrough video shows the pre-fix 503. See [WALKTHROUGH.md](../WALKTHROUGH.md#failure-handling).

**Idempotent session folding.** A nullable `folded_at` column on SQLite turns prevents re-summarizing already folded history. Rejected deleting old turns immediately after summarization.

**Singleton `:Meta` for index metadata.** `index_version` and `last_indexed_at` are stored on one node keyed by `key`. Rejected deriving version ad hoc from file nodes on every statistics read.

**Secret filtering from repo env files.** Optional uncommitted overlay config files (`.env.{APP_ENV}`) remain ergonomic while API keys and passwords must be exported. Rejected loading secrets from committed `.env*` files; only `.env.example` is committed.

**Clone URL allowlist.** `clone_repo` rejects anything that is not `https` to an allowlisted host (default `github.com`) before spawning git. `file://`, link-local `http://`, and `git@` / SSH remotes are errors (`UnsafeCloneUrlError`). SSH is not enabled: it can target internal SSH endpoints and is a common metacharacter-injection vehicle; HTTPS to GitHub is enough for the default FastAPI clone.

**No default Neo4j password.** `GraphSettings` requires `NEO4J_PASSWORD`. Compose interpolates `NEO4J_AUTH` / `NEO4J_PASSWORD` with no fallback value.

**Entity extraction in fallback routing.** When LLM routing fails, rule-based fallback still extracts likely entities for specialist lookups. Rejected silent no-op specialists on routing failure.

**Deterministic rule-based routing for golden evals.** Simple lookups route to `graph_query` only; multi-part analysis escalates to LLM routing with multiple agents. Rejected defaulting most queries to mixed multi-agent plans.

**Synthesis short-circuit for absent entities.** When graph and snippet lookups find nothing, synthesis returns an honest out-of-repo message without calling the LLM. Rejected always prompting the LLM with empty results (hallucination risk on trap queries).

**Unified eval entrypoint.** `make eval` runs executed-plan routing checks, the QA scorecard, and trap pytest in one pass. Rejected separate manual eval commands.

**CI eval with host-cloned FastAPI.** The eval job clones FastAPI on the runner and indexes via Docker Compose so trap tests see both host-readable paths and the live graph. Rejected container-only repo volumes for integration evals.

**Rules-first routing with LLM escalation.** Cheap deterministic routing handles simple queries; ambiguous or long queries call the LLM router. Rejected always routing through the LLM first.

**In-process token ledger.** Per-request token totals are accumulated and emitted in gateway `done` events, including the resolved model, cached vs uncached prompt tokens, and per-model cost. Code analyst LLM calls (`explain_implementation`, `analyze_function`, and the other analysis tools) return usage on the MCP payload and are recorded under `by_purpose.analysis`, so `ORCH_REQUEST_TOKEN_BUDGET` / `ORCH_REQUEST_COST_USD_MAX` can see the largest consumer. Rejected provider-only logs with no request-level aggregation.

**Purpose-aware model selection.** `complete(purpose=...)` selects `ORCH_MODEL_ROUTING`, `ORCH_MODEL_SYNTHESIS`, or `CA_MODEL_ANALYSIS`. Routing and analysis fall back to `OPENROUTER_MODEL` (`anthropic/claude-sonnet-4.5`); synthesis defaults to `openai/gpt-4.1-mini`. Routing stays on sonnet-4.5 because it passes 61% of routing traps versus 33% for gpt-4.1-mini. Synthesis uses gpt-4.1-mini because the bake-off measured 89% quality at 11,714 ms p95 and $1.18/1k versus sonnet-4.5's 82% at 19,943 ms and $14.43/1k, with the same 83% trap pass rate.

**Unmeasured synthesis-reserve slope.** `REFERENCE_SYNTHESIS_PROMPT_TOKENS` (4000) and `SYNTHESIS_EXTRA_SECONDS_PER_TOKEN` (0.002) are unmeasured engineering defaults. The bake-off recorded p95 per model, not a latency-versus-prompt-size slope; a fitted slope from per-case bake-off latency and prompt tokens would validate them.

**Prompt caching on static system prompts.** Anthropic models send `cache_control: ephemeral` on the first system message. Cached vs uncached input tokens are recorded separately in the ledger.


---

## Request budget and latency hierarchy

These layers share one wall-clock; they are not independent timeouts stacked to 55s+.

1. **Outer request budget** (`RequestBudget` in `core/orchestration/budget.py`) spans routing + every specialist call + synthesis. Ceilings: `ORCH_REQUEST_DEADLINE_S` (default 55s), `ORCH_REQUEST_TOKEN_BUDGET`, `ORCH_REQUEST_COST_USD_MAX`. Spend is read from the token ledger's `cost_usd` (priced per purpose-resolved model via `core.llm.pricing`), including code_analyst LLM calls recorded under `metadata.tokens.by_purpose.analysis`. When a ceiling trips mid-plan, no new calls are issued and synthesis uses evidence already gathered (`metadata.budget_exhausted` = `deadline` \| `tokens` \| `cost`).
2. **Plan deadline** (`ORCH_PLAN_DEADLINE_S`, default 35s) bounds heuristic refinement rounds. Specialist calls also stop once remaining request time drops to the effective synthesis reserve. `ORCH_SYNTHESIS_RESERVE_S` (default 20s) is the floor for a typical prompt; at runtime the reserve scales with the actual synthesis prompt token count from `core/orchestration/prompt_budget.py` so a bulky mixed (non-comparison) fan-out claims more wall-clock than the constant. Settings validation still requires `plan_deadline_s + synthesis_reserve_s <= request_deadline_s` on the configured constants only, not the scaled runtime reserve: a ~12k-token prompt raises that reserve to roughly 36s, `plan_remaining_s` clamps to zero, and heuristic refinement stops early so retrieval is starved in favour of synthesizing evidence already gathered. The same validation still requires `synthesis_reserve_s − synthesis_safety_margin_s >=` the bake-off synthesis p95 of the resolved `ORCH_MODEL_SYNTHESIS` model. After a full plan the derived synthesis timeout is 18s for a typical prompt (`55 - 35 - 2`), which covers `openai/gpt-4.1-mini` p95 (11.7s) with live-prompt margin; `anthropic/claude-sonnet-4.5` p95 (19.9s) still does not fit, which is why it is not the default synthesizer. Each request logs `orchestrator.synthesis_window` with the derived `synthesis_timeout_s` and `plan_duration_s` (Prometheus: `repochat_synthesis_timeout_seconds`, `repochat_plan_duration_seconds`).
3. **Per-agent MCP timeouts** (`ORCH_GRAPH_QUERY_TIMEOUT_S`, `ORCH_CODE_ANALYST_TIMEOUT_S`, …) are clipped to remaining request time minus the (possibly scaled) synthesis reserve.
4. **Synthesis timeout** is derived: `remaining(request_deadline) − ORCH_SYNTHESIS_SAFETY_MARGIN_S`. `ORCH_SYNTHESIS_TIMEOUT_S` is the typical-prompt cap; large prompts raise that cap so leftover request time is not left unused. Because the plan phase cannot spend the reserve, this derived timeout stays above zero even when planning uses its full deadline.
5. **Per-prompt synthesis token budget** (`ORCH_SYNTHESIS_PROMPT_TOKEN_BUDGET`) is unchanged and only truncates the synthesis prompt.

Synthesis tokens stream from the LLM provider through orchestrator MCP progress notifications (streamable HTTP SSE; `json_response=False` on the orchestrator only) into the existing SSE/WebSocket `answer` events, so time-to-first-token stays under 2s of synthesis start. Clients that do not pass a progress callback, and the non-streaming JSON chat path, still wait for the full answer and then post-hoc chunk it. A timeout or mid-stream error keeps tokens already emitted and marks the answer `partial` rather than replacing it with the evidence-only renderer.

Prometheus: `repochat_synthesis_duration_seconds`, `repochat_time_to_first_token_seconds`, `repochat_synthesis_timeout_seconds`, `repochat_plan_duration_seconds`, `repochat_budget_exhausted_total{ceiling}`, `repochat_evidence_only_total`.

**Diagnostic long-budget profile (env only).** Defaults stay at 55 / 35 / 20 / 90. For live two-round queries that need more synthesis headroom, set these in the environment (not as committed defaults): `ORCH_REQUEST_DEADLINE_S=110`, `ORCH_PLAN_DEADLINE_S=45`, `ORCH_SYNTHESIS_RESERVE_S=60`, `GATEWAY_CHAT_TIMEOUT_S=130`. Settings still require `plan_deadline_s + synthesis_reserve_s <= request_deadline_s` (45 + 60 = 105 ≤ 110) and `synthesis_reserve_s − synthesis_safety_margin_s` ≥ the bake-off synthesis p95 of `ORCH_MODEL_SYNTHESIS`. Raise the gateway timeout too: `GatewaySettings.chat_timeout_s` (default 90s) is the timeout on the orchestrator `handle_query` call in `gateway/src/gateway/mcp_client.py`, so raising only the orchestrator deadline moves the failure to the gateway.
