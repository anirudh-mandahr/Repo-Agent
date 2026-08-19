# Gateway API reference

Base URL: `http://localhost:8000`. Compose sets `GATEWAY_API_KEY` to `dev-gateway-key` by default. Pass `X-API-Key: <key>` on all routes except `GET /health`, `GET /api/agents/health`, and the OpenAPI docs. `GET /metrics` requires the API key when one is configured. Agent MCP servers share the same secret (`MCP_SHARED_SECRET`) — including their `/metrics` routes — and are not published to the host.

| Method | Path | Description |
| --- | --- | --- |
| `POST` | `/api/chat` | Chat (JSON body; `stream=true` for SSE) |
| `WS` | `/ws/chat` | WebSocket chat (same event protocol) |
| `POST` | `/api/index` | Start background index job (`mode`: `incremental` hash-skip or `full` re-parse) |
| `GET` | `/api/index/status/{job_id}` | Poll index job status |
| `GET` | `/api/agents/health` | Per-agent health aggregation |
| `GET` | `/api/graph/statistics` | Graph counts + `index_version` |
| `GET` | `/health` | Alias for agents health |
| `GET` | `/metrics` | Prometheus metrics (request count, latency, errors, cache, LLM tokens/cost) |

Chat messages are capped at 8,000 characters (`GATEWAY_MAX_MESSAGE_CHARS`; `422` when exceeded). The gateway applies a sliding-window rate limit (`GATEWAY_RATE_LIMIT_REQUESTS` / `GATEWAY_RATE_LIMIT_WINDOW_S`, default 60 requests per 60s) to HTTP routes **and** to `/ws/chat` (handshake and each message) and returns `429` / WebSocket close `1008` when exceeded. Each agent and the gateway expose `GET /metrics` behind the API key. OpenTelemetry spans share the request `correlation_id` across MCP calls so one chat query is a single trace (`OTEL_EXPORTER_OTLP_ENDPOINT` to export). Interactive OpenAPI is at `/docs`.

## Examples

**Chat (JSON)**

```bash
curl -s http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-gateway-key' \
  -d '{"message":"What is the FastAPI class?","session_id":"demo-1"}' | jq .
```

**Chat (SSE)**

```bash
curl -N http://localhost:8000/api/chat \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-gateway-key' \
  -d '{"message":"What is the FastAPI class?","stream":true}'
```

**WebSocket** — connect to `ws://localhost:8000/ws/chat` with `X-API-Key: dev-gateway-key`. Auth and the rate limiter run on the handshake; each subsequent JSON message is also rate-limited. Send JSON shaped like `ChatRequest`:

```json
{"message": "What is the FastAPI class?", "session_id": "demo-1"}
```

**Start index job**

```bash
curl -s http://localhost:8000/api/index \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: dev-gateway-key' \
  -d '{"mode":"incremental"}' | jq .
```

**Index job status**

```bash
curl -s http://localhost:8000/api/index/status/<job_id> \
  -H 'X-API-Key: dev-gateway-key' | jq .
```

**Agents health**

```bash
curl -s http://localhost:8000/api/agents/health | jq .
```

**Graph statistics**

```bash
curl -s http://localhost:8000/api/graph/statistics \
  -H 'X-API-Key: dev-gateway-key' | jq .
```

## SSE / WebSocket event protocol

Both transports emit the same four event types in order:

| Event | Payload highlights |
| --- | --- |
| `routing` | `routing_mode`, `agents`, `cached`, `degraded` |
| `agent_result` | `agent`, `ok`, `cached`, `degraded` |
| `answer` | `chunk` (live synthesis token when the orchestrator MCP session negotiates progress; otherwise the gateway splits the finished answer at `GATEWAY_ANSWER_CHUNK_CHARS`) |
| `done` | `latency_ms`, `cached`, `degraded`, `routing_mode`, `tokens` (`total`, `prompt`, `completion`, `cached_prompt`, `uncached_prompt`, `llm_calls`, `cost_usd`, `by_purpose`, `by_model`); when synthesis fell back, also `evidence_only`, `degraded_reason`, and optionally `prompt_truncated`; when tokens already streamed were kept, also `partial` |

**SSE format:** `event: <type>\ndata: {"type":"<type>","correlation_id":"...","..."}\n\n`

**WebSocket format:** `{"type":"<type>","correlation_id":"...","data":{...}}`

Every response includes an `x-correlation-id` header for log correlation.

## Security

- **Auth.** `X-API-Key` is required on chat, index, graph, metrics, and WebSocket when `GATEWAY_API_KEY` is set. `GET /health` and `GET /api/agents/health` stay unauthenticated for probes. Agent MCP `/metrics` uses the same shared secret.
- **Rate limits.** The sliding-window limiter applies to HTTP middleware, the `/ws/chat` handshake, and each WebSocket message.
- **Clone SSRF.** Indexer `repo_url` must be `https` to a host in `INDEXER_CLONE_ALLOWED_HOSTS` (default `github.com`). `file://`, `http://`, SSH, and `git@` remotes are rejected in `core` before git runs.
- **Secrets.** No Neo4j password default in code or Compose. Committed `.env.example`, `.env.development`, and `.env.production` omit secret values.

