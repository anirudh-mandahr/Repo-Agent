# FastAPI Repository Chat Agent

Five MCP servers (orchestrator, indexer, graph_query, code_analyst, memory) operate over a Neo4j knowledge graph of the FastAPI repository, fronted by a FastAPI gateway, all running in Docker Compose.

## Repo layout

```
repo/
  core/            # framework-free logic: ast parsing, graph ops, routing, llm provider interface
  agents/
    orchestrator/  # FastMCP server, MCP client of the other four
    indexer/
    graph_query/
    code_analyst/
    memory/
  gateway/         # FastAPI app
  docker-compose.yml
```

## Hard rules

- ALL business logic lives in `core/`. Agent packages are thin FastMCP adapters that import from `core/`. Never put parsing, Cypher, or LLM logic inside an agent package.
- Python 3.12, uv workspace, type hints everywhere, Pydantic v2 models for all tool inputs/outputs.
- Logging: structlog, JSON output, every log line carries `correlation_id`. No print statements.
- Only indexer and graph_query may open Neo4j sessions. code_analyst reads code via graph results + the read-only `/repo` volume.
- All LLM calls go through `core.llm.LLMProvider` (protocol). Tests use the StubProvider, never a real API.
- Neo4j writes are batched with `UNWIND`. All user-facing Cypher execution is read-only transactions.

## Commands

- `make up` / `make down` — compose
- `make test` — pytest with coverage on `core/`
- `make index` — trigger indexing
- `make smoke` — MCP smoke: find_entity("FastAPI") → get_dependents → explain_implementation
- `make lint` — ruff + mypy
