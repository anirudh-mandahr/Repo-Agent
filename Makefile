.PHONY: up down test test-all index lint smoke prove-incremental smoke-day3 eval eval-live tokens-report eval-models

.env:
	cp .env.example .env

up: .env
	docker compose up --build -d

down:
	docker compose down

COVERAGE_PACKAGES := --cov=core --cov=orchestrator --cov=indexer --cov=graph_query --cov=code_analyst --cov=memory --cov=gateway
COVERAGE_GATE := --cov-fail-under=79

test:
	uv sync --all-packages --group dev
	uv run pytest $(COVERAGE_PACKAGES) --cov-report=term-missing $(COVERAGE_GATE) -m "not integration and not live"

test-all:
	uv sync --all-packages --group dev
	uv run pytest $(COVERAGE_PACKAGES) --cov-report=term-missing $(COVERAGE_GATE)

prove-incremental:
	uv sync --all-packages --group dev
	uv run python scripts/prove_incremental.py

index: .env
	docker compose exec -T indexer python -c "from core.indexing import trigger_index; trigger_index()"

smoke: .env
	docker compose exec -T \
		-e SMOKE_GRAPH_QUERY_URL=http://graph_query:8003/mcp \
		-e SMOKE_CODE_ANALYST_URL=http://code_analyst:8004/mcp \
		gateway python -m core.mcp.smoke

smoke-day3: .env
	uv run --package gateway python scripts/smoke_day3.py

eval:
	uv sync --all-packages --group dev
	uv run python scripts/eval_all.py

eval-live:
	uv sync --all-packages --group dev
	@test -n "$$OPENROUTER_API_KEY" || (echo "eval-live requires OPENROUTER_API_KEY exported in the environment" && exit 1)
	@test -n "$$NEO4J_PASSWORD" || (echo "eval-live requires NEO4J_PASSWORD exported in the environment" && exit 1)
	EVAL_PROVIDER=live EVAL_WRITE_README=1 \
		NEO4J_URI=bolt://127.0.0.1:7687 \
		NEO4J_USER=$${NEO4J_USER:-neo4j} \
		NEO4J_PASSWORD=$$NEO4J_PASSWORD \
		uv run python scripts/eval_qa.py

tokens-report:
	uv sync --all-packages --group dev
	uv run python scripts/report_tokens.py

eval-models:
	uv sync --all-packages --group dev
	PYTHONUNBUFFERED=1 uv run python scripts/eval_models.py --write-readme

lint:
	uv sync --all-packages --group dev
	uv run ruff check .
	uv run mypy
