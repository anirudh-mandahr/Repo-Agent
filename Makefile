.PHONY: up down test test-all index lint smoke prove-incremental smoke-day3 eval tokens-report

.env:
	cp .env.example .env

up: .env
	docker compose up --build -d

down:
	docker compose down

test:
	uv sync --all-packages --group dev
	uv run pytest --cov=core/src/core --cov-report=term-missing --cov-fail-under=70 -m "not integration and not live"

test-all:
	uv sync --all-packages --group dev
	uv run pytest --cov=core/src/core --cov-report=term-missing --cov-fail-under=70

prove-incremental:
	uv sync --all-packages --group dev
	uv run python scripts/prove_incremental.py

index: .env
	docker compose exec -T indexer python -c "from core.indexing import trigger_index; trigger_index()"

smoke: .env
	uv run --package gateway python scripts/smoke_day2.py

smoke-day3: .env
	uv run --package gateway python scripts/smoke_day3.py

eval:
	uv sync --all-packages --group dev
	uv run python scripts/eval_all.py

tokens-report:
	uv sync --all-packages --group dev
	uv run python scripts/report_tokens.py

lint:
	uv sync --all-packages --group dev
	uv run ruff check .
	uv run mypy
