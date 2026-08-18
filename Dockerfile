# syntax=docker/dockerfile:1

# Multi-stage uv image parameterized by PACKAGE (workspace member to run).
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_NO_DEV=1

WORKDIR /app

ARG PACKAGE

COPY pyproject.toml uv.lock ./
COPY core/pyproject.toml core/pyproject.toml
COPY gateway/pyproject.toml gateway/pyproject.toml
COPY agents/orchestrator/pyproject.toml agents/orchestrator/pyproject.toml
COPY agents/indexer/pyproject.toml agents/indexer/pyproject.toml
COPY agents/graph_query/pyproject.toml agents/graph_query/pyproject.toml
COPY agents/code_analyst/pyproject.toml agents/code_analyst/pyproject.toml
COPY agents/memory/pyproject.toml agents/memory/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-workspace --package "${PACKAGE}"

COPY core /app/core
COPY gateway /app/gateway
COPY agents /app/agents

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-editable --package "${PACKAGE}"

FROM python:3.12-slim-trixie

RUN groupadd --system --gid 999 app \
    && useradd --system --gid 999 --uid 999 --create-home app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

ARG PACKAGE
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PACKAGE="${PACKAGE}"

RUN if [ "$PACKAGE" = "indexer" ]; then \
      apt-get update \
      && apt-get install -y --no-install-recommends git ca-certificates \
      && rm -rf /var/lib/apt/lists/*; \
    fi

USER app
WORKDIR /app

CMD ["sh", "-c", "exec python -m \"$PACKAGE\""]
