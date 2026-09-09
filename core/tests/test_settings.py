"""Settings env-file precedence, overlays, and .env.example completeness."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from core.settings import GatewaySettings, OrchestratorSettings

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ENV_EXAMPLE = _REPO_ROOT / ".env.example"
_REQUIRED_EXAMPLE_KEYS = (
    "GATEWAY_RATE_LIMIT_REQUESTS",
    "GATEWAY_RATE_LIMIT_WINDOW_S",
    "GATEWAY_MAX_MESSAGE_CHARS",
    "MCP_SHARED_SECRET",
    "ORCH_ROUTING_STRATEGY",
    "ORCH_RULES_MAX_QUERY_TOKENS",
    "OTEL_EXPORTER_OTLP_ENDPOINT",
    "ORCH_BREAKER_GRAPH_QUERY_FAILURE_THRESHOLD",
    "ORCH_BREAKER_GRAPH_QUERY_COOLDOWN_S",
    "ORCH_BREAKER_CODE_ANALYST_FAILURE_THRESHOLD",
    "ORCH_BREAKER_CODE_ANALYST_COOLDOWN_S",
    "ORCH_BREAKER_INDEXER_FAILURE_THRESHOLD",
    "ORCH_BREAKER_INDEXER_COOLDOWN_S",
    "ORCH_BREAKER_MEMORY_FAILURE_THRESHOLD",
    "ORCH_BREAKER_MEMORY_COOLDOWN_S",
    "INDEXER_CLONE_ALLOWED_HOSTS",
    "INDEXER_CLONE_ALLOWED_SCHEMES",
)
_SECRET_KEYS = (
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "NEO4J_PASSWORD",
    "GATEWAY_API_KEY",
    "MCP_SHARED_SECRET",
)


def test_settings_env_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / ".env").write_text(
        "GATEWAY_TOKEN_BUDGET=111\nGATEWAY_CACHE_TTL_SECONDS=222\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.development").write_text(
        "GATEWAY_TOKEN_BUDGET=333\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("core.settings._REPO_ROOT", tmp_path)
    monkeypatch.setenv("APP_ENV", "development")
    monkeypatch.delenv("GATEWAY_TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("GATEWAY_CACHE_TTL_SECONDS", raising=False)

    overlaid = GatewaySettings.from_env()
    assert overlaid.token_budget == 333
    assert overlaid.cache_ttl_seconds == 222

    monkeypatch.setenv("GATEWAY_TOKEN_BUDGET", "999")
    from_real_env = GatewaySettings.from_env()
    assert from_real_env.token_budget == 999
    assert from_real_env.cache_ttl_seconds == 222

    monkeypatch.delenv("GATEWAY_TOKEN_BUDGET", raising=False)
    monkeypatch.delenv("GATEWAY_CACHE_TTL_SECONDS", raising=False)
    (tmp_path / ".env").write_text("", encoding="utf-8")
    (tmp_path / ".env.development").write_text("", encoding="utf-8")
    defaults = GatewaySettings.from_env()
    assert defaults.token_budget == 3000
    assert defaults.cache_ttl_seconds == 300


def test_plan_plus_synthesis_reserve_fits_request_deadline() -> None:
    settings = OrchestratorSettings()
    assert settings.plan_deadline_s + settings.synthesis_reserve_s <= settings.request_deadline_s
    usable = settings.synthesis_reserve_s - settings.synthesis_safety_margin_s
    from core.llm.pricing import measured_synthesis_p95_s
    from core.settings import LLMSettings

    p95 = measured_synthesis_p95_s(LLMSettings().resolve_model("synthesis"))
    assert p95 is not None
    assert usable >= p95


def test_synthesis_reserve_covers_measured_p95() -> None:
    from core.llm.pricing import SYNTHESIS_P95_MS, measured_synthesis_p95_s
    from core.settings import LLMSettings

    settings = OrchestratorSettings()
    model = LLMSettings().resolve_model("synthesis")
    p95 = measured_synthesis_p95_s(model)
    assert p95 is not None
    assert settings.synthesis_reserve_s - settings.synthesis_safety_margin_s >= p95
    # Fail if bake-off p95 for the default synthesis model is updated without
    # raising ORCH_SYNTHESIS_RESERVE_S (or switching models).
    catalog_s = SYNTHESIS_P95_MS[model] / 1000.0
    assert catalog_s == p95


def test_sonnet_synthesis_p95_does_not_fit_default_reserve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pydantic import ValidationError

    monkeypatch.setenv("ORCH_MODEL_SYNTHESIS", "anthropic/claude-sonnet-4.5")
    with pytest.raises(ValidationError, match="measured synthesis p95"):
        OrchestratorSettings()


def test_plan_plus_synthesis_reserve_rejects_oversubscribed_budget() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="ORCH_PLAN_DEADLINE_S"):
        OrchestratorSettings(
            plan_deadline_s=40.0,
            synthesis_reserve_s=20.0,
            request_deadline_s=55.0,
        )


def test_compose_requires_neo4j_password() -> None:
    compose = (_REPO_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "changeme123" not in compose
    assert "NEO4J_AUTH:?Set NEO4J_AUTH" in compose
    assert "NEO4J_PASSWORD:?Set NEO4J_PASSWORD" in compose


def test_env_example_documents_required_settings() -> None:
    text = _ENV_EXAMPLE.read_text(encoding="utf-8")
    for key in _REQUIRED_EXAMPLE_KEYS:
        assert key in text, f"{key} missing from .env.example"


def test_committed_env_files_exclude_secret_values() -> None:
    assert _ENV_EXAMPLE.is_file(), f"missing {_ENV_EXAMPLE.name}"
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, _, value = stripped.partition("=")
        if key in _SECRET_KEYS:
            assert value.strip() == "", f"{_ENV_EXAMPLE.name} must not assign {key}"


def test_env_example_is_the_only_committed_env_file() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", ".env*"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    committed = {name for name in tracked.split("\0") if name}
    assert committed == {".env.example"}, f"unexpected committed env files: {committed}"


def test_orchestrator_breaker_overrides_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ORCH_BREAKER_GRAPH_QUERY_FAILURE_THRESHOLD", "9")
    monkeypatch.setenv("ORCH_BREAKER_GRAPH_QUERY_COOLDOWN_S", "15")
    settings = OrchestratorSettings.from_env()
    threshold, cooldown = settings.breaker_for("graph_query")
    assert threshold == 9
    assert cooldown == 15.0
