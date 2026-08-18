"""Runtime settings shared by gateway and agent adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
AppEnv = Literal["development", "testing", "production"]

DEFAULT_REPO_URL = "https://github.com/fastapi/fastapi"
DEFAULT_REPO_ROOT = "/repo"
DEFAULT_INDEX_REPORT_PATH = "/tmp/index_report.json"
DEFAULT_OPENROUTER_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GRAPH_QUERY_URL = "http://graph_query:8003/mcp"
DEFAULT_MEMORY_DB_PATH = "/data/memory.db"
DEFAULT_MEMORY_CACHE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MEMORY_TOKEN_BUDGET = 3000
DEFAULT_MEMORY_RECENT_TURNS = 6

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARED_ENV_FILE = _REPO_ROOT / ".env"
_SECRET_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "NEO4J_PASSWORD",
    "GATEWAY_API_KEY",
}


def _app_env() -> AppEnv:
    raw = os.environ.get("APP_ENV", "development").strip().lower()
    if raw in {"development", "testing", "production"}:
        return raw  # type: ignore[return-value]
    return "development"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _agent_prefix(agent: str) -> str:
    return {
        "orchestrator": "ORCH_",
        "indexer": "INDEXER_",
        "graph_query": "GQ_",
        "code_analyst": "CA_",
        "memory": "MEM_",
    }.get(agent, "")


class _FilteredDotEnvSettingsSource(DotEnvSettingsSource):
    def __call__(self) -> dict[str, Any]:
        values = super().__call__()
        return {key: value for key, value in values.items() if key not in _SECRET_ENV_KEYS}


class RepoSettings(BaseSettings):
    """Base settings with shared env-file precedence and secret filtering."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        overlay_file = _REPO_ROOT / f".env.{_app_env()}"
        shared = _FilteredDotEnvSettingsSource(settings_cls, env_file=_SHARED_ENV_FILE)
        overlay = _FilteredDotEnvSettingsSource(settings_cls, env_file=overlay_file)
        _ = dotenv_settings
        return (init_settings, env_settings, overlay, shared, file_secret_settings)


class ServiceSettings(RepoSettings):
    """Common service knobs required by the spec."""

    request_timeout_s: float = 10.0
    retry_count: int = 1
    model_name: str = DEFAULT_OPENROUTER_MODEL
    token_budget: int = 3000
    cache_ttl_seconds: int = 300


class AgentRuntimeSettings(ServiceSettings):
    """Host/port/log-level for a FastMCP streamable-HTTP server."""

    host: str = "0.0.0.0"
    port: int
    log_level: LogLevel = "INFO"
    agent: str

    @classmethod
    def from_env(cls, *, agent: str, default_port: int) -> AgentRuntimeSettings:
        return cls(  # type: ignore[call-arg]
            agent=agent,
            host=os.environ.get("MCP_HOST", "0.0.0.0"),
            port=int(os.environ.get("MCP_PORT", str(default_port))),
            log_level=cast(LogLevel, os.environ.get("LOG_LEVEL", "INFO").upper()),
            _env_prefix=_agent_prefix(agent),
        )


class GatewaySettings(ServiceSettings):
    """HTTP bind settings and MCP endpoint URLs for the gateway."""

    model_config = SettingsConfigDict(env_prefix="GATEWAY_", extra="ignore", case_sensitive=False)

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: LogLevel = Field(
        default="INFO",
        validation_alias=AliasChoices("GATEWAY_LOG_LEVEL", "LOG_LEVEL"),
    )
    orchestrator_url: str = Field(
        default="http://orchestrator:8001/mcp",
        validation_alias=AliasChoices("GATEWAY_ORCHESTRATOR_URL", "ORCHESTRATOR_MCP_URL"),
    )
    indexer_url: str = Field(
        default="http://indexer:8002/mcp",
        validation_alias=AliasChoices("GATEWAY_INDEXER_URL", "INDEXER_MCP_URL"),
    )
    graph_query_url: str = Field(
        default="http://graph_query:8003/mcp",
        validation_alias=AliasChoices("GATEWAY_GRAPH_QUERY_URL", "GRAPH_QUERY_MCP_URL"),
    )
    code_analyst_url: str = Field(
        default="http://code_analyst:8004/mcp",
        validation_alias=AliasChoices("GATEWAY_CODE_ANALYST_URL", "CODE_ANALYST_MCP_URL"),
    )
    memory_url: str = Field(
        default="http://memory:8005/mcp",
        validation_alias=AliasChoices("GATEWAY_MEMORY_URL", "MEMORY_MCP_URL"),
    )
    health_timeout_s: float = 2.0
    answer_chunk_chars: int = 160
    api_key: SecretStr | None = None

    @classmethod
    def from_env(cls) -> GatewaySettings:
        return cls()

    def agent_urls(self) -> dict[str, str]:
        return {
            "orchestrator": self.orchestrator_url,
            "indexer": self.indexer_url,
            "graph_query": self.graph_query_url,
            "code_analyst": self.code_analyst_url,
            "memory": self.memory_url,
        }


class Neo4jSettings(RepoSettings):
    """Bolt connection settings. Only indexer and graph_query may open sessions."""

    uri: str = Field(default="bolt://neo4j:7687", validation_alias=AliasChoices("NEO4J_URI", "uri"))
    user: str = Field(default="neo4j", validation_alias=AliasChoices("NEO4J_USER", "user"))
    password: SecretStr = Field(
        default=SecretStr(""),
        validation_alias=AliasChoices("NEO4J_PASSWORD", "password"),
    )

    @classmethod
    def from_env(cls) -> Neo4jSettings:
        return cls()


class LLMSettings(RepoSettings):
    """OpenRouter connection settings. Tests never construct a real client."""

    model: str = Field(
        default=DEFAULT_OPENROUTER_MODEL,
        validation_alias=AliasChoices("OPENROUTER_MODEL", "LLM_MODEL", "model"),
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "api_key"),
    )
    base_url: str = Field(
        default=DEFAULT_OPENROUTER_BASE_URL,
        validation_alias=AliasChoices("OPENROUTER_BASE_URL", "base_url"),
    )

    @classmethod
    def from_env(cls) -> LLMSettings:
        return cls()


class AnalysisSettings(ServiceSettings):
    """Read-only repo root and Graph Query MCP URL for the Code Analyst."""

    model_config = SettingsConfigDict(env_prefix="CA_", extra="ignore", case_sensitive=False)

    repo_root: str = Field(
        default=DEFAULT_REPO_ROOT,
        validation_alias=AliasChoices("REPO_ROOT", "repo_root"),
    )
    graph_query_url: str = Field(
        default=DEFAULT_GRAPH_QUERY_URL,
        validation_alias=AliasChoices("CA_GRAPH_QUERY_URL", "GRAPH_QUERY_MCP_URL"),
    )

    @classmethod
    def from_env(cls) -> AnalysisSettings:
        return cls()


class IndexingSettings(ServiceSettings):
    """Clone and crawl settings for the indexer."""

    model_config = SettingsConfigDict(env_prefix="INDEXER_", extra="ignore", case_sensitive=False)

    repo_url: str = Field(
        default=DEFAULT_REPO_URL,
        validation_alias=AliasChoices("REPO_URL", "repo_url"),
    )
    repo_root: str = Field(
        default=DEFAULT_REPO_ROOT,
        validation_alias=AliasChoices("REPO_ROOT", "repo_root"),
    )
    skip_tests: bool = Field(
        default=True,
        validation_alias=AliasChoices("INDEXER_SKIP_TESTS", "INDEX_SKIP_TESTS", "skip_tests"),
    )
    skip_docs: bool = Field(
        default=True,
        validation_alias=AliasChoices("INDEXER_SKIP_DOCS", "INDEX_SKIP_DOCS", "skip_docs"),
    )
    report_path: str = Field(
        default=DEFAULT_INDEX_REPORT_PATH,
        validation_alias=AliasChoices("INDEXER_REPORT_PATH", "INDEX_REPORT_PATH", "report_path"),
    )

    @classmethod
    def from_env(cls) -> IndexingSettings:
        return cls()


class MemorySettings(ServiceSettings):
    """SQLite-backed conversation memory settings."""

    model_config = SettingsConfigDict(env_prefix="MEM_", extra="ignore", case_sensitive=False)

    db_path: str = Field(
        default=DEFAULT_MEMORY_DB_PATH,
        validation_alias=AliasChoices("MEMORY_DB_PATH", "MEM_DB_PATH", "db_path"),
    )
    cache_ttl_seconds: int = Field(
        default=DEFAULT_MEMORY_CACHE_TTL_SECONDS,
        validation_alias=AliasChoices(
            "MEM_CACHE_TTL_SECONDS",
            "MEMORY_CACHE_TTL_SECONDS",
            "cache_ttl_seconds",
        ),
    )
    token_budget: int = Field(
        default=DEFAULT_MEMORY_TOKEN_BUDGET,
        validation_alias=AliasChoices("MEM_TOKEN_BUDGET", "MEMORY_TOKEN_BUDGET", "token_budget"),
    )
    recent_turns_to_keep: int = Field(
        default=DEFAULT_MEMORY_RECENT_TURNS,
        validation_alias=AliasChoices(
            "MEM_RECENT_TURNS",
            "MEMORY_RECENT_TURNS",
            "recent_turns_to_keep",
        ),
    )

    @classmethod
    def from_env(cls) -> MemorySettings:
        return cls()


class OrchestratorSettings(ServiceSettings):
    """Runtime settings for the orchestrator core loop."""

    model_config = SettingsConfigDict(env_prefix="ORCH_", extra="ignore", case_sensitive=False)

    graph_query_timeout_s: float = 5.0
    code_analyst_timeout_s: float = 15.0
    indexer_timeout_s: float = 30.0
    synthesis_timeout_s: float = 20.0
    rules_max_query_tokens: int = 60
    routing_strategy: Literal["rules_first", "llm_first"] = "rules_first"

    @classmethod
    def from_env(cls) -> OrchestratorSettings:
        return cls()


class GraphQuerySettings(ServiceSettings):
    """Graph Query service settings."""

    model_config = SettingsConfigDict(env_prefix="GQ_", extra="ignore", case_sensitive=False)

    @classmethod
    def from_env(cls) -> GraphQuerySettings:
        return cls()


class CodeAnalystSettings(AnalysisSettings):
    """Alias class for the Code Analyst service."""


class MemoryAgentSettings(MemorySettings):
    """Alias class for the Memory service."""
