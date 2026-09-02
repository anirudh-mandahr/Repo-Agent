"""Runtime settings shared by gateway and agent adapters."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Self, cast

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
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
DEFAULT_SYNTHESIS_MODEL = "openai/gpt-4.1-mini"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_GRAPH_QUERY_URL = "http://graph_query:8003/mcp"
DEFAULT_MEMORY_DB_PATH = "/data/memory.db"
DEFAULT_MEMORY_CACHE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_MEMORY_TOKEN_BUDGET = 3000
DEFAULT_MEMORY_RECENT_TURNS = 6
DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
DEFAULT_EMBEDDING_DIMENSIONS = 256
DEFAULT_EMBEDDING_BATCH_SIZE = 96
# Neo4j normalizes cosine to (1 + cos) / 2, so 0.5 is orthogonal. Fitted to the
# hash backend, whose unrelated text scores exactly 0.5.
DEFAULT_EMBEDDING_MIN_SCORE = 0.6

_REPO_ROOT = Path(__file__).resolve().parents[3]
if TYPE_CHECKING:
    from core.llm.pricing import ModelRates

_SECRET_ENV_KEYS = {
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "NEO4J_PASSWORD",
    "GATEWAY_API_KEY",
    "MCP_SHARED_SECRET",
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
        """Call  .
        
        Returns:
            dict[str, Any].
        """
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
        """Settings customise sources.
        
        Args:
            settings_cls: type[BaseSettings].
            init_settings: PydanticBaseSettingsSource.
            env_settings: PydanticBaseSettingsSource.
            dotenv_settings: PydanticBaseSettingsSource.
            file_secret_settings: PydanticBaseSettingsSource.

        Returns:
            tuple[PydanticBaseSettingsSource, ...].
        """
        overlay_file = _REPO_ROOT / f".env.{_app_env()}"
        shared_file = _REPO_ROOT / ".env"
        shared = _FilteredDotEnvSettingsSource(settings_cls, env_file=shared_file)
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
    breaker_failure_threshold: int = 3
    breaker_cooldown_s: float = 30.0


class AgentRuntimeSettings(ServiceSettings):
    """Host/port/log-level for a FastMCP streamable-HTTP server."""

    host: str = "0.0.0.0"
    port: int
    log_level: LogLevel = "INFO"
    agent: str

    @classmethod
    def from_env(cls, *, agent: str, default_port: int) -> AgentRuntimeSettings:
        """From env.
        
        Args:
            agent: str.
            default_port: int.

        Returns:
            AgentRuntimeSettings.
        """
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
    chat_timeout_s: float = 90.0
    index_poll_interval_s: float = Field(
        default=2.0,
        description="Gap between get_index_status polls while an index job runs.",
        validation_alias=AliasChoices("GATEWAY_INDEX_POLL_INTERVAL_S", "index_poll_interval_s"),
    )
    index_start_grace_s: float = Field(
        default=30.0,
        description=(
            "How long a dispatched index may take to report running before the "
            "job is treated as never started."
        ),
        validation_alias=AliasChoices("GATEWAY_INDEX_START_GRACE_S", "index_start_grace_s"),
    )
    index_timeout_s: float = Field(
        default=3600.0,
        description=(
            "Ceiling on one index job. Separate from request_timeout_s, which "
            "bounds a single MCP round trip and is far shorter than an index."
        ),
        validation_alias=AliasChoices("GATEWAY_INDEX_TIMEOUT_S", "index_timeout_s"),
    )
    api_key: SecretStr | None = None
    rate_limit_requests: int = 60
    rate_limit_window_s: float = 60.0
    max_message_chars: int = 8000

    @classmethod
    def from_env(cls) -> GatewaySettings:
        """From env.
        
        Returns:
            GatewaySettings.
        """
        return cls()

    def agent_urls(self) -> dict[str, str]:
        """Agent urls.
        
        Returns:
            dict[str, str].
        """
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
        """From env.
        
        Returns:
            Neo4jSettings.
        """
        return cls()


class LLMSettings(RepoSettings):
    """OpenRouter connection settings. Tests never construct a real client."""

    model: str = Field(
        default=DEFAULT_OPENROUTER_MODEL,
        validation_alias=AliasChoices("OPENROUTER_MODEL", "LLM_MODEL", "model"),
    )
    routing_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ORCH_MODEL_ROUTING", "routing_model"),
    )
    synthesis_model: str | None = Field(
        default=DEFAULT_SYNTHESIS_MODEL,
        validation_alias=AliasChoices("ORCH_MODEL_SYNTHESIS", "synthesis_model"),
    )
    analysis_model: str | None = Field(
        default=None,
        validation_alias=AliasChoices("CA_MODEL_ANALYSIS", "analysis_model"),
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "api_key"),
    )
    base_url: str = Field(
        default=DEFAULT_OPENROUTER_BASE_URL,
        validation_alias=AliasChoices("OPENROUTER_BASE_URL", "base_url"),
    )
    prompt_usd_per_million: float = Field(
        default=3.0,
        validation_alias=AliasChoices("LLM_PROMPT_USD_PER_MILLION", "prompt_usd_per_million"),
    )
    completion_usd_per_million: float = Field(
        default=15.0,
        validation_alias=AliasChoices(
            "LLM_COMPLETION_USD_PER_MILLION",
            "completion_usd_per_million",
        ),
    )
    cached_prompt_usd_per_million: float = Field(
        default=0.30,
        validation_alias=AliasChoices(
            "LLM_CACHED_PROMPT_USD_PER_MILLION",
            "cached_prompt_usd_per_million",
        ),
    )
    prices_json: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_MODEL_PRICES_JSON", "LLM_MODEL_PRICES"),
    )
    prompt_cache: bool = Field(
        default=True,
        validation_alias=AliasChoices("LLM_PROMPT_CACHE", "prompt_cache"),
    )

    @field_validator("routing_model", "synthesis_model", "analysis_model", mode="before")
    @classmethod
    def _blank_model_override(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @classmethod
    def from_env(cls) -> LLMSettings:
        """From env.

        Returns:
            LLMSettings.
        """
        return cls()

    def resolve_model(self, purpose: str) -> str:
        """Return the model id for ``purpose``, falling back to ``OPENROUTER_MODEL``.

        Args:
            purpose: ``routing``, ``synthesis``, ``analysis``, or ``summarization``.

        Returns:
            OpenRouter model id.
        """
        overrides = {
            "routing": self.routing_model,
            "synthesis": self.synthesis_model,
            "analysis": self.analysis_model,
        }
        override = overrides.get(purpose)
        if override:
            return override
        return self.model

    def rates_for(self, model: str) -> ModelRates:
        """Return per-1M rates for ``model``.

        Args:
            model: OpenRouter model id.

        Returns:
            Overlay JSON, then the built-in catalog, then global defaults.
        """
        from core.llm.pricing import ModelRates, parse_prices_json, resolve_rates

        fallback = ModelRates(
            prompt_usd_per_million=self.prompt_usd_per_million,
            completion_usd_per_million=self.completion_usd_per_million,
            cached_prompt_usd_per_million=self.cached_prompt_usd_per_million,
        )
        return resolve_rates(
            model,
            overlay=parse_prices_json(self.prices_json),
            fallback=fallback,
        )


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
        """From env.
        
        Returns:
            AnalysisSettings.
        """
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
        default=False,
        validation_alias=AliasChoices("INDEXER_SKIP_TESTS", "INDEX_SKIP_TESTS", "skip_tests"),
    )
    skip_docs: bool = Field(
        default=False,
        validation_alias=AliasChoices("INDEXER_SKIP_DOCS", "INDEX_SKIP_DOCS", "skip_docs"),
    )
    report_path: str = Field(
        default=DEFAULT_INDEX_REPORT_PATH,
        validation_alias=AliasChoices("INDEXER_REPORT_PATH", "INDEX_REPORT_PATH", "report_path"),
    )
    clone_allowed_hosts: str = Field(
        default="github.com",
        validation_alias=AliasChoices(
            "INDEXER_CLONE_ALLOWED_HOSTS",
            "CLONE_ALLOWED_HOSTS",
            "clone_allowed_hosts",
        ),
    )
    clone_allowed_schemes: str = Field(
        default="https",
        validation_alias=AliasChoices(
            "INDEXER_CLONE_ALLOWED_SCHEMES",
            "CLONE_ALLOWED_SCHEMES",
            "clone_allowed_schemes",
        ),
    )

    @classmethod
    def from_env(cls) -> IndexingSettings:
        """From env.
        
        Returns:
            IndexingSettings.
        """
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
        """From env.
        
        Returns:
            MemorySettings.
        """
        return cls()


class OrchestratorSettings(ServiceSettings):
    """Runtime settings for the orchestrator core loop."""

    model_config = SettingsConfigDict(env_prefix="ORCH_", extra="ignore", case_sensitive=False)

    graph_query_timeout_s: float = 5.0
    code_analyst_timeout_s: float = 15.0
    indexer_timeout_s: float = 30.0
    synthesis_timeout_s: float = 20.0
    synthesis_prompt_token_budget: int = 12000
    rules_max_query_tokens: int = 60
    routing_strategy: Literal["rules_first", "llm_first"] = "rules_first"
    max_plan_iterations: int = 2
    plan_deadline_s: float = 35.0
    request_deadline_s: float = 55.0
    request_token_budget: int = 50_000
    request_cost_usd_max: float = 0.15
    request_budgets_enabled: bool = True
    synthesis_safety_margin_s: float = 2.0
    synthesis_min_timeout_s: float = 0.0
    synthesis_reserve_s: float = 20.0
    prompt_truncation_order: Literal["snippets_then_lists", "lists_then_snippets"] = (
        "snippets_then_lists"
    )
    breaker_graph_query_failure_threshold: int | None = None
    breaker_graph_query_cooldown_s: float | None = None
    breaker_code_analyst_failure_threshold: int | None = None
    breaker_code_analyst_cooldown_s: float | None = None
    breaker_indexer_failure_threshold: int | None = None
    breaker_indexer_cooldown_s: float | None = None
    breaker_memory_failure_threshold: int | None = None
    breaker_memory_cooldown_s: float | None = None

    @model_validator(mode="after")
    def _plan_plus_synthesis_reserve_fits_request(self) -> Self:
        """Fail loudly when the plan phase can legally starve synthesis.

        Returns:
            This settings instance.

        Raises:
            ValueError: When ``plan_deadline_s + synthesis_reserve_s`` exceeds
                ``request_deadline_s``.
        """
        allocated = self.plan_deadline_s + self.synthesis_reserve_s
        if allocated > self.request_deadline_s:
            raise ValueError(
                "ORCH_PLAN_DEADLINE_S + ORCH_SYNTHESIS_RESERVE_S must be "
                f"<= ORCH_REQUEST_DEADLINE_S ({self.plan_deadline_s} + "
                f"{self.synthesis_reserve_s} > {self.request_deadline_s})"
            )
        self._assert_reserve_covers_synthesis_p95()
        return self

    def _assert_reserve_covers_synthesis_p95(self) -> None:
        """Fail when a full plan leaves less synthesis time than the bake-off p95.

        Skipped when request budgets are disabled or the synthesis reserve is
        zero (test clocks). Uses the resolved ``ORCH_MODEL_SYNTHESIS`` model.
        """
        if not self.request_budgets_enabled or self.synthesis_reserve_s <= 0:
            return
        from core.llm.pricing import measured_synthesis_p95_s

        model = LLMSettings().resolve_model("synthesis")
        p95 = measured_synthesis_p95_s(model)
        if p95 is None:
            return
        usable = self.synthesis_reserve_s - self.synthesis_safety_margin_s
        if usable < p95:
            raise ValueError(
                "ORCH_SYNTHESIS_RESERVE_S - ORCH_SYNTHESIS_SAFETY_MARGIN_S must be "
                f">= measured synthesis p95 for {model} ({self.synthesis_reserve_s} - "
                f"{self.synthesis_safety_margin_s} < {p95:.3f}s)"
            )

    @classmethod
    def from_env(cls) -> OrchestratorSettings:
        """From env.

        Returns:
            OrchestratorSettings.
        """
        return cls()

    def breaker_for(self, agent: str) -> tuple[int, float]:
        """Return ``(failure_threshold, cooldown_s)`` for ``agent``.

        Args:
            agent: Upstream specialist name.

        Returns:
            Per-agent circuit breaker thresholds, falling back to the shared defaults.
        """
        overrides: dict[str, tuple[int | None, float | None]] = {
            "graph_query": (
                self.breaker_graph_query_failure_threshold,
                self.breaker_graph_query_cooldown_s,
            ),
            "code_analyst": (
                self.breaker_code_analyst_failure_threshold,
                self.breaker_code_analyst_cooldown_s,
            ),
            "indexer": (
                self.breaker_indexer_failure_threshold,
                self.breaker_indexer_cooldown_s,
            ),
            "memory": (
                self.breaker_memory_failure_threshold,
                self.breaker_memory_cooldown_s,
            ),
        }
        threshold, cooldown = overrides.get(agent, (None, None))
        return (
            threshold if threshold is not None else self.breaker_failure_threshold,
            cooldown if cooldown is not None else self.breaker_cooldown_s,
        )


class EmbeddingSettings(RepoSettings):
    """Which vector backend produces embeddings, and how it is called.

    Credentials default to the OpenRouter values the LLM stack already uses, so
    enabling a real model needs no new secret. ``backend`` stays ``hash`` unless
    set explicitly: an API key being present must not silently turn indexing
    into a paid, network-dependent operation.
    """

    backend: Literal["hash", "openrouter"] = Field(
        default="hash",
        description=(
            "'hash' is the offline lexical fallback; 'openrouter' calls a real "
            "embedding model. Explicit opt-in -- an API key alone does not switch it."
        ),
        validation_alias=AliasChoices("EMBEDDING_BACKEND", "backend"),
    )
    model: str = Field(
        default=DEFAULT_EMBEDDING_MODEL,
        validation_alias=AliasChoices("EMBEDDING_MODEL", "model"),
    )
    dimensions: int = Field(
        default=DEFAULT_EMBEDDING_DIMENSIONS,
        description=(
            "Must equal the dimension the Neo4j vector indexes were created "
            "with. Changing it requires dropping and recreating those indexes."
        ),
        validation_alias=AliasChoices("EMBEDDING_DIMENSIONS", "dimensions"),
    )
    batch_size: int = Field(
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        validation_alias=AliasChoices("EMBEDDING_BATCH_SIZE", "batch_size"),
    )
    max_attempts: int = Field(
        default=4,
        description="Attempts per batch, including the first. Backoff is exponential.",
        validation_alias=AliasChoices("EMBEDDING_MAX_ATTEMPTS", "max_attempts"),
    )
    timeout_s: float = Field(
        default=30.0,
        validation_alias=AliasChoices("EMBEDDING_TIMEOUT_S", "timeout_s"),
    )
    api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "OPENROUTER_API_KEY", "api_key"),
    )
    base_url: str = Field(
        default=DEFAULT_OPENROUTER_BASE_URL,
        validation_alias=AliasChoices("EMBEDDING_BASE_URL", "OPENROUTER_BASE_URL", "base_url"),
    )
    usd_per_million: float = Field(
        default=0.02,
        description="Reporting only; used to log the cost of an index pass.",
        validation_alias=AliasChoices("EMBEDDING_USD_PER_MILLION", "usd_per_million"),
    )
    min_score: float | None = Field(
        default=None,
        description=(
            "Vector-index score floor. Backend-specific when unset, because "
            "the two backends do not produce comparable score distributions."
        ),
        validation_alias=AliasChoices("EMBEDDING_MIN_SCORE", "min_score"),
    )

    @field_validator("min_score", mode="before")
    @classmethod
    def _blank_min_score(cls, value: object) -> object:
        # Compose passes an unset optional through as "", which is not a float.
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("dimensions", "batch_size", "max_attempts")
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be at least 1")
        return value

    @classmethod
    def from_env(cls) -> EmbeddingSettings:
        """From env.

        Returns:
            EmbeddingSettings.
        """
        return cls()

    def resolve_min_score(self) -> float:
        """Return the vector-index score floor for the configured backend.

        Neo4j reports cosine as ``(1 + cos) / 2``, so 0.5 is orthogonal, not 0.
        The hash backend gives unrelated text a raw cosine of exactly 0 -- no
        shared tokens -- which lands on 0.5, so 0.6 cleanly means "some
        overlap". A real model has no such floor: unrelated code text still
        scores around 0.24 raw, or 0.62 normalized, which would sail past 0.6
        and make the tier return matches for everything.

        Returns:
            Explicit ``EMBEDDING_MIN_SCORE`` when set, else the backend default.
        """
        if self.min_score is not None:
            return self.min_score
        return 0.7 if self.backend == "openrouter" else DEFAULT_EMBEDDING_MIN_SCORE


class GraphQuerySettings(ServiceSettings):
    """Graph Query service settings."""

    model_config = SettingsConfigDict(env_prefix="GQ_", extra="ignore", case_sensitive=False)

    embeddings_enabled: bool = Field(
        default=False,
        description=(
            "Enable the hash-based lexical fallback (third find_entity tier). "
            "Default off. Injecting an EmbeddingProvider does not turn this on."
        ),
        validation_alias=AliasChoices(
            "GQ_EMBEDDINGS_ENABLED",
            "EMBEDDINGS_ENABLED",
            "embeddings_enabled",
        ),
    )

    @classmethod
    def from_env(cls) -> GraphQuerySettings:
        """From env.
        
        Returns:
            GraphQuerySettings.
        """
        return cls()


class CodeAnalystSettings(AnalysisSettings):
    """Alias class for the Code Analyst service."""


class MemoryAgentSettings(MemorySettings):
    """Alias class for the Memory service."""
