"""Orchestrator-facing MCP client that maps pooled tool calls onto specialist APIs."""

from __future__ import annotations

from typing import Any

from core.mcp.client import PooledAgentClient
from core.memory import ConversationContext
from core.resilience.session_pool import AgentSessionPool
from core.settings import GatewaySettings, OrchestratorSettings


class PooledOrchestratorClients:
    """Adapt :class:`PooledAgentClient` instances to the orchestrator client protocol.

    A single object exposes ``memory``, ``graph_query``, ``code_analyst``, and
    ``indexer`` attributes while forwarding each method to the matching MCP tool.
    """

    def __init__(
        self,
        pool: AgentSessionPool,
        *,
        correlation_id: str,
        settings: OrchestratorSettings | None = None,
    ) -> None:
        """Bind pooled sessions for one ``handle_query`` request.

        Args:
            pool: Shared MCP session pool.
            correlation_id: Request id forwarded on every tool call.
            settings: Per-agent timeouts; loaded from env when omitted.
        """
        orch = settings or OrchestratorSettings.from_env()
        self._correlation_id = correlation_id
        self._graph = PooledAgentClient(
            pool,
            "graph_query",
            timeout_s=orch.graph_query_timeout_s,
            retry_count=orch.retry_count,
        )
        self._code = PooledAgentClient(
            pool,
            "code_analyst",
            timeout_s=orch.code_analyst_timeout_s,
            retry_count=orch.retry_count,
        )
        self._indexer = PooledAgentClient(
            pool,
            "indexer",
            timeout_s=orch.indexer_timeout_s,
            retry_count=orch.retry_count,
        )
        self._memory = PooledAgentClient(
            pool,
            "memory",
            timeout_s=orch.request_timeout_s,
            retry_count=orch.retry_count,
        )
        self.memory = self
        self.graph_query = self
        self.code_analyst = self
        self.indexer = self

    @classmethod
    def from_pool(
        cls,
        pool: AgentSessionPool,
        *,
        correlation_id: str,
        settings: OrchestratorSettings | None = None,
    ) -> PooledOrchestratorClients:
        """Build clients from an existing pool.

        Args:
            pool: Shared MCP session pool.
            correlation_id: Request id.
            settings: Optional orchestrator settings.

        Returns:
            Wired specialist clients.
        """
        return cls(pool, correlation_id=correlation_id, settings=settings)

    async def get_context(
        self,
        session_id: str,
        token_budget: int = 3000,
    ) -> ConversationContext:
        """Load rolling summary plus recent turns from Memory.

        Args:
            session_id: Conversation id.
            token_budget: Maximum tokens of context to return.

        Returns:
            Conversation context model.
        """
        result = await self._memory.call(
            "get_context",
            {"session_id": session_id, "token_budget": token_budget},
            correlation_id=self._correlation_id,
        )
        return ConversationContext.model_validate(result)

    async def get_cached_response(self, cache_key: str) -> Any | None:
        """Return a cached orchestrator payload when present.

        Args:
            cache_key: Caller-computed cache key.

        Returns:
            Cached payload or ``None``.
        """
        return await self._memory.call(
            "get_cached_response",
            {"cache_key": cache_key},
            correlation_id=self._correlation_id,
        )

    async def cache_response(self, cache_key: str, response_json: Any) -> None:
        """Store an orchestrator payload in Memory.

        Args:
            cache_key: Caller-computed cache key.
            response_json: Opaque JSON-serializable payload.
        """
        await self._memory.call(
            "cache_response",
            {"cache_key": cache_key, "response_json": response_json},
            correlation_id=self._correlation_id,
        )

    async def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Append one conversation turn.

        Args:
            session_id: Conversation id.
            role: ``user`` or ``assistant``.
            content: Turn text.
        """
        await self._memory.call(
            "append_turn",
            {"session_id": session_id, "role": role, "content": content},
            correlation_id=self._correlation_id,
        )

    async def get_statistics(self) -> Any:
        """Return graph statistics including ``index_version``.

        Returns:
            Statistics payload.
        """
        return await self._graph.call(
            "get_statistics",
            {},
            correlation_id=self._correlation_id,
        )

    async def find_entity(self, name: str, entity_type: str | None = None) -> Any:
        """Look up an entity by name.

        Args:
            name: Entity name or phrase.
            entity_type: Optional label filter.

        Returns:
            Match payload.
        """
        return await self._graph.call(
            "find_entity",
            {"name": name, "entity_type": entity_type},
            correlation_id=self._correlation_id,
        )

    async def get_dependencies(self, name: str) -> Any:
        """Return outgoing neighbors for ``name``.

        Args:
            name: Qualified name.

        Returns:
            Neighbor payload.
        """
        return await self._graph.call(
            "get_dependencies",
            {"name": name},
            correlation_id=self._correlation_id,
        )

    async def get_dependents(self, name: str) -> Any:
        """Return incoming neighbors for ``name``.

        Args:
            name: Qualified name.

        Returns:
            Neighbor payload.
        """
        return await self._graph.call(
            "get_dependents",
            {"name": name},
            correlation_id=self._correlation_id,
        )

    async def find_related(self, name: str, relationship_type: str) -> Any:
        """Return neighbors along one relationship type.

        Args:
            name: Qualified name.
            relationship_type: Graph relationship type.

        Returns:
            Related-entity payload.
        """
        return await self._graph.call(
            "find_related",
            {"name": name, "relationship_type": relationship_type},
            correlation_id=self._correlation_id,
        )

    async def trace_imports(self, module: str, depth: int = 5) -> Any:
        """Follow import chains from ``module``.

        Args:
            module: Module name.
            depth: Traversal cap.

        Returns:
            Import-trace payload.
        """
        return await self._graph.call(
            "trace_imports",
            {"module": module, "depth": depth},
            correlation_id=self._correlation_id,
        )

    async def get_code_snippet(
        self,
        *,
        qualified_name: str | None = None,
        file_path: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> Any:
        """Fetch a numbered source snippet.

        Args:
            qualified_name: Optional graph coordinate.
            file_path: Optional repo-relative path.
            line_start: Inclusive start line.
            line_end: Inclusive end line.

        Returns:
            Snippet payload.
        """
        return await self._code.call(
            "get_code_snippet",
            {
                "qualified_name": qualified_name,
                "file_path": file_path,
                "line_start": line_start,
                "line_end": line_end,
            },
            correlation_id=self._correlation_id,
        )

    async def explain_implementation(self, qualified_name: str) -> Any:
        """Explain how a function or class is implemented.

        Args:
            qualified_name: Graph coordinate.

        Returns:
            Explanation payload.
        """
        return await self._code.call(
            "explain_implementation",
            {"qualified_name": qualified_name},
            correlation_id=self._correlation_id,
        )

    async def analyze_function(self, qualified_name: str) -> Any:
        """Analyze a function or method.

        Args:
            qualified_name: Graph coordinate.

        Returns:
            Analysis payload.
        """
        return await self._code.call(
            "analyze_function",
            {"qualified_name": qualified_name},
            correlation_id=self._correlation_id,
        )

    async def analyze_class(self, qualified_name: str) -> Any:
        """Analyze a class.

        Args:
            qualified_name: Graph coordinate.

        Returns:
            Analysis payload.
        """
        return await self._code.call(
            "analyze_class",
            {"qualified_name": qualified_name},
            correlation_id=self._correlation_id,
        )

    async def compare_implementations(self, name_a: str, name_b: str) -> Any:
        """Compare two implementations.

        Args:
            name_a: First qualified name.
            name_b: Second qualified name.

        Returns:
            Comparison payload.
        """
        return await self._code.call(
            "compare_implementations",
            {"name_a": name_a, "name_b": name_b},
            correlation_id=self._correlation_id,
        )

    async def find_patterns(
        self,
        pattern: str,
        path_prefix: str | None = None,
    ) -> Any:
        """Find supported code patterns.

        Args:
            pattern: Pattern name.
            path_prefix: Optional module or file-path prefix for decorator scoping.

        Returns:
            Pattern payload.
        """
        return await self._code.call(
            "find_patterns",
            {"pattern": pattern, "path_prefix": path_prefix},
            correlation_id=self._correlation_id,
        )

    async def index_repository(self, repo_url: str | None = None) -> Any:
        """Trigger a repository index.

        Args:
            repo_url: Optional clone URL override.

        Returns:
            Index report payload.
        """
        return await self._indexer.call(
            "index_repository",
            {"repo_url": repo_url},
            correlation_id=self._correlation_id,
        )


def build_orchestrator_pool() -> AgentSessionPool:
    """Build the orchestrator's session pool for the four specialist agents.

    Returns:
        Pool covering indexer, graph_query, code_analyst, and memory.
    """
    from time import monotonic

    from core.resilience.circuit_breaker import CircuitBreakerRegistry

    urls = GatewaySettings.from_env().agent_urls()
    urls.pop("orchestrator", None)
    settings = OrchestratorSettings.from_env()
    breakers = CircuitBreakerRegistry.from_orchestrator_settings(settings, clock=monotonic)
    return AgentSessionPool(urls, breakers=breakers, clock=monotonic)
