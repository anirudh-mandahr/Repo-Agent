"""MCP transport helpers shared by the orchestrator and gateway."""

from .auth import compare_secrets, mcp_request_headers, mcp_shared_secret
from .client import PooledAgentClient, open_streamable_http_session, parse_call_tool_result
from .context import bind_mcp_context
from .server import run_agent_mcp

__all__ = [
    "PooledAgentClient",
    "bind_mcp_context",
    "compare_secrets",
    "mcp_request_headers",
    "mcp_shared_secret",
    "open_streamable_http_session",
    "parse_call_tool_result",
    "run_agent_mcp",
]
