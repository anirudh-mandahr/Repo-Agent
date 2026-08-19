"""Day-2 smoke wrapper. Prefers the compose network because MCP ports are internal."""

from __future__ import annotations

import os
import subprocess


def main() -> None:
    if os.environ.get("SMOKE_IN_CONTAINER") == "1":
        from core.mcp.smoke import main as smoke_main

        smoke_main()
        return
    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",
        "-e",
        "SMOKE_IN_CONTAINER=1",
        "-e",
        "SMOKE_GRAPH_QUERY_URL=http://graph_query:8003/mcp",
        "-e",
        "SMOKE_CODE_ANALYST_URL=http://code_analyst:8004/mcp",
        "gateway",
        "python",
        "-m",
        "core.mcp.smoke",
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
