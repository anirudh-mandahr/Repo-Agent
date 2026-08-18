"""Run the FastAPI gateway with uvicorn."""

from __future__ import annotations

import uvicorn

from gateway.app import app, settings


def main() -> None:
    """Bind the gateway HTTP server."""
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
