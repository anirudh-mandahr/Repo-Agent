"""Logging always includes correlation_id."""

from __future__ import annotations

from core.logging import bind_correlation_id, configure_logging, get_correlation_id, get_logger


def test_correlation_id_defaults_and_binds() -> None:
    configure_logging("INFO")
    assert get_correlation_id() == "-"
    bound = bind_correlation_id("abc-123")
    assert bound == "abc-123"
    assert get_correlation_id() == "abc-123"
    log = get_logger("test")
    log.info("ping")
