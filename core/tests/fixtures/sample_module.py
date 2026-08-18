"""Fixture module for indexer parser tests."""

from typing import Any as Anything
import os


class Base:
    """Base class."""

    pass


def helper(value: int) -> int:
    return value


class Worker(Base):
    """Does work."""

    @staticmethod
    def run(count: int) -> int:
        helper(count)
        return count

    @property
    def label(self) -> str:
        return "worker"


async def fetch_all(limit: int) -> None:
    """Load records."""
    helper(limit)


@app.get
def ping() -> str:
    helper(0)
    return "ok"
