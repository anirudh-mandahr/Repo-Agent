"""Parse Python source into an AST. Agent packages must not call ast directly."""

from __future__ import annotations

import ast


def parse_python_source(source: str, *, filename: str = "<unknown>") -> ast.Module:
    """Parse Python source and return the module AST.
    
    Args:
        source: str.
        filename: str.

    Returns:
        ast.Module.

    Raises:
        TypeError: See exception message.
    """
    tree = ast.parse(source, filename=filename)
    if not isinstance(tree, ast.Module):
        raise TypeError(f"expected ast.Module, got {type(tree)!r}")
    return tree
