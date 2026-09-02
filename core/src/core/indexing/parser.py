"""Parse Python source files into structured models. Never raises on syntax errors."""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.settings import IndexingSettings

SOURCE_SUMMARY_MAX_LINES = 24


class ParsedParameter(BaseModel):
    """A parameter of a function or method."""

    name: str
    annotation: str | None = None
    default: str | None = None
    position: int


class ParsedDocstring(BaseModel):
    """A docstring: full text plus first-line summary."""

    text: str
    summary: str


class ParsedClass(BaseModel):
    """A class definition extracted from a module."""

    name: str
    qualified_name: str
    parent_qualified_name: str = ""
    bases: list[str]
    decorators: list[str] = Field(default_factory=list)
    line_start: int
    line_end: int
    docstring: ParsedDocstring | None = None
    source_summary: str = ""


class ParsedCallable(BaseModel):
    """A function or method definition."""

    name: str
    qualified_name: str
    parent_qualified_name: str = ""
    parameters: list[ParsedParameter]
    decorators: list[str]
    line_start: int
    line_end: int
    is_async: bool
    calls: list[str] = Field(default_factory=list)
    docstring: ParsedDocstring | None = None
    source_summary: str = ""


class ParsedImport(BaseModel):
    """An ``import`` or ``from ... import`` statement."""

    module: str | None
    names: list[str]
    alias: str | None = None


class ParsedCall(BaseModel):
    """A best-effort call site: textual callee name, no name resolution."""

    caller_qualified_name: str
    callee: str


class ParseError(BaseModel):
    """A file that could not be parsed."""

    path: str
    message: str


class ParsedFile(BaseModel):
    """AST extract for one Python source file."""

    path: str
    module: str
    content_hash: str
    line_start: int = 1
    line_end: int = 1
    docstring: ParsedDocstring | None = None
    classes: list[ParsedClass] = Field(default_factory=list)
    functions: list[ParsedCallable] = Field(default_factory=list)
    methods: list[ParsedCallable] = Field(default_factory=list)
    imports: list[ParsedImport] = Field(default_factory=list)
    calls: list[ParsedCall] = Field(default_factory=list)
    error: str | None = None


class ExtractedGraph(BaseModel):
    """Flat entity and relationship lists derived from a ``ParsedFile``."""

    entities: list[dict[str, Any]] = Field(default_factory=list)
    relationships: list[dict[str, Any]] = Field(default_factory=list)


def hash_bytes(data: bytes) -> str:
    """Return the SHA-256 hex digest of ``data``.
    
    Args:
        data: bytes.

    Returns:
        str.
    """
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of the file's bytes.
    
    Args:
        path: str | Path.

    Returns:
        str.
    """
    return hash_bytes(Path(path).read_bytes())


def module_name_from_path(path: Path, repo_root: Path) -> str:
    """Derive a dotted module name from ``path`` relative to ``repo_root``.
    
    Args:
        path: Path.
        repo_root: Path.

    Returns:
        str.
    """
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        rel = Path(path.name)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def parse_file(path: str | Path, repo_root: str | Path | None = None) -> ParsedFile:
    """Parse ``path`` into a ``ParsedFile``. Syntax errors are recorded, never raised.
    
    Args:
        path: str | Path.
        repo_root: str | Path | None.

    Returns:
        ParsedFile.
    """
    file_path = Path(path)
    root = Path(repo_root) if repo_root is not None else file_path.parent
    rel = _relative_posix(file_path, root)
    module = module_name_from_path(file_path, root)
    raw = file_path.read_bytes()
    return _parse_source(raw, rel=rel, module=module, filename=str(file_path))


def parse_code(source: str, *, path: str = "<string>", module: str = "<string>") -> ParsedFile:
    """Parse a source string into a ``ParsedFile``. Syntax errors are recorded, never raised.
    
    Args:
        source: str.
        path: str.
        module: str.

    Returns:
        ParsedFile.
    """
    return _parse_source(source.encode("utf-8"), rel=path, module=module, filename=path)


def parse_path_or_code(
    path_or_code: str,
    repo_root: str | Path | None = None,
) -> ParsedFile:
    """Parse ``path_or_code`` as a file path if it exists, otherwise as source.
    
    Args:
        path_or_code: str.
        repo_root: str | Path | None.

    Returns:
        ParsedFile.
    """
    settings_root = (
        Path(repo_root) if repo_root is not None else Path(IndexingSettings.from_env().repo_root)
    )
    candidates = [Path(path_or_code)]
    if not Path(path_or_code).is_absolute():
        candidates.append(settings_root / path_or_code)
    for candidate in candidates:
        try:
            if candidate.is_file():
                root = (
                    settings_root
                    if _is_relative_to(candidate, settings_root)
                    else candidate.parent
                )
                return parse_file(candidate, root)
        except OSError:
            continue
    return parse_code(path_or_code)


def parse_python_ast(
    path_or_code: str,
    repo_root: str | Path | None = None,
) -> ParsedFile:
    """Thin wrapper: parse a path or source string into ``ParsedFile``.
    
    Args:
        path_or_code: str.
        repo_root: str | Path | None.

    Returns:
        ParsedFile.
    """
    return parse_path_or_code(path_or_code, repo_root=repo_root)


def extract_entities(
    path_or_code: str,
    repo_root: str | Path | None = None,
) -> ExtractedGraph:
    """Return flat entity and relationship lists derived from a ``ParsedFile``.
    
    Args:
        path_or_code: str.
        repo_root: str | Path | None.

    Returns:
        ExtractedGraph.
    """
    return extracted_graph_from_parsed(parse_python_ast(path_or_code, repo_root=repo_root))


def extracted_graph_from_parsed(parsed: ParsedFile) -> ExtractedGraph:
    """Flatten a ``ParsedFile`` into entity and relationship dicts.
    
    Args:
        parsed: ParsedFile.

    Returns:
        ExtractedGraph.
    """
    entities: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    entities.append(
        {
            "type": "File",
            "path": parsed.path,
            "content_hash": parsed.content_hash,
        }
    )
    module_name = parsed.module.rsplit(".", 1)[-1] if parsed.module else parsed.path
    entities.append(
        {
            "type": "Module",
            "qualified_name": parsed.module,
            "name": module_name,
            "file_path": parsed.path,
            "line_start": parsed.line_start,
            "line_end": parsed.line_end,
        }
    )
    relationships.append(
        {"type": "CONTAINS", "source": parsed.path, "target": parsed.module}
    )
    _emit_docstring(entities, relationships, parsed.module, parsed.docstring)

    for cls in parsed.classes:
        entities.append(
            {
                "type": "Class",
                "qualified_name": cls.qualified_name,
                "name": cls.name,
                "file_path": parsed.path,
                "line_start": cls.line_start,
                "line_end": cls.line_end,
                "bases": list(cls.bases),
            }
        )
        relationships.append(
            {
                "type": "CONTAINS",
                "source": cls.parent_qualified_name or parsed.module,
                "target": cls.qualified_name,
            }
        )
        _emit_docstring(entities, relationships, cls.qualified_name, cls.docstring)
        _emit_decorators(entities, relationships, cls.qualified_name, cls.decorators)
        for base in cls.bases:
            relationships.append(
                {"type": "INHERITS_FROM", "source": cls.qualified_name, "target": base}
            )

    for fn in parsed.functions:
        _emit_callable(
            entities,
            relationships,
            fn,
            "Function",
            parsed.path,
            fn.parent_qualified_name or parsed.module,
        )
    for method in parsed.methods:
        parent_qn = method.parent_qualified_name or method.qualified_name.rsplit(".", 1)[0]
        _emit_callable(entities, relationships, method, "Method", parsed.path, parent_qn)

    for imported in parsed.imports:
        module = imported.module or ""
        entities.append(
            {
                "type": "Import",
                "module": imported.module,
                "names": list(imported.names),
                "alias": imported.alias,
                "file_path": parsed.path,
            }
        )
        relationships.append(
            {"type": "IMPORTS", "source": parsed.module, "target": module}
        )
        if module:
            relationships.append(
                {"type": "DEPENDS_ON", "source": parsed.module, "target": module.lstrip(".")}
            )

    for call in parsed.calls:
        relationships.append(
            {
                "type": "CALLS",
                "source": call.caller_qualified_name,
                "target": call.callee,
            }
        )

    return ExtractedGraph(entities=entities, relationships=relationships)


def _emit_callable(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    item: ParsedCallable,
    label: str,
    file_path: str,
    parent_qn: str,
) -> None:
    entities.append(
        {
            "type": label,
            "qualified_name": item.qualified_name,
            "name": item.name,
            "file_path": file_path,
            "line_start": item.line_start,
            "line_end": item.line_end,
            "is_async": item.is_async,
        }
    )
    relationships.append(
        {"type": "CONTAINS", "source": parent_qn, "target": item.qualified_name}
    )
    _emit_docstring(entities, relationships, item.qualified_name, item.docstring)
    _emit_decorators(entities, relationships, item.qualified_name, item.decorators)
    for param in item.parameters:
        entities.append(
            {
                "type": "Parameter",
                "name": param.name,
                "annotation": param.annotation,
                "default": param.default,
                "position": param.position,
                "owner_qualified_name": item.qualified_name,
                "file_path": file_path,
            }
        )
        relationships.append(
            {
                "type": "HAS_PARAMETER",
                "source": item.qualified_name,
                "target": param.name,
            }
        )


def _emit_docstring(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    owner_qualified_name: str,
    docstring: ParsedDocstring | None,
) -> None:
    if docstring is None:
        return
    entities.append(
        {
            "type": "Docstring",
            "text": docstring.text,
            "summary": docstring.summary,
            "owner_qualified_name": owner_qualified_name,
        }
    )
    relationships.append(
        {
            "type": "DOCUMENTED_BY",
            "source": owner_qualified_name,
            "target": docstring.summary,
        }
    )


def _emit_decorators(
    entities: list[dict[str, Any]],
    relationships: list[dict[str, Any]],
    owner_qualified_name: str,
    decorators: list[str],
) -> None:
    seen: set[str] = set()
    for name in decorators:
        if name not in seen:
            entities.append({"type": "Decorator", "name": name})
            seen.add(name)
        relationships.append(
            {"type": "DECORATED_BY", "source": owner_qualified_name, "target": name}
        )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _parse_source(raw: bytes, *, rel: str, module: str, filename: str) -> ParsedFile:
    content_hash = hash_bytes(raw)

    try:
        tree = ast.parse(raw, filename=filename)
    except SyntaxError as exc:
        return ParsedFile(
            path=rel,
            module=module,
            content_hash=content_hash,
            line_start=1,
            line_end=1,
            error=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive
        return ParsedFile(
            path=rel,
            module=module,
            content_hash=content_hash,
            line_start=1,
            line_end=1,
            error=str(exc),
        )

    classes: list[ParsedClass] = []
    functions: list[ParsedCallable] = []
    methods: list[ParsedCallable] = []
    imports: list[ParsedImport] = []
    calls: list[ParsedCall] = []
    try:
        source = raw.decode("utf-8")
    except UnicodeDecodeError:
        source = raw.decode("utf-8", errors="replace")
    source_lines = source.splitlines()

    _collect_imports(tree.body, imports)
    _collect_definitions(
        tree.body,
        namespace=module,
        class_qn=None,
        classes=classes,
        functions=functions,
        methods=methods,
        calls=calls,
        source_lines=source_lines,
    )

    header_start, header_end = module_header_span(tree)

    return ParsedFile(
        path=rel,
        module=module,
        content_hash=content_hash,
        line_start=header_start,
        line_end=header_end,
        docstring=_parse_docstring(tree),
        classes=classes,
        functions=functions,
        methods=methods,
        imports=imports,
        calls=calls,
    )


def module_header_span(tree: ast.Module) -> tuple[int, int]:
    """Citation span for a Module: docstring if present, else leading imports.

    Class and function spans are independent and must not use this helper.
    A Module is never cited as ``1–<file length>``.

    Args:
        tree: Parsed module AST.

    Returns:
        Inclusive ``(line_start, line_end)`` for the module header.
    """
    body = list(tree.body)
    if body:
        first = body[0]
        if isinstance(first, ast.Expr):
            value = first.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                start, end = _span(first)
                return start, end
    header_end = 1
    saw_import = False
    for stmt in body:
        if isinstance(stmt, ast.Import | ast.ImportFrom):
            _, end = _span(stmt)
            header_end = max(header_end, end)
            saw_import = True
            continue
        break
    if saw_import:
        return 1, header_end
    return 1, 1


def _relative_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def _collect_imports(body: list[ast.stmt], imports: list[ParsedImport]) -> None:
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(
                    ParsedImport(
                        module=alias.name,
                        names=[alias.name],
                        alias=alias.asname,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            names: list[str] = []
            aliases: list[str | None] = []
            for alias in node.names:
                names.append(alias.name)
                aliases.append(alias.asname)
            module = _from_module(node.module, node.level)
            unique_aliases = {item for item in aliases if item is not None}
            alias_value: str | None
            if len(names) == 1:
                alias_value = aliases[0]
            elif len(unique_aliases) == 1:
                alias_value = next(iter(unique_aliases))
            else:
                alias_value = None
            imports.append(ParsedImport(module=module, names=names, alias=alias_value))
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            _collect_imports(node.body, imports)


def _is_type_checking(test: ast.expr) -> bool:
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _from_module(module: str | None, level: int) -> str | None:
    if level:
        prefix = "." * level
        return f"{prefix}{module}" if module else prefix
    return module


SCOPELESS_BLOCKS: tuple[type[ast.stmt], ...] = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,
    ast.Match,
)


def _block_bodies(node: ast.stmt) -> list[list[ast.stmt]]:
    """Return the statement lists of a block that does not open a new scope.

    ``if``, ``try``, ``with`` and friends nest statements without changing the
    enclosing namespace, so definitions inside them belong to the same owner.
    """
    bodies: list[list[ast.stmt]] = []
    for field in ("body", "orelse", "finalbody"):
        value = getattr(node, field, None)
        if isinstance(value, list):
            bodies.append(value)
    for handler in getattr(node, "handlers", []):
        bodies.append(handler.body)
    for case in getattr(node, "cases", []):
        bodies.append(case.body)
    return bodies


def _collect_definitions(
    body: list[ast.stmt],
    *,
    namespace: str,
    class_qn: str | None,
    classes: list[ParsedClass],
    functions: list[ParsedCallable],
    methods: list[ParsedCallable],
    calls: list[ParsedCall],
    source_lines: Sequence[str],
) -> None:
    """Walk ``body``, recording definitions at every nesting depth.

    Recurses into both class and function bodies, so a class or helper defined
    inside a function is still a node. ``class_qn`` marks a *direct* class parent:
    a callable is a Method only when its immediate parent is a class, so a closure
    inside a method is a Function, not a Method.

    Blocks that do not open a scope (``if``, ``try``, ``with``, loops, ``match``)
    are walked with the namespace unchanged, so a definition guarded by
    ``if TYPE_CHECKING:`` or declared inside a ``try`` is still recorded.
    """
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parsed = _parse_callable(node, namespace, source_lines)
            if class_qn is not None:
                methods.append(parsed)
            else:
                functions.append(parsed)
            for callee in parsed.calls:
                calls.append(
                    ParsedCall(caller_qualified_name=parsed.qualified_name, callee=callee)
                )
            _collect_definitions(
                node.body,
                namespace=parsed.qualified_name,
                class_qn=None,
                classes=classes,
                functions=functions,
                methods=methods,
                calls=calls,
                source_lines=source_lines,
            )
        elif isinstance(node, ast.ClassDef):
            qn = _join_qn(namespace, node.name)
            start, end = _span(node)
            classes.append(
                ParsedClass(
                    name=node.name,
                    qualified_name=qn,
                    parent_qualified_name=namespace,
                    bases=[_as_written(base) for base in node.bases],
                    decorators=[_decorator_name(dec) for dec in node.decorator_list],
                    line_start=start,
                    line_end=end,
                    docstring=_parse_docstring(node),
                    source_summary=_source_summary(source_lines, start, end),
                )
            )
            _collect_definitions(
                node.body,
                namespace=qn,
                class_qn=qn,
                classes=classes,
                functions=functions,
                methods=methods,
                calls=calls,
                source_lines=source_lines,
            )
        elif isinstance(node, SCOPELESS_BLOCKS):
            for block in _block_bodies(node):
                _collect_definitions(
                    block,
                    namespace=namespace,
                    class_qn=class_qn,
                    classes=classes,
                    functions=functions,
                    methods=methods,
                    calls=calls,
                    source_lines=source_lines,
                )


def _parse_callable(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    namespace: str,
    source_lines: Sequence[str],
) -> ParsedCallable:
    start, end = _span(node)
    return ParsedCallable(
        name=node.name,
        qualified_name=_join_qn(namespace, node.name),
        parent_qualified_name=namespace,
        parameters=_parse_parameters(node.args),
        decorators=[_decorator_name(dec) for dec in node.decorator_list],
        line_start=start,
        line_end=end,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        calls=_collect_calls(node),
        docstring=_parse_docstring(node),
        source_summary=_source_summary(source_lines, start, end),
    )


def _source_summary(lines: Sequence[str], start: int, end: int) -> str:
    if start < 1:
        return ""
    chunk = list(lines[start - 1 : max(start, end)])
    if len(chunk) > SOURCE_SUMMARY_MAX_LINES:
        chunk = chunk[:SOURCE_SUMMARY_MAX_LINES]
    return "\n".join(chunk).strip()


def _parse_parameters(args: ast.arguments) -> list[ParsedParameter]:
    params: list[ParsedParameter] = []
    position = 0
    positional = list(args.posonlyargs) + list(args.args)
    default_offset = len(positional) - len(args.defaults)
    for index, arg in enumerate(positional):
        default: str | None = None
        if index >= default_offset:
            default = _as_written(args.defaults[index - default_offset])
        params.append(
            ParsedParameter(
                name=arg.arg,
                annotation=_annotation(arg),
                default=default,
                position=position,
            )
        )
        position += 1
    if args.vararg is not None:
        params.append(
            ParsedParameter(
                name=args.vararg.arg,
                annotation=_annotation(args.vararg),
                default=None,
                position=position,
            )
        )
        position += 1
    for index, arg in enumerate(args.kwonlyargs):
        default_node = args.kw_defaults[index]
        default = _as_written(default_node) if default_node is not None else None
        params.append(
            ParsedParameter(
                name=arg.arg,
                annotation=_annotation(arg),
                default=default,
                position=position,
            )
        )
        position += 1
    if args.kwarg is not None:
        params.append(
            ParsedParameter(
                name=args.kwarg.arg,
                annotation=_annotation(args.kwarg),
                default=None,
                position=position,
            )
        )
    return params


def _annotation(arg: ast.arg) -> str | None:
    if arg.annotation is None:
        return None
    written = _as_written(arg.annotation)
    return written or None


def _collect_calls(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    collector = _CallCollector()
    for stmt in fn_node.body:
        collector.visit(stmt)
    return collector.calls


class _CallCollector(ast.NodeVisitor):
    """Collect ``Name`` / ``Attribute`` callees for one callable's own body.

    Nested ``def``/``class`` bodies are skipped: they are collected as their own
    nodes, and each callable owns only the calls it makes directly.

    A call whose attribute chain does not bottom out in a plain name --
    ``super().__init__()``, ``make()().run()``, ``items[0].get()`` -- is skipped
    too. Recording only the trailing ``__init__`` or ``get`` would invite the
    resolver to link it to an unrelated same-named callable.
    """

    def __init__(self) -> None:
        """Create a visitor that records callee names."""
        self.calls: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Skip a nested function body; it is collected as its own node.
        
        Args:
            node: ast.FunctionDef.
        """
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Skip a nested async function body; it is collected as its own node.
        
        Args:
            node: ast.AsyncFunctionDef.
        """
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit classdef.
        
        Args:
            node: ast.ClassDef.
        """
        return

    def visit_Call(self, node: ast.Call) -> None:
        """Visit call.
        
        Args:
            node: ast.Call.
        """
        if _chain_root_is_name(node.func):
            name = _callee_name(node.func)
            if name:
                self.calls.append(name)
        self.generic_visit(node)


def _chain_root_is_name(func: ast.expr) -> bool:
    """True when ``func`` is a name or an attribute chain rooted in a name."""
    current = func
    while isinstance(current, ast.Attribute):
        current = current.value
    return isinstance(current, ast.Name)


def _callee_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = []
        current: ast.expr = func
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
            return ".".join(reversed(parts))
        return ".".join(reversed(parts)) if parts else None
    return None


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    dotted = _callee_name(node)
    if dotted:
        return dotted
    return _as_written(node)


def _as_written(node: ast.expr) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return getattr(node, "id", "") or ""


def _span(node: ast.AST) -> tuple[int, int]:
    start = int(getattr(node, "lineno", 1))
    decorator_list = getattr(node, "decorator_list", [])
    if decorator_list:
        start = min(start, min(int(dec.lineno) for dec in decorator_list))
    end = getattr(node, "end_lineno", None)
    return start, int(end) if end is not None else start


def _parse_docstring(
    node: ast.AsyncFunctionDef | ast.FunctionDef | ast.ClassDef | ast.Module,
) -> ParsedDocstring | None:
    doc = ast.get_docstring(node)
    if not doc:
        return None
    summary = doc.split("\n", 1)[0].strip()
    if not summary:
        return None
    return ParsedDocstring(text=doc, summary=summary)


def _join_qn(namespace: str, name: str) -> str:
    return f"{namespace}.{name}" if namespace else name
