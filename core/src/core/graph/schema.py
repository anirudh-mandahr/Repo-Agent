"""Idempotent Neo4j schema for the FastAPI repository knowledge graph.

Node labels
-----------
- ``:File`` — a source file on disk.
- ``:Module`` — a Python module.
- ``:Class`` — a class definition.
- ``:Function`` — a module-level function.
- ``:Method`` — a function defined on a class.
- ``:Parameter`` — a parameter of a function or method.
- ``:Decorator`` — a decorator name, shared across uses.
- ``:Import`` — an import statement.
- ``:Docstring`` — documentation attached to a code entity.

Node property conventions
-------------------------
Every code node (``Module``, ``Class``, ``Function``, ``Method``) has:

- ``qualified_name`` (unique per label)
- ``name``
- ``file_path``
- ``line_start``
- ``line_end``

``File`` nodes have ``path`` (unique) and ``content_hash``.

Relationship types
------------------
- ``CONTAINS`` — ``(:File)-[:CONTAINS]->(:Module)-[:CONTAINS]->(:Class|:Function)``,
  ``(:Class)-[:CONTAINS]->(:Method)``
- ``IMPORTS`` — ``(:Module)-[:IMPORTS]->(:Import)``
- ``INHERITS_FROM`` — ``(:Class)-[:INHERITS_FROM]->(:Class)``
- ``CALLS`` — ``(:Function|:Method)-[:CALLS]->(:Function|:Method)``
- ``DECORATED_BY`` — ``(:Function|:Method|:Class)-[:DECORATED_BY]->(:Decorator)``
- ``HAS_PARAMETER`` — ``(:Function|:Method)-[:HAS_PARAMETER]->(:Parameter)``
- ``DOCUMENTED_BY`` — ``(:Module|:Class|:Function|:Method)-[:DOCUMENTED_BY]->(:Docstring)``
- ``DEPENDS_ON`` — ``(:Import)-[:DEPENDS_ON]->(:Module)``, ``(:Module)-[:DEPENDS_ON]->(:Module)``
"""

from __future__ import annotations

from neo4j import ManagedTransaction

from core.graph.client import GraphClient
from core.logging import get_logger

log = get_logger(__name__)

LABEL_FILE = "File"
LABEL_MODULE = "Module"
LABEL_CLASS = "Class"
LABEL_FUNCTION = "Function"
LABEL_METHOD = "Method"
LABEL_PARAMETER = "Parameter"
LABEL_DECORATOR = "Decorator"
LABEL_IMPORT = "Import"
LABEL_DOCSTRING = "Docstring"
LABEL_META = "Meta"

NODE_LABELS: tuple[str, ...] = (
    LABEL_MODULE,
    LABEL_CLASS,
    LABEL_FUNCTION,
    LABEL_METHOD,
    LABEL_PARAMETER,
    LABEL_DECORATOR,
    LABEL_IMPORT,
    LABEL_DOCSTRING,
    LABEL_FILE,
    LABEL_META,
)

FIND_ENTITY_LABELS: tuple[str, ...] = (
    LABEL_MODULE,
    LABEL_CLASS,
    LABEL_FUNCTION,
    LABEL_METHOD,
    LABEL_FILE,
)

REL_CONTAINS = "CONTAINS"
REL_IMPORTS = "IMPORTS"
REL_INHERITS_FROM = "INHERITS_FROM"
REL_CALLS = "CALLS"
REL_DECORATED_BY = "DECORATED_BY"
REL_HAS_PARAMETER = "HAS_PARAMETER"
REL_DOCUMENTED_BY = "DOCUMENTED_BY"
REL_DEPENDS_ON = "DEPENDS_ON"

RELATIONSHIP_TYPES: tuple[str, ...] = (
    REL_CONTAINS,
    REL_IMPORTS,
    REL_INHERITS_FROM,
    REL_CALLS,
    REL_DECORATED_BY,
    REL_HAS_PARAMETER,
    REL_DOCUMENTED_BY,
    REL_DEPENDS_ON,
)

CONSTRAINT_STATEMENTS: tuple[str, ...] = (
    (
        "CREATE CONSTRAINT module_qualified_name IF NOT EXISTS "
        "FOR (n:Module) REQUIRE n.qualified_name IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT class_qualified_name IF NOT EXISTS "
        "FOR (n:Class) REQUIRE n.qualified_name IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT function_qualified_name IF NOT EXISTS "
        "FOR (n:Function) REQUIRE n.qualified_name IS UNIQUE"
    ),
    (
        "CREATE CONSTRAINT method_qualified_name IF NOT EXISTS "
        "FOR (n:Method) REQUIRE n.qualified_name IS UNIQUE"
    ),
    ("CREATE CONSTRAINT file_path IF NOT EXISTS FOR (n:File) REQUIRE n.path IS UNIQUE"),
)

NAME_INDEX_STATEMENTS: tuple[str, ...] = (
    "CREATE INDEX function_name IF NOT EXISTS FOR (n:Function) ON (n.name)",
    "CREATE INDEX class_name IF NOT EXISTS FOR (n:Class) ON (n.name)",
    "CREATE INDEX method_name IF NOT EXISTS FOR (n:Method) ON (n.name)",
    "CREATE INDEX decorator_name IF NOT EXISTS FOR (n:Decorator) ON (n.name)",
)

FULLTEXT_INDEX_NAME = "code_search"
VECTOR_INDEX_DIMENSIONS = 256
VECTOR_SIMILARITY_FUNCTION = "cosine"
VECTOR_INDEX_NAMES: dict[str, str] = {
    LABEL_FUNCTION: "function_embeddings",
    LABEL_METHOD: "method_embeddings",
    LABEL_CLASS: "class_embeddings",
}

FULLTEXT_INDEX_STATEMENTS: tuple[str, ...] = (
    (
        "CREATE FULLTEXT INDEX code_search IF NOT EXISTS "
        "FOR (n:Module|Class|Function|Method|Docstring) "
        "ON EACH [n.name, n.qualified_name, n.text]"
    ),
)

VECTOR_INDEX_STATEMENTS: tuple[str, ...] = tuple(
    (
        f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
        f"FOR (n:{label}) ON (n.embedding) "
        "OPTIONS {indexConfig: {"
        f"`vector.dimensions`: {VECTOR_INDEX_DIMENSIONS}, "
        f"`vector.similarity_function`: '{VECTOR_SIMILARITY_FUNCTION}'"
        "}}"
    )
    for label, index_name in VECTOR_INDEX_NAMES.items()
)

INDEX_STATEMENTS: tuple[str, ...] = (
    NAME_INDEX_STATEMENTS + FULLTEXT_INDEX_STATEMENTS + VECTOR_INDEX_STATEMENTS
)

SCHEMA_STATEMENTS: tuple[str, ...] = CONSTRAINT_STATEMENTS + INDEX_STATEMENTS


def _run_statement(tx: ManagedTransaction, statement: str) -> None:
    tx.run(statement)


def ensure_schema(client: GraphClient) -> None:
    """Create uniqueness constraints, name indexes, full-text, and vector indexes.
    
    Args:
        client: GraphClient.
    """
    log.info("neo4j.ensure_schema.start")
    with client.session(write=True) as session:
        for statement in SCHEMA_STATEMENTS:
            session.execute_write(_run_statement, statement)
    log.info("neo4j.ensure_schema.done")
