"""Automatic database schema introspection, semantic table catalog, and FK join-path finder.

Introspects SQLite databases while strictly enforcing :class:`Policy` allowlists and
column deny-lists so protected PII columns never enter the catalog or LLM prompts.
"""

from __future__ import annotations

import dataclasses
import re
import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .policy import Policy


@dataclass(frozen=True)
class ColumnInfo:
    """Policy-filtered column metadata with inferred semantic role."""

    name: str
    data_type: str
    nullable: bool
    primary_key: bool
    semantic_role: str


@dataclass(frozen=True)
class ForeignKeyInfo:
    """Directed foreign-key edge from ``column`` to ``ref_table.ref_column``."""

    column: str
    ref_table: str
    ref_column: str


@dataclass(frozen=True)
class TableInfo:
    """Introspected table metadata, row count, columns, foreign keys, and semantic tags."""

    name: str
    row_count: int
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKeyInfo]
    description: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


def infer_semantic_role(column_name: str, data_type: str, primary_key: bool) -> str:
    """Infer a coarse semantic role for schema-linking and prompt enrichment."""
    lower = column_name.lower()
    dtype = data_type.upper()
    if primary_key or lower.endswith("id"):
        return "identifier"
    if any(k in lower for k in ("price", "total", "cost", "amount", "revenue")):
        return "currency"
    if any(k in lower for k in ("date", "time", "created", "updated")):
        return "timestamp"
    if any(k in lower for k in ("country", "city", "state", "region")):
        return "geography"
    if any(k in dtype for k in ("INT", "REAL", "NUMERIC", "FLOAT", "DOUBLE")):
        return "metric"
    return "text"


def _tokenize(text: str) -> set[str]:
    # Split camelCase and snake_case / whitespace into lowercase search tokens.
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return {tok for tok in re.findall(r"[a-z0-9]+", spaced.lower()) if len(tok) > 1}


class SemanticCatalog:
    """In-memory policy-safe schema catalog with lexical schema linking and FK join pathfinding."""

    def __init__(self, tables: list[TableInfo]) -> None:
        self.tables: dict[str, TableInfo] = {t.name: t for t in tables}
        self._by_lower: dict[str, str] = {t.name.lower(): t.name for t in tables}

    def get_table(self, name: str) -> TableInfo | None:
        canonical = self._by_lower.get(name.lower())
        return self.tables[canonical] if canonical else None

    def search(self, query: str, *, top_k: int = 5) -> list[TableInfo]:
        """Rank tables by token overlap across table names, descriptions, tags, and column names."""
        q_tokens = _tokenize(query)
        if not q_tokens:
            return list(self.tables.values())[:top_k]

        scored: list[tuple[float, str, TableInfo]] = []
        for table in self.tables.values():
            name_tokens = _tokenize(table.name)
            col_tokens = set().union(*(_tokenize(c.name) for c in table.columns)) if table.columns else set()
            desc_tokens = (
                _tokenize(table.description) | set().union(*(_tokenize(t) for t in table.tags))
                if table.tags
                else _tokenize(table.description)
            )
            score = 3.0 * len(q_tokens & name_tokens) + 1.5 * len(q_tokens & col_tokens) + 1.0 * len(q_tokens & desc_tokens)
            scored.append((score, table.name, table))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [t for score, _, t in scored[:top_k] if score > 0] or [t for _, _, t in scored[:top_k]]

    def join_path(self, source_table: str, target_table: str) -> list[tuple[str, str, str, str]]:
        """Find the shortest undirected foreign-key join path between two tables.

        Returns a list of ``(left_table, left_col, right_table, right_col)`` hops.
        """
        src = self._by_lower.get(source_table.lower())
        dst = self._by_lower.get(target_table.lower())
        if src is None or dst is None:
            raise KeyError(f"Unknown table in join_path: {source_table!r} -> {target_table!r}")
        if src == dst:
            return []

        adj: dict[str, list[tuple[str, tuple[str, str, str, str]]]] = {name: [] for name in self.tables}
        for table in self.tables.values():
            for fk in table.foreign_keys:
                ref = self._by_lower.get(fk.ref_table.lower())
                if ref is None:
                    continue
                adj[table.name].append((ref, (table.name, fk.column, ref, fk.ref_column)))
                adj[ref].append((table.name, (ref, fk.ref_column, table.name, fk.column)))

        queue: deque[tuple[str, list[tuple[str, str, str, str]]]] = deque([(src, [])])
        visited = {src}
        while queue:
            curr, path = queue.popleft()
            for nxt, hop in adj[curr]:
                if nxt == dst:
                    return [*path, hop]
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, [*path, hop]))
        return []

    def to_prompt_markdown(self, table_names: list[str] | None = None) -> str:
        """Render a compact Markdown schema catalog for LLM system prompts."""
        selected = [self.tables[n] for n in table_names if n in self.tables] if table_names is not None else list(self.tables.values())
        lines: list[str] = []
        for t in selected:
            header = f"### {t.name} ({t.row_count:,} rows)"
            if t.description:
                header += f" — {t.description}"
            lines.append(header)
            col_desc = ", ".join(f"`{c.name}` {c.data_type} [{c.semantic_role}]" for c in t.columns)
            lines.append(f"- Columns: {col_desc}")
            if t.foreign_keys:
                fk_desc = ", ".join(f"`{fk.column}` -> `{fk.ref_table}.{fk.ref_column}`" for fk in t.foreign_keys)
                lines.append(f"- Foreign keys: {fk_desc}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {"tables": [dataclasses.asdict(t) for t in self.tables.values()]}


def introspect_sqlite(
    db_path: str | Path,
    policy: Policy,
    *,
    descriptions: dict[str, str] | None = None,
) -> SemanticCatalog:
    """Introspect a SQLite database in read-only mode, filtering strictly by ``policy``."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    descriptions = descriptions or {}
    tables: list[TableInfo] = []
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall()
        for (name,) in rows:
            if name.lower() not in policy.tables:
                continue
            row_count = int(conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0])
            columns = [
                ColumnInfo(
                    name=row[1],
                    data_type=row[2] or "TEXT",
                    nullable=not bool(row[3]),
                    primary_key=bool(row[5]),
                    semantic_role=infer_semantic_role(row[1], row[2] or "TEXT", bool(row[5])),
                )
                for row in conn.execute(f'PRAGMA table_info("{name}")')
                if policy.column_allowed(name, row[1])
            ]
            foreign_keys = [
                ForeignKeyInfo(column=row[3], ref_table=row[2], ref_column=row[4])
                for row in conn.execute(f'PRAGMA foreign_key_list("{name}")')
                if policy.column_allowed(name, row[3]) and row[2].lower() in policy.tables
            ]
            tags = tuple(sorted({c.semantic_role for c in columns}))
            tables.append(
                TableInfo(
                    name=name,
                    row_count=row_count,
                    columns=columns,
                    foreign_keys=foreign_keys,
                    description=descriptions.get(name, ""),
                    tags=tags,
                )
            )
    finally:
        conn.close()
    return SemanticCatalog(tables)
