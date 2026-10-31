"""Second layer: a connection that can't do harm even if the guard has a bug.

SQLite's authorizer runs while a statement is compiled, on SQLite's own resolved view of the query, so aliases,
quoting and star expansion have already been dealt with. It would catch a guard bypass that a parser disagreement
made possible. The attack suite tests this layer with the guard switched off.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .policy import Policy


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[list]
    truncated: bool
    elapsed_ms: float


class QueryError(Exception):
    """A query that the database refused or failed to run."""


def connect(path: str | Path, policy: Policy) -> sqlite3.Connection:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, isolation_level=None, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    conn.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, policy.max_cell_bytes)
    conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, policy.max_sql_chars * 2)
    # Installed last, because from here on even PRAGMA is refused.
    conn.set_authorizer(_authorizer(policy))
    return conn


def _authorizer(policy: Policy):
    def decide(action: int, arg1: str | None, arg2: str | None, _db: str | None, _trigger: str | None) -> int:
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table, column = (arg1 or "").lower(), arg2 or ""
            # An empty column name is how SQLite reports a table touched without reading a column, as in COUNT(*).
            if table in policy.tables and (not column or policy.column_allowed(table, column)):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_OK if (arg2 or "").lower() in policy.functions else sqlite3.SQLITE_DENY
        # Everything else, recursive CTEs, writes, ATTACH and PRAGMA included, is refused.
        return sqlite3.SQLITE_DENY

    return decide


def run(conn: sqlite3.Connection, sql: str, policy: Policy) -> QueryResult:
    """Execute under a time limit and a row cap, whatever the SQL says."""
    start = time.monotonic()
    deadline = start + policy.timeout_seconds
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)
    try:
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(policy.max_rows + 1)
        columns = [item[0] for item in cursor.description or []]
    except sqlite3.Error as error:
        raise QueryError(str(error)) from error
    finally:
        conn.set_progress_handler(None, 0)
    truncated = len(rows) > policy.max_rows
    return QueryResult(columns, [list(r) for r in rows[: policy.max_rows]], truncated, (time.monotonic() - start) * 1000)
