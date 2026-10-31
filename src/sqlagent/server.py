"""The MCP server: the only route from a model to the database.

Schema tools hide denied columns, so the model is never told they exist. Refusals are returned as data rather than
raised, because an agent needs the reason to fix its query, and every call leaves an audit record.

    sqlagent-server --db data/chinook.sqlite --audit audit.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from . import database, guard
from .policy import CHINOOK_POLICY, Policy


class SqlTools:
    """Tool logic, kept free of MCP so it can be tested and audited directly."""

    def __init__(self, db_path: str | Path, policy: Policy, audit_path: str | Path | None = None, use_guard: bool = True):
        self.db_path = db_path
        self.policy = policy
        self.audit_path = Path(audit_path) if audit_path else None
        self.use_guard = use_guard  # only the attack suite turns this off, to test the database layer alone
        self.tables = self._read_tables()

    def _read_tables(self) -> dict[str, dict]:
        # Schema reads use a separate plain read-only connection. The tool connections can't reach sqlite_master.
        uri = Path(self.db_path).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            found = {}
            for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name").fetchall():
                if name.lower() not in self.policy.tables:
                    continue
                columns = [
                    {"name": row[1], "type": row[2], "primary_key": bool(row[5])}
                    for row in conn.execute(f'PRAGMA table_info("{name}")')
                    if self.policy.column_allowed(name, row[1])
                ]
                keys = [
                    {"column": row[3], "references": f"{row[2]}.{row[4]}"}
                    for row in conn.execute(f'PRAGMA foreign_key_list("{name}")')
                    if self.policy.column_allowed(name, row[3])
                ]
                found[name] = {"columns": columns, "foreign_keys": keys}
            return found
        finally:
            conn.close()

    def list_tables(self) -> list[str]:
        return list(self.tables)

    def describe_table(self, table: str, samples: int = 0) -> dict[str, Any]:
        match = next((name for name in self.tables if name.lower() == table.lower()), None)
        if match is None:
            return {"ok": False, "error": f"unknown table: {table}"}
        result: dict[str, Any] = {"ok": True, "table": match, **self.tables[match]}
        if samples > 0:
            result["sample_values"] = self._sample_values(match, samples)
        return result

    def _sample_values(self, table: str, limit: int) -> dict[str, list]:
        """Distinct values of short text columns: this is how a model learns that a country is spelled 'USA'."""
        conn = database.connect(self.db_path, self.policy)
        values = {}
        try:
            for column in self.tables[table]["columns"]:
                if not column["type"].upper().startswith(("NVARCHAR", "VARCHAR", "TEXT")):
                    continue
                result = database.run(
                    conn,
                    f'SELECT DISTINCT "{column["name"]}" FROM "{table}" WHERE "{column["name"]}" IS NOT NULL LIMIT {limit}',
                    self.policy,
                )
                found = [row[0] for row in result.rows if len(str(row[0])) <= 40]
                if found:
                    values[column["name"]] = found
        finally:
            conn.close()
        return values

    def run_query(self, sql: str) -> dict[str, Any]:
        started = time.monotonic()
        code, outcome = "ok", None
        try:
            to_run = sql
            if self.use_guard:
                verdict = guard.check(sql, self.policy)
                if not verdict.allowed:
                    code = verdict.code
                    outcome = {"ok": False, "code": verdict.code, "error": verdict.reason}
                    return outcome
                to_run = verdict.sql
            conn = database.connect(self.db_path, self.policy)
            try:
                result = database.run(conn, to_run, self.policy)
            except database.QueryError as error:
                code = "database_refused" if "not authorized" in str(error) or "prohibited" in str(error) else "execution_error"
                outcome = {"ok": False, "code": code, "error": str(error)}
                return outcome
            finally:
                conn.close()
            outcome = {
                "ok": True,
                "code": "ok",
                "sql": to_run,
                "columns": result.columns,
                "rows": result.rows,
                "truncated": result.truncated,
                "elapsed_ms": round(result.elapsed_ms, 2),
            }
            return outcome
        finally:
            self._audit(sql, code, outcome, time.monotonic() - started)

    def _audit(self, sql: str, code: str, outcome: dict | None, seconds: float) -> None:
        if self.audit_path is None:
            return
        record = {
            "ts": time.time(),
            "sha256": hashlib.sha256(sql.encode()).hexdigest()[:16],
            "sql": sql[:2000],
            "code": code,
            "rows": len(outcome["rows"]) if outcome and outcome.get("ok") else 0,
            "ms": round(seconds * 1000, 2),
        }
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")


def build_server(tools: SqlTools) -> MCPServer:
    server = MCPServer(
        "sql-agent", instructions="Read-only access to one SQL database. Call describe_table before writing a query.", log_level="WARNING"
    )

    @server.tool()
    def list_tables() -> list[str]:
        """List the tables you can query."""
        return tools.list_tables()

    @server.tool()
    def describe_table(table: str, samples: int = 0) -> dict:
        """Describe a table's columns and foreign keys, with up to `samples` example values per text column."""
        return tools.describe_table(table, min(max(samples, 0), 20))

    @server.tool()
    def run_query(sql: str) -> dict:
        """Run one read-only SQL SELECT. Returns columns and rows, or a code and an error to fix the query from."""
        return tools.run_query(sql)

    return server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default="data/chinook.sqlite")
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()
    build_server(SqlTools(args.db, CHINOOK_POLICY, args.audit)).run("stdio")


if __name__ == "__main__":
    main()
