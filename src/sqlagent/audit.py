"""Query timeout, row-limit & byte-cap guard, and tamper-evident structured audit logger.

Provides:

* :class:`QueryBudget`: configurable timeout, row cap, and serialized result byte budget.
* :class:`StructuredAuditLogger`: JSONL audit writer with SHA-256 hash chaining
  (:func:`verify_audit_chain`) and latency/outcome summary metrics.
* :class:`GuardedExecutor`: executes read-only queries under strict timeout, row, and byte
  budgets while emitting structured :class:`AuditEvent` records on every call.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import database, guard
from .policy import Policy


@dataclass(frozen=True)
class QueryBudget:
    """Hard resource limits enforced during SQL execution."""

    timeout_seconds: float = 2.0
    max_rows: int = 200
    max_result_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")
        if self.max_rows < 1:
            raise ValueError("max_rows must be >= 1")
        if self.max_result_bytes < 64:
            raise ValueError("max_result_bytes must be >= 64")


@dataclass(frozen=True)
class AuditEvent:
    """Structured audit record for a single SQL execution attempt."""

    timestamp_utc: str
    request_id: str
    tenant_id: str
    sql_sha256: str
    sql_preview: str
    status: str  # ok | blocked | timeout | error | byte_limit_exceeded
    code: str
    row_count: int
    truncated: bool
    elapsed_ms: float
    prev_hash: str
    entry_hash: str


def _compute_entry_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_audit_chain(path: str | Path) -> bool:
    """Verify the SHA-256 hash chain of a JSONL audit log file."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_prev = "0" * 64
    for line in lines:
        record = json.loads(line)
        recorded_hash = record.pop("entry_hash", None)
        if record.get("prev_hash") != expected_prev:
            return False
        computed = _compute_entry_hash(record)
        if computed != recorded_hash:
            return False
        expected_prev = computed
    return True


class StructuredAuditLogger:
    """Append-only, hash-chained JSONL audit logger with in-memory summary analytics."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.events: list[AuditEvent] = []
        self._last_hash = "0" * 64

    def log(
        self,
        *,
        sql: str,
        tenant_id: str,
        status: str,
        code: str,
        row_count: int,
        truncated: bool,
        elapsed_ms: float,
        request_id: str | None = None,
    ) -> AuditEvent:
        base = {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "request_id": request_id or uuid.uuid4().hex[:12],
            "tenant_id": tenant_id,
            "sql_sha256": hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16],
            "sql_preview": sql.strip().replace("\n", " ")[:300],
            "status": status,
            "code": code,
            "row_count": row_count,
            "truncated": truncated,
            "elapsed_ms": round(elapsed_ms, 3),
            "prev_hash": self._last_hash,
        }
        entry_hash = _compute_entry_hash(base)
        event = AuditEvent(**base, entry_hash=entry_hash)
        self._last_hash = entry_hash
        self.events.append(event)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dataclasses.asdict(event)) + "\n")
        return event

    def summary(self) -> dict[str, Any]:
        total = len(self.events)
        if total == 0:
            return {"total": 0, "ok": 0, "blocked": 0, "timeouts": 0, "truncated": 0, "mean_ms": 0.0}
        return {
            "total": total,
            "ok": sum(1 for e in self.events if e.status == "ok"),
            "blocked": sum(1 for e in self.events if e.status == "blocked"),
            "timeouts": sum(1 for e in self.events if e.status == "timeout"),
            "truncated": sum(1 for e in self.events if e.truncated),
            "mean_ms": round(sum(e.elapsed_ms for e in self.events) / total, 3),
        }


class GuardedExecutor:
    """Executes SQL queries under policy, timeout, row-limit, and result byte-size guards."""

    def __init__(
        self,
        db_path: str | Path,
        policy: Policy,
        *,
        budget: QueryBudget | None = None,
        audit_logger: StructuredAuditLogger | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.budget = budget or QueryBudget(timeout_seconds=policy.timeout_seconds, max_rows=policy.max_rows)
        self.policy = Policy(
            tables=policy.tables,
            denied_columns=policy.denied_columns,
            functions=policy.functions,
            max_rows=self.budget.max_rows,
            max_sql_chars=policy.max_sql_chars,
            timeout_seconds=self.budget.timeout_seconds,
            max_cell_bytes=policy.max_cell_bytes,
        )
        self.audit = audit_logger or StructuredAuditLogger()

    def execute(self, sql: str, *, tenant_id: str = "default") -> dict[str, Any]:
        started = time.monotonic()
        verdict = guard.check(sql, self.policy)
        if not verdict.allowed or verdict.sql is None:
            elapsed = (time.monotonic() - started) * 1000.0
            self.audit.log(
                sql=sql,
                tenant_id=tenant_id,
                status="blocked",
                code=verdict.code,
                row_count=0,
                truncated=False,
                elapsed_ms=elapsed,
            )
            return {"ok": False, "code": verdict.code, "error": verdict.reason}

        conn = database.connect(self.db_path, self.policy)
        try:
            deadline = time.monotonic() + self.budget.timeout_seconds
            conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 500)
            cursor = conn.execute(verdict.sql)
            raw_rows = cursor.fetchmany(self.budget.max_rows + 1)
            columns = [item[0] for item in cursor.description or []]
        except sqlite3.OperationalError as exc:
            elapsed = (time.monotonic() - started) * 1000.0
            is_timeout = "interrupted" in str(exc).lower()
            status = "timeout" if is_timeout else "error"
            code = "query_timeout" if is_timeout else "execution_error"
            self.audit.log(
                sql=sql,
                tenant_id=tenant_id,
                status=status,
                code=code,
                row_count=0,
                truncated=False,
                elapsed_ms=elapsed,
            )
            return {"ok": False, "code": code, "error": str(exc)}
        except sqlite3.Error as exc:
            elapsed = (time.monotonic() - started) * 1000.0
            self.audit.log(
                sql=sql,
                tenant_id=tenant_id,
                status="blocked",
                code="database_refused",
                row_count=0,
                truncated=False,
                elapsed_ms=elapsed,
            )
            return {"ok": False, "code": "database_refused", "error": str(exc)}
        finally:
            conn.set_progress_handler(None, 0)
            conn.close()

        truncated = len(raw_rows) > self.budget.max_rows
        rows = [list(r) for r in raw_rows[: self.budget.max_rows]]
        result_bytes = len(json.dumps(rows, default=str).encode("utf-8"))
        elapsed = (time.monotonic() - started) * 1000.0

        if result_bytes > self.budget.max_result_bytes:
            self.audit.log(
                sql=sql,
                tenant_id=tenant_id,
                status="byte_limit_exceeded",
                code="result_too_large",
                row_count=len(rows),
                truncated=True,
                elapsed_ms=elapsed,
            )
            return {
                "ok": False,
                "code": "result_too_large",
                "error": f"Result size {result_bytes} bytes exceeds budget {self.budget.max_result_bytes}",
            }

        self.audit.log(
            sql=sql,
            tenant_id=tenant_id,
            status="ok",
            code="ok",
            row_count=len(rows),
            truncated=truncated,
            elapsed_ms=elapsed,
        )
        return {
            "ok": True,
            "code": "ok",
            "columns": columns,
            "rows": rows,
            "truncated": truncated,
            "elapsed_ms": round(elapsed, 3),
        }
