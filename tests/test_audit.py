"""Tests for query timeout, row-limit & byte-cap guard, and tamper-evident audit logger."""

from __future__ import annotations

from pathlib import Path

from sqlagent.audit import GuardedExecutor, QueryBudget, StructuredAuditLogger, verify_audit_chain
from sqlagent.policy import CHINOOK_POLICY

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "chinook.sqlite"


def test_guarded_executor_enforces_row_limit_and_audit_chain(tmp_path: Path) -> None:
    audit_file = tmp_path / "audit.jsonl"
    logger = StructuredAuditLogger(audit_file)
    executor = GuardedExecutor(DB, CHINOOK_POLICY, budget=QueryBudget(timeout_seconds=2.0, max_rows=5), audit_logger=logger)

    ok = executor.execute("SELECT Name FROM Track", tenant_id="tenant-a")
    assert ok["ok"] is True
    assert len(ok["rows"]) == 5

    blocked = executor.execute("SELECT Email FROM Customer", tenant_id="tenant-a")
    assert blocked["ok"] is False
    assert blocked["code"] == "column_denied"

    summary = logger.summary()
    assert summary["total"] == 2
    assert summary["ok"] == 1
    assert summary["blocked"] == 1
    assert verify_audit_chain(audit_file) is True

    # Tampering with a single byte breaks the hash chain.
    tampered = audit_file.read_text(encoding="utf-8").replace("tenant-a", "tenant-z", 1)
    audit_file.write_text(tampered, encoding="utf-8")
    assert verify_audit_chain(audit_file) is False


def test_guarded_executor_enforces_result_byte_cap() -> None:
    executor = GuardedExecutor(
        DB,
        CHINOOK_POLICY,
        budget=QueryBudget(timeout_seconds=2.0, max_rows=50, max_result_bytes=100),
    )
    res = executor.execute("SELECT Name FROM Track")
    assert res["ok"] is False
    assert res["code"] == "result_too_large"


def test_guarded_executor_enforces_query_timeout() -> None:
    executor = GuardedExecutor(
        DB,
        CHINOOK_POLICY,
        budget=QueryBudget(timeout_seconds=0.0001, max_rows=50),
    )
    # Cross-joining Track with itself forces enough SQLite VM steps to trip the 0.1ms progress handler.
    res = executor.execute("SELECT COUNT(*) FROM Track a, Track b, Track c")
    assert res["ok"] is False
    assert res["code"] == "query_timeout"
    assert executor.audit.summary()["timeouts"] == 1
