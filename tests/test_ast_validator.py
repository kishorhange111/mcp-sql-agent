"""Tests for the read-only SQL AST validator and DDL/DML/injection blocker."""

from __future__ import annotations

import pytest

from sqlagent.ast_validator import ReadOnlyASTValidator


def test_allows_clean_select_and_extracts_metadata() -> None:
    validator = ReadOnlyASTValidator(allowed_tables=frozenset({"Album", "Artist"}))
    report = validator.validate(
        "SELECT ar.Name, COUNT(al.AlbumId) AS n FROM Artist ar JOIN Album al ON ar.ArtistId = al.ArtistId GROUP BY ar.Name"
    )
    assert report.allowed is True
    assert report.risk_level == "low"
    assert report.findings == []
    assert report.tables_referenced == ("album", "artist")
    assert "count" in report.functions_used
    assert report.normalized_sql is not None


@pytest.mark.parametrize(
    ("sql", "expected_code"),
    [
        ("DROP TABLE Customer", "ddl_blocked"),
        ("CREATE TABLE evil (id INT)", "ddl_blocked"),
        ("ALTER TABLE Customer ADD COLUMN x TEXT", "ddl_blocked"),
        ("INSERT INTO Genre (Name) VALUES ('hacked')", "dml_blocked"),
        ("UPDATE Customer SET Country = 'X'", "dml_blocked"),
        ("DELETE FROM Invoice WHERE InvoiceId = 1", "dml_blocked"),
        ("PRAGMA table_info(Customer)", "system_command_blocked"),
        ("SELECT 1; DROP TABLE Genre", "stacked_queries"),
        ("SELECT * FROM sqlite_master", "system_table_blocked"),
        ("SELECT load_extension('evil.so')", "dangerous_function"),
        ("SELECT Name FROM Artist WHERE ArtistId = 1 OR 1 = 1", "tautology_injection"),
        ("SELECT Name FROM Artist WHERE Name = 'x' OR 'a' = 'a'", "tautology_injection"),
        ("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT x FROM r", "recursive_cte_blocked"),
    ],
)
def test_blocks_ddl_dml_and_injection_payloads(sql: str, expected_code: str) -> None:
    validator = ReadOnlyASTValidator()
    report = validator.validate(sql)
    assert report.allowed is False
    assert any(f.code == expected_code for f in report.findings)


def test_blocks_excessive_subquery_depth_and_joins() -> None:
    validator = ReadOnlyASTValidator(max_subquery_depth=2, max_joins=2)
    deep = "SELECT * FROM (SELECT * FROM (SELECT * FROM (SELECT 1)))"
    r_deep = validator.validate(deep)
    assert r_deep.allowed is False
    assert any(f.code == "subquery_depth_exceeded" for f in r_deep.findings)

    many_joins = "SELECT * FROM a JOIN b ON a.id=b.id JOIN c ON b.id=c.id JOIN d ON c.id=d.id"
    r_joins = validator.validate(many_joins)
    assert r_joins.allowed is False
    assert any(f.code == "join_limit_exceeded" for f in r_joins.findings)


def test_enforces_table_allowlist() -> None:
    validator = ReadOnlyASTValidator(allowed_tables=frozenset({"Genre"}))
    assert validator.validate("SELECT Name FROM Genre").allowed is True
    denied = validator.validate("SELECT Name FROM Artist")
    assert denied.allowed is False
    assert any(f.code == "table_denied" for f in denied.findings)
