"""Tests for multi-tenant RLS filter injection and column masking."""

from __future__ import annotations

from pathlib import Path

import pytest

from sqlagent import database
from sqlagent.policy import CHINOOK_POLICY
from sqlagent.rls import MaskingRule, RLSFilterInjector, RLSRule, TenantContext, mask_rows

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "chinook.sqlite"


def test_rls_injects_predicate_and_prevents_or_bypass_on_chinook() -> None:
    injector = RLSFilterInjector([RLSRule("Customer", "Country", "tenant_id")])
    tenant = TenantContext(tenant_id="USA")

    # Even with an OR clause attempting to include Brazil, RLS restricts results strictly to USA.
    sql = "SELECT FirstName, Country FROM Customer WHERE Country = 'Brazil' OR Country = 'USA'"
    rewritten = injector.rewrite(sql, tenant)
    assert "(Country = 'Brazil' OR Country = 'USA') AND Customer.Country = 'USA'" in rewritten

    conn = database.connect(DB, CHINOOK_POLICY)
    try:
        res = database.run(conn, rewritten, CHINOOK_POLICY)
    finally:
        conn.close()
    assert len(res.rows) > 0
    assert all(row[1] == "USA" for row in res.rows)


def test_rls_respects_table_aliases_in_joins() -> None:
    injector = RLSFilterInjector(
        [
            RLSRule("Customer", "Country", "tenant_id"),
            RLSRule("Invoice", "BillingCountry", "tenant_id"),
        ]
    )
    tenant = TenantContext(tenant_id="Canada")
    sql = "SELECT c.FirstName, i.Total FROM Customer c JOIN Invoice i ON c.CustomerId = i.CustomerId"
    rewritten = injector.rewrite(sql, tenant)
    assert "c.Country = 'Canada'" in rewritten
    assert "i.BillingCountry = 'Canada'" in rewritten


def test_rls_rejects_empty_tenant_or_multiple_statements() -> None:
    injector = RLSFilterInjector([RLSRule("Customer", "Country")])
    with pytest.raises(ValueError, match="tenant_id"):
        injector.rewrite("SELECT FirstName FROM Customer", TenantContext(tenant_id=""))
    with pytest.raises(ValueError, match="one SQL statement"):
        injector.rewrite("SELECT 1; SELECT 2", TenantContext(tenant_id="USA"))


def test_mask_rows_applies_all_masking_strategies() -> None:
    cols = ["FirstName", "LastName", "Email", "Phone", "AccountId"]
    rows = [["Alice", "Smith", "alice@example.com", "555-123-4567", "ACC9988"]]
    rules = [
        MaskingRule("LastName", "redact"),
        MaskingRule("Email", "partial_email"),
        MaskingRule("Phone", "last4"),
        MaskingRule("AccountId", "hash"),
    ]
    masked = mask_rows(cols, rows, rules)
    assert masked[0][0] == "Alice"
    assert masked[0][1] == "[REDACTED]"
    assert masked[0][2] == "a***@example.com"
    assert masked[0][3] == "***4567"
    assert masked[0][4].startswith("sha256:")
    # Original input rows must not be mutated.
    assert rows[0][1] == "Smith"
