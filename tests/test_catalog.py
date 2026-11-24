"""Tests for automatic database schema introspection and semantic table catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from sqlagent.catalog import introspect_sqlite
from sqlagent.policy import CHINOOK_POLICY

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "chinook.sqlite"


def test_introspect_sqlite_respects_policy_and_collects_row_counts() -> None:
    catalog = introspect_sqlite(DB, CHINOOK_POLICY, descriptions={"Invoice": "Customer purchase invoices"})
    assert "Customer" in catalog.tables
    assert "Invoice" in catalog.tables
    customer = catalog.get_table("customer")
    assert customer is not None
    assert customer.row_count > 0
    col_names = {c.name for c in customer.columns}
    for denied in ("Email", "Phone", "Address", "PostalCode", "Fax"):
        assert denied not in col_names
    assert "Country" in col_names
    country_col = next(c for c in customer.columns if c.name == "Country")
    assert country_col.semantic_role == "geography"


def test_semantic_search_ranks_relevant_tables() -> None:
    catalog = introspect_sqlite(DB, CHINOOK_POLICY)
    results = catalog.search("total invoice billing country customer", top_k=3)
    names = [t.name for t in results]
    assert "Invoice" in names[:2]
    assert "Customer" in names[:3]


def test_join_path_finds_shortest_fk_route() -> None:
    catalog = introspect_sqlite(DB, CHINOOK_POLICY)
    path = catalog.join_path("Artist", "Track")
    assert len(path) == 2
    assert path[0][0] == "Artist" and path[0][2] == "Album"
    assert path[1][0] == "Album" and path[1][2] == "Track"
    assert catalog.join_path("Genre", "Genre") == []
    with pytest.raises(KeyError):
        catalog.join_path("Genre", "NonExistent")


def test_prompt_markdown_and_dict_hide_denied_columns() -> None:
    catalog = introspect_sqlite(DB, CHINOOK_POLICY)
    md = catalog.to_prompt_markdown(["Customer", "Invoice"])
    assert "### Customer" in md
    assert "### Invoice" in md
    assert "Email" not in md
    assert "BillingAddress" not in md
    data = catalog.to_dict()
    assert len(data["tables"]) == len(CHINOOK_POLICY.tables)
