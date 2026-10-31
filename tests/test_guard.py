from pathlib import Path

import pytest

from sqlagent import guard
from sqlagent.evaluate import load_jsonl, validate_reference_set
from sqlagent.policy import CHINOOK_POLICY, Policy
from sqlagent.redteam import evaluate, load_attacks

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "chinook.sqlite"


def check(sql: str) -> guard.Verdict:
    return guard.check(sql, CHINOOK_POLICY)


@pytest.mark.parametrize(
    ("sql", "code"),
    [
        ("DROP TABLE Customer", "not_select"),
        ("SELECT 1; DELETE FROM Genre", "multiple_statements"),
        ("SELECT Email FROM Customer", "column_denied"),
        ("SELECT eMaIl FROM cUsToMeR", "column_denied"),
        ("SELECT c.Email FROM Customer c", "column_denied"),
        ("SELECT * FROM Customer", "star_denied"),
        ("SELECT c.* FROM Customer c", "star_denied"),
        ("SELECT name FROM sqlite_master", "table_denied"),
        ("SELECT name FROM main.Genre", "table_denied"),
        ("SELECT load_extension('x')", "function_denied"),
        ("WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT x FROM r", "recursive"),
        ("SELEC 1 FROM", "parse_error"),
        ("", "parse_error"),
    ],
)
def test_guard_refuses_with_the_right_code(sql, code):
    verdict = check(sql)
    assert not verdict.allowed
    assert verdict.code == code


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM Customer NATURAL JOIN Employee",
        "SELECT count(*) FROM Customer NATURAL LEFT JOIN Employee",
        "SELECT count(*) FROM Customer JOIN Employee USING (Email)",
    ],
)
def test_implicit_column_comparisons_are_refused(sql):
    """Found by the attack suite: these read protected columns no query text names, past both layers."""
    assert not check(sql).allowed


def test_natural_join_on_unrestricted_tables_is_allowed():
    assert check("SELECT count(*) FROM Album NATURAL JOIN Artist").allowed


def test_using_an_allowed_column_is_allowed():
    assert check("SELECT count(*) FROM Album JOIN Artist USING (ArtistId)").allowed


def test_alias_reused_across_scopes_still_counts_as_the_restricted_table():
    assert not check("SELECT c.Email FROM Customer c WHERE EXISTS (SELECT 1 FROM Genre c)").allowed


def test_cte_cannot_shadow_a_real_table():
    assert not check("WITH Customer AS (SELECT Email FROM Customer) SELECT Email FROM Customer").allowed


def test_star_is_allowed_on_tables_without_protected_columns():
    assert check("SELECT * FROM Genre").allowed


def test_count_star_is_allowed_on_restricted_tables():
    assert check("SELECT count(*) FROM Customer").allowed


def test_allowed_columns_of_restricted_tables_work():
    assert check("SELECT FirstName, Country FROM Customer").allowed


@pytest.mark.parametrize("sql", ["SELECT Name FROM Genre", "SELECT Name FROM Genre LIMIT -1", "SELECT Name FROM Genre LIMIT 100000"])
def test_limit_is_capped(sql):
    assert "LIMIT 200" in check(sql).sql


def test_small_limit_is_kept():
    assert "LIMIT 5" in check("SELECT Name FROM Genre LIMIT 5").sql


def test_too_long_is_refused():
    policy = Policy(tables=frozenset({"Genre"}), max_sql_chars=10)
    assert guard.check("SELECT Name FROM Genre", policy).code == "too_long"


def test_policy_is_case_insensitive():
    policy = Policy(tables=frozenset({"T"}), denied_columns={"T": frozenset({"Secret"})})
    assert not policy.column_allowed("t", "SECRET")
    assert policy.column_allowed("T", "other")


def test_guard_never_blocks_a_reference_query():
    """The other half of a security check: it must not refuse legitimate work."""
    questions = load_jsonl(ROOT / "eval" / "questions.jsonl")
    report = validate_reference_set(DB, questions, CHINOOK_POLICY)
    assert report["rejected_by_guard"] == []
    assert report["refused_by_database"] == []
    assert report["changed_by_rewrite"] == []
    assert report["empty_results"] == []
    assert report["questions"] == 60


def test_attack_suite_defences_hold():
    report = evaluate(DB, load_attacks(ROOT / "eval" / "attacks_sql.jsonl"))
    assert report["not_stopped_by_both"] == []
    assert report["values_leaked_with_both"] == 0
    assert report["database_file_unchanged"] is True


def test_database_layer_stands_on_its_own_except_for_known_join_gaps():
    """With the guard off, the only things that get through are the joins SQLite's authorizer can't see."""
    report = evaluate(DB, load_attacks(ROOT / "eval" / "attacks_sql.jsonl"))
    # schema_cte_shadow runs, but it only reads its own CTE. It returns the literal 1 and no real schema.
    assert set(report["not_stopped_by_database"]) <= {"pii_natural_join", "pii_natural_left_join", "pii_using_join", "schema_cte_shadow"}
    assert report["values_leaked_database_alone"] == 0
