"""First layer: decide from the parsed SQL whether a query may run.

Regexes can't do this job, since comments, quoting and case all defeat them. The guard works on the syntax
tree and uses allow-lists throughout, so a construct it has never heard of is refused rather than let through.
It's deliberately conservative: when it can't tell which table a column belongs to, it assumes the worst.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from .policy import Policy

# sqlglot logs a warning for each statement it falls back on, and each of those is refused here anyway.
logging.getLogger("sqlglot").setLevel(logging.ERROR)

# Rejections a retry can't fix. The agent stops on these, which also denies an attacker a probing loop.
POLICY_CODES = frozenset(
    {"not_select", "multiple_statements", "table_denied", "column_denied", "function_denied", "star_denied", "recursive", "join_denied"}
)


_PARSER_NAMES = {"time_to_str": "strftime", "ts_or_ds_to_timestamp": "strftime"}


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    code: str
    reason: str
    sql: str | None = None  # the query to run, with the row limit applied


def _deny(code: str, reason: str) -> Verdict:
    return Verdict(False, code, reason)


def check(sql: str, policy: Policy) -> Verdict:
    if len(sql) > policy.max_sql_chars:
        return _deny("too_long", f"query is longer than {policy.max_sql_chars} characters")
    if "\x00" in sql:
        return _deny("parse_error", "query contains a NUL byte")
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except SqlglotError as error:
        return _deny("parse_error", f"could not parse the query: {str(error).splitlines()[0][:200]}")
    if not statements:
        return _deny("parse_error", "query is empty")
    if len(statements) > 1:
        return _deny("multiple_statements", "only one statement may run at a time")
    root = statements[0]
    if not isinstance(root, exp.Select | exp.SetOperation):
        return _deny("not_select", f"only SELECT is allowed, got {type(root).__name__}")

    for check_part in (_check_nodes, _check_tables_and_columns):
        verdict = check_part(root, policy)
        if verdict is not None:
            return verdict
    return _apply_limit(root, policy)


def _check_nodes(root: exp.Expression, policy: Policy) -> Verdict | None:
    for node in root.walk():
        if isinstance(
            node, exp.Insert | exp.Update | exp.Delete | exp.Drop | exp.Create | exp.Alter | exp.Command | exp.Merge | exp.Pragma
        ):
            return _deny("not_select", f"{type(node).__name__} is not allowed")
        if isinstance(node, exp.With) and node.args.get("recursive"):
            return _deny("recursive", "recursive queries are not allowed")
        if isinstance(node, exp.Func) and not isinstance(node, exp.Case | exp.If | exp.And | exp.Or | exp.Xor):
            name = _function_name(node)
            if name not in policy.functions:
                return _deny("function_denied", f"function {name} is not allowed")
    return None


def _function_name(node: exp.Func) -> str:
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    # sqlglot normalises some functions to its own names, which the policy shouldn't have to know about.
    name = node.sql_names()[0].lower()
    return _PARSER_NAMES.get(name, name)


def _check_tables_and_columns(root: exp.Expression, policy: Policy) -> Verdict | None:
    cte_names = {cte.alias.lower() for cte in root.find_all(exp.CTE)}
    for name in cte_names:
        # A CTE named after a real table would make the checks below mistake one for the other.
        if name in policy.tables or name.startswith(("sqlite_", "pragma_")):
            return _deny("table_denied", f"a common table expression may not be named {name}")
    # One alias can name several tables across scopes, or even in one scope, so it maps to a set.
    aliases: dict[str, set[str]] = defaultdict(set)
    real_tables: set[str] = set()
    for table in root.find_all(exp.Table):
        name = table.name.lower()
        if not name:
            return _deny("table_denied", "only plain table names are allowed")
        if name in cte_names and not table.db:
            continue
        if table.db or table.catalog:
            return _deny("table_denied", f"schema-qualified table {table.sql(dialect='sqlite')} is not allowed")
        if name not in policy.tables:
            return _deny("table_denied", f"table {table.name} is not allowed")
        real_tables.add(name)
        aliases[(table.alias or table.name).lower()].add(name)

    restricted = {t for t in real_tables if policy.denied_columns.get(t)}

    # These two joins compare columns no query text names, and SQLite's authorizer doesn't report those reads
    # either, so this is the only layer that can stop them. Found by the attack suite.
    for join in root.find_all(exp.Join):
        if str(join.args.get("method") or "").upper() == "NATURAL" and restricted:
            return _deny("join_denied", f"NATURAL JOIN is not allowed on {', '.join(sorted(restricted))}; use ON")
        for identifier in join.args.get("using") or []:
            for table in real_tables:
                if not policy.column_allowed(table, identifier.name):
                    return _deny("column_denied", f"column {identifier.name} of {table} is not allowed")

    for star in root.find_all(exp.Star):
        # COUNT(*) reads no column values. Any other star could expand to a denied column.
        if isinstance(star.parent, exp.Count):
            continue
        if restricted:
            return _deny("star_denied", f"SELECT * is not allowed on {', '.join(sorted(restricted))}; name the columns")

    for column in root.find_all(exp.Column):
        name = column.name.lower()
        qualifier = column.table.lower()
        # Unqualified, or qualified by a CTE or subquery alias: it could be any table in the query.
        candidates = aliases.get(qualifier, real_tables)
        for table in candidates:
            if not policy.column_allowed(table, name):
                return _deny("column_denied", f"column {column.name} of {table} is not allowed")
    return None


def _apply_limit(root: exp.Expression, policy: Policy) -> Verdict:
    limit = root.args.get("limit")
    cap = policy.max_rows
    if limit is None:
        root = root.limit(cap, copy=False)
    else:
        value = limit.expression
        # SQLite treats a negative limit as no limit, and an expression could evaluate to anything.
        if not (isinstance(value, exp.Literal) and value.is_int and 0 <= int(value.this) <= cap):
            limit.set("expression", exp.Literal.number(cap))
    return Verdict(True, "ok", "allowed", root.sql(dialect="sqlite"))
