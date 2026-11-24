"""Read-only SQL AST validator and DDL / DML / SQL-injection blocker.

Inspects parsed ``sqlglot`` syntax trees to enforce strict read-only semantics and detect
common SQL-injection and query-DoS patterns before execution:

* Blocks all DDL (``CREATE``, ``DROP``, ``ALTER``) and DML (``INSERT``, ``UPDATE``, ``DELETE``, ``MERGE``).
* Blocks stacked multi-statement payloads, ``PRAGMA`` / ``ATTACH`` / ``DETACH`` / ``VACUUM`` commands,
  and dangerous functions (e.g., ``load_extension``, ``randomblob``).
* Detects boolean tautology injection predicates (e.g., ``OR 1 = 1``, ``OR 'a' = 'a'``).
* Enforces structural complexity bounds (subquery nesting depth and join fanout).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

logging.getLogger("sqlglot").setLevel(logging.ERROR)

RiskLevel = Literal["low", "medium", "high", "critical"]

DDL_NODES = (exp.Create, exp.Drop, exp.Alter)
DML_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Merge)
SYSTEM_NODES = (exp.Command, exp.Pragma, exp.Transaction, exp.Commit, exp.Rollback)
DANGEROUS_FUNCTIONS = frozenset({"load_extension", "randomblob", "zeroblob", "fts3_tokenizer", "sqlite_version", "edit"})


@dataclass(frozen=True)
class SecurityFinding:
    """Single security violation discovered during AST inspection."""

    code: str
    severity: RiskLevel
    message: str


@dataclass(frozen=True)
class ASTValidationReport:
    """Structured result of AST-level read-only and injection validation."""

    allowed: bool
    risk_level: RiskLevel
    findings: list[SecurityFinding] = field(default_factory=list)
    normalized_sql: str | None = None
    tables_referenced: tuple[str, ...] = ()
    functions_used: tuple[str, ...] = ()


def _is_constant_tautology(node: exp.Expression) -> bool:
    """Detect always-true literal equality predicates such as ``1 = 1`` or ``'a' = 'a'``."""
    if isinstance(node, exp.EQ):
        left, right = node.left, node.right
        if isinstance(left, exp.Literal) and isinstance(right, exp.Literal):
            return left.this == right.this and left.is_string == right.is_string
    return isinstance(node, exp.Boolean) and node.this is True


def _subquery_depth(node: exp.Expression, depth: int = 0) -> int:
    max_d = depth
    for child in node.iter_expressions():
        next_d = depth + 1 if isinstance(child, exp.Subquery) else depth
        max_d = max(max_d, _subquery_depth(child, next_d))
    return max_d


class ReadOnlyASTValidator:
    """Dialect-aware AST validator that enforces read-only queries and blocks SQL injection."""

    def __init__(
        self,
        *,
        dialect: str = "sqlite",
        max_sql_chars: int = 4000,
        max_subquery_depth: int = 4,
        max_joins: int = 8,
        block_tautologies: bool = True,
        allowed_tables: frozenset[str] | None = None,
    ) -> None:
        self.dialect = dialect
        self.max_sql_chars = max_sql_chars
        self.max_subquery_depth = max_subquery_depth
        self.max_joins = max_joins
        self.block_tautologies = block_tautologies
        self.allowed_tables = frozenset(t.lower() for t in allowed_tables) if allowed_tables is not None else None

    def validate(self, sql: str) -> ASTValidationReport:
        findings: list[SecurityFinding] = []
        if not sql or not sql.strip():
            findings.append(SecurityFinding("empty_sql", "medium", "SQL query is empty"))
            return ASTValidationReport(False, "medium", findings)
        if len(sql) > self.max_sql_chars:
            findings.append(SecurityFinding("too_long", "medium", f"Query exceeds {self.max_sql_chars} characters"))
            return ASTValidationReport(False, "medium", findings)
        if "\x00" in sql:
            findings.append(SecurityFinding("nul_byte", "critical", "Query contains a NUL byte"))
            return ASTValidationReport(False, "critical", findings)

        try:
            statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        except SqlglotError as exc:
            msg = str(exc).splitlines()[0][:200]
            findings.append(SecurityFinding("parse_error", "high", f"Failed to parse SQL: {msg}"))
            return ASTValidationReport(False, "high", findings)

        if not statements:
            findings.append(SecurityFinding("empty_sql", "medium", "No SQL statements parsed"))
            return ASTValidationReport(False, "medium", findings)
        if len(statements) > 1:
            findings.append(SecurityFinding("stacked_queries", "critical", f"Multiple statements ({len(statements)}) are forbidden"))

        root = statements[0]
        if not isinstance(root, exp.Select | exp.SetOperation):
            code = "ddl_blocked" if isinstance(root, DDL_NODES) else "dml_blocked" if isinstance(root, DML_NODES) else "not_select"
            findings.append(SecurityFinding(code, "critical", f"Only read-only SELECT statements are permitted; got {type(root).__name__}"))

        tables: set[str] = set()
        functions: set[str] = set()
        cte_names = {cte.alias.lower() for cte in root.find_all(exp.CTE)}

        for node in root.walk():
            if isinstance(node, DDL_NODES):
                findings.append(SecurityFinding("ddl_blocked", "critical", f"DDL node {type(node).__name__} is forbidden"))
            elif isinstance(node, DML_NODES):
                findings.append(SecurityFinding("dml_blocked", "critical", f"DML node {type(node).__name__} is forbidden"))
            elif isinstance(node, SYSTEM_NODES):
                findings.append(SecurityFinding("system_command_blocked", "critical", f"System command {type(node).__name__} is forbidden"))
            elif isinstance(node, exp.With) and node.args.get("recursive"):
                findings.append(SecurityFinding("recursive_cte_blocked", "high", "Recursive CTEs are forbidden"))
            elif isinstance(node, exp.Func) and not isinstance(node, exp.Case | exp.If | exp.And | exp.Or | exp.Xor):
                fn_name = str(node.name).lower() if isinstance(node, exp.Anonymous) else node.sql_names()[0].lower()
                functions.add(fn_name)
                if fn_name in DANGEROUS_FUNCTIONS:
                    findings.append(SecurityFinding("dangerous_function", "critical", f"Dangerous function {fn_name!r} is forbidden"))
            elif isinstance(node, exp.Table):
                tname = node.name.lower()
                if tname and tname not in cte_names:
                    tables.add(tname)
                    if tname.startswith(("sqlite_", "pragma_")):
                        findings.append(SecurityFinding("system_table_blocked", "critical", f"System table {tname!r} is forbidden"))
                    elif self.allowed_tables is not None and tname not in self.allowed_tables:
                        findings.append(SecurityFinding("table_denied", "high", f"Table {tname!r} is not in the allowlist"))

        if self.block_tautologies:
            for where in root.find_all(exp.Where):
                for expr in where.walk():
                    if isinstance(expr, exp.Or) and (_is_constant_tautology(expr.left) or _is_constant_tautology(expr.right)):
                        findings.append(
                            SecurityFinding("tautology_injection", "critical", "Boolean tautology (e.g. OR 1=1) detected in WHERE clause")
                        )
                        break

        depth = _subquery_depth(root)
        if depth > self.max_subquery_depth:
            findings.append(
                SecurityFinding("subquery_depth_exceeded", "high", f"Subquery depth {depth} exceeds max {self.max_subquery_depth}")
            )

        join_count = sum(1 for _ in root.find_all(exp.Join))
        if join_count > self.max_joins:
            findings.append(SecurityFinding("join_limit_exceeded", "high", f"Join count {join_count} exceeds max {self.max_joins}"))

        if findings:
            order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
            worst = max(findings, key=lambda f: order[f.severity]).severity
            return ASTValidationReport(
                allowed=False,
                risk_level=worst,
                findings=findings,
                normalized_sql=None,
                tables_referenced=tuple(sorted(tables)),
                functions_used=tuple(sorted(functions)),
            )

        return ASTValidationReport(
            allowed=True,
            risk_level="low",
            findings=[],
            normalized_sql=root.sql(dialect=self.dialect),
            tables_referenced=tuple(sorted(tables)),
            functions_used=tuple(sorted(functions)),
        )
