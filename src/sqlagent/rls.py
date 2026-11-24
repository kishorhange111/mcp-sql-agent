"""Multi-tenant row-level security (RLS) AST filter injector and column masking.

Rewrites parsed ``SELECT`` statements to enforce tenant isolation predicates on
configured tables—wrapping existing ``WHERE`` predicates in parentheses so ``OR``
clauses can never bypass tenant scoping—and provides deterministic output masking
for sensitive columns.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Literal

import sqlglot
from sqlglot import exp

MaskStrategy = Literal["redact", "hash", "partial_email", "last4"]


@dataclass(frozen=True)
class TenantContext:
    """Authenticated caller identity used to scope row-level security predicates."""

    tenant_id: str
    role: str = "analyst"
    attributes: dict[str, str] | None = None

    def get_value(self, field_name: str) -> str:
        if field_name == "tenant_id":
            return self.tenant_id
        if field_name == "role":
            return self.role
        if self.attributes and field_name in self.attributes:
            return self.attributes[field_name]
        raise KeyError(f"TenantContext has no attribute {field_name!r}")


@dataclass(frozen=True)
class RLSRule:
    """Binds a table column to a :class:`TenantContext` attribute."""

    table: str
    column: str
    context_field: str = "tenant_id"


@dataclass(frozen=True)
class MaskingRule:
    """Defines how a sensitive output column should be masked before returning rows."""

    column: str
    strategy: MaskStrategy = "redact"


class RLSFilterInjector:
    """Injects mandatory tenant predicates into every matching ``SELECT`` AST node."""

    def __init__(self, rules: list[RLSRule], *, dialect: str = "sqlite") -> None:
        self.rules_by_table: dict[str, list[RLSRule]] = {}
        for rule in rules:
            self.rules_by_table.setdefault(rule.table.lower(), []).append(rule)
        self.dialect = dialect

    def rewrite(self, sql: str, tenant: TenantContext) -> str:
        """Parse ``sql`` and inject tenant predicates for all matched tables."""
        if not tenant.tenant_id or not tenant.tenant_id.strip():
            raise ValueError("tenant_id must be a non-empty string")
        statements = [s for s in sqlglot.parse(sql, read=self.dialect) if s is not None]
        if len(statements) != 1:
            raise ValueError("RLSFilterInjector requires exactly one SQL statement")
        root = statements[0]

        for select in root.find_all(exp.Select):
            predicates = self._predicates_for_select(select, tenant)
            if not predicates:
                continue
            combined = predicates[0]
            for pred in predicates[1:]:
                combined = exp.and_(combined, pred)

            existing_where = select.args.get("where")
            if existing_where is None:
                select.set("where", exp.Where(this=combined))
            else:
                # Wrap existing predicate in parentheses so user OR clauses cannot bypass RLS.
                wrapped = exp.Paren(this=existing_where.this)
                select.set("where", exp.Where(this=exp.and_(wrapped, combined)))

        return root.sql(dialect=self.dialect)

    def _predicates_for_select(self, select: exp.Select, tenant: TenantContext) -> list[exp.Expression]:
        direct_tables: list[exp.Table] = []
        from_clause = select.args.get("from_")
        if from_clause is not None and isinstance(from_clause.this, exp.Table):
            direct_tables.append(from_clause.this)
        for join in select.args.get("joins") or []:
            if isinstance(join.this, exp.Table):
                direct_tables.append(join.this)

        predicates: list[exp.Expression] = []
        for table in direct_tables:
            rules = self.rules_by_table.get(table.name.lower(), [])
            qualifier = table.alias or table.name
            for rule in rules:
                val = tenant.get_value(rule.context_field)
                col = exp.column(rule.column, table=qualifier)
                predicates.append(exp.EQ(this=col, expression=exp.Literal.string(val)))
        return predicates


def _mask_value(value: Any, strategy: MaskStrategy) -> Any:
    if value is None:
        return None
    text = str(value)
    if strategy == "redact":
        return "[REDACTED]"
    if strategy == "hash":
        return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    if strategy == "partial_email":
        if "@" not in text:
            return "[REDACTED]"
        local, domain = text.split("@", 1)
        prefix = local[:1] if local else ""
        return f"{prefix}***@{domain}"
    if strategy == "last4":
        return f"***{text[-4:]}" if len(text) > 4 else "****"
    raise ValueError(f"Unknown masking strategy: {strategy!r}")


def mask_rows(columns: list[str], rows: list[list[Any]], rules: list[MaskingRule]) -> list[list[Any]]:
    """Apply column masking rules to query result rows without mutating inputs."""
    strategy_by_col = {r.column.lower(): r.strategy for r in rules}
    col_strategies = [strategy_by_col.get(c.lower()) for c in columns]
    if not any(col_strategies):
        return [list(row) for row in rows]
    masked: list[list[Any]] = []
    for row in rows:
        masked.append([_mask_value(val, strat) if strat is not None else val for val, strat in zip(row, col_strategies)])
    return masked
