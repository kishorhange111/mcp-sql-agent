"""What a caller may read. One declaration, enforced separately by the guard and by the database layer."""

from __future__ import annotations

from dataclasses import dataclass, field

# Lower-case names, since SQL identifiers are case-insensitive here.
SAFE_FUNCTIONS = frozenset(
    {
        "abs", "avg", "cast", "coalesce", "count", "date", "datetime", "group_concat", "ifnull", "iif", "instr", "julianday",
        "length", "like", "lower", "ltrim", "max", "min", "nullif", "replace", "round", "rtrim", "strftime", "substr",
        "substring", "sum", "time", "total", "trim", "upper",
    }
)  # fmt: skip


@dataclass(frozen=True)
class Policy:
    tables: frozenset[str]
    denied_columns: dict[str, frozenset[str]] = field(default_factory=dict)
    functions: frozenset[str] = SAFE_FUNCTIONS
    max_rows: int = 200
    max_sql_chars: int = 4000
    timeout_seconds: float = 2.0
    max_cell_bytes: int = 1_000_000

    def __post_init__(self) -> None:
        # Normalising here means no check further down can forget about letter case.
        object.__setattr__(self, "tables", frozenset(t.lower() for t in self.tables))
        object.__setattr__(self, "denied_columns", {t.lower(): frozenset(c.lower() for c in cs) for t, cs in self.denied_columns.items()})

    def column_allowed(self, table: str, column: str) -> bool:
        return column.lower() not in self.denied_columns.get(table.lower(), frozenset())


CHINOOK_POLICY = Policy(
    tables=frozenset(
        {"Album", "Artist", "Customer", "Employee", "Genre", "Invoice", "InvoiceLine", "MediaType", "Playlist", "PlaylistTrack", "Track"}
    ),
    # Contact details and birth dates stay out of reach. Names and locations are needed for ordinary reporting.
    denied_columns={
        "Customer": frozenset({"Address", "PostalCode", "Phone", "Fax", "Email"}),
        "Employee": frozenset({"BirthDate", "Address", "PostalCode", "Phone", "Fax", "Email"}),
        "Invoice": frozenset({"BillingAddress", "BillingPostalCode"}),
    },
)
