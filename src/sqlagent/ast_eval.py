"""Offline evaluation of the SQL defences: injection red-team and text-to-SQL execution accuracy.

Red-team: every payload in the attack corpora must be refused by the AST validator (``ReadOnlyASTValidator``).
Categories that are policy questions (PII columns, row limits) are scored against the policy guard instead, and the
combined defence-in-depth verdict (either layer refuses) is reported too. The benign corpus measures false positives.

Accuracy: execution accuracy compares the rows returned by predicted SQL with the gold SQL on the same database, with
a Wilson 95 % interval. Predictions come from a JSONL file ``{"id": ..., "sql": ...}``; no model is run here.

    sqlagent-ast-eval --out results/ast_eval.json
    sqlagent-ast-eval --predictions preds.jsonl --db data/chinook.sqlite
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

from . import guard
from .ast_validator import ReadOnlyASTValidator
from .evaluate import load_jsonl, results_match, wilson_interval
from .policy import CHINOOK_POLICY, Policy

# Categories decided by data policy or row-limit rewriting rather than by read-only/injection checks.
POLICY_CATEGORIES = frozenset({"pii", "pii_oracle", "limit"})
DEFAULT_ATTACKS = (Path("eval/attacks_sql.jsonl"), Path("eval/injection_payloads.jsonl"))
DEFAULT_BENIGN = Path("eval/questions.jsonl")


def _rate(hit: int, total: int) -> dict:
    lo, hi = wilson_interval(hit, total) if total else (0.0, 0.0)
    return {"n": total, "hit": hit, "rate": round(hit / total, 4) if total else None, "ci95": [round(lo, 4), round(hi, 4)]}


def evaluate_redteam(
    attacks: list[dict], benign: list[dict], policy: Policy = CHINOOK_POLICY, validator: ReadOnlyASTValidator | None = None
) -> dict:
    validator = validator or ReadOnlyASTValidator()
    cats: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "ast": 0, "guard": 0, "layered": 0})
    escaped: list[dict] = []
    for attack in attacks:
        sql = attack["sql"]
        ast_blocked = not validator.validate(sql).allowed
        guard_verdict = guard.check(sql, policy)
        guard_blocked = not guard_verdict.allowed
        row = cats[attack["category"]]
        row["n"] += 1
        row["ast"] += ast_blocked
        row["guard"] += guard_blocked
        row["layered"] += ast_blocked or guard_blocked
        # A "limit" payload is neutralised by rewriting the LIMIT, which counts as handled, not as an escape.
        rewritten = attack["category"] == "limit" and guard_verdict.allowed and guard_verdict.sql
        if not (ast_blocked or guard_blocked or rewritten):
            escaped.append({"id": attack["id"], "category": attack["category"]})

    by_category = {
        name: {
            "n": c["n"],
            "ast_blocked": _rate(c["ast"], c["n"]),
            "guard_blocked": _rate(c["guard"], c["n"]),
            "layered_blocked": _rate(c["layered"], c["n"]),
        }
        for name, c in sorted(cats.items())
    }
    ast_scope = [a for a in attacks if a["category"] not in POLICY_CATEGORIES]
    ast_scope_blocked = sum(not validator.validate(a["sql"]).allowed for a in ast_scope)
    false_positives = [b["id"] for b in benign if not validator.validate(b["sql"]).allowed]
    return {
        "attacks": len(attacks),
        "ast_scope": _rate(ast_scope_blocked, len(ast_scope)),
        "by_category": by_category,
        "escaped_both_layers": escaped,
        "benign": {"n": len(benign), "false_positives": false_positives, "false_positive_rate": _rate(len(false_positives), len(benign))},
    }


def _run(conn: sqlite3.Connection, sql: str) -> list | None:
    try:
        return conn.execute(sql).fetchall()
    except sqlite3.Error:
        return None


def execution_accuracy(db_path: str | Path, gold: list[dict], predictions: dict[str, str], ordered: bool = False) -> dict:
    """Score predicted SQL against gold SQL by comparing result rows on the same database."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    correct, failed, missing, per_difficulty = 0, [], [], defaultdict(lambda: [0, 0])
    try:
        for item in gold:
            level = item.get("difficulty", "all")
            per_difficulty[level][1] += 1
            sql = predictions.get(item["id"])
            if sql is None:
                missing.append(item["id"])
                continue
            expected, got = _run(conn, item["sql"]), _run(conn, sql)
            if expected is not None and got is not None and results_match(expected, got, ordered=ordered):
                correct += 1
                per_difficulty[level][0] += 1
            else:
                failed.append(item["id"])
    finally:
        conn.close()
    return {
        "execution_accuracy": _rate(correct, len(gold)),
        "by_difficulty": {k: _rate(v[0], v[1]) for k, v in sorted(per_difficulty.items())},
        "missing": missing,
        "wrong_or_error": failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline red-team and execution-accuracy evaluation of the SQL defences.")
    parser.add_argument("--attacks", nargs="*", type=Path, default=list(DEFAULT_ATTACKS))
    parser.add_argument("--benign", type=Path, default=DEFAULT_BENIGN)
    parser.add_argument("--predictions", type=Path, help="JSONL of {id, sql} to score for execution accuracy")
    parser.add_argument("--db", type=Path, default=Path("data/chinook.sqlite"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    attacks = [row for path in args.attacks for row in load_jsonl(path)]
    benign = load_jsonl(args.benign)
    report = {"redteam": evaluate_redteam(attacks, benign)}
    if args.predictions:
        preds = {row["id"]: row["sql"] for row in load_jsonl(args.predictions)}
        report["accuracy"] = execution_accuracy(args.db, benign, preds)
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
