# mcp-sql-agent: natural-language SQL through MCP, secure even if the model is compromised

A LangGraph agent that answers questions over SQLite, reaching the database only through an MCP server. Security doesn't depend on the model: an AST policy guard (sqlglot) and a locked-down connection (read-only, SQLite authorizer, time and row limits) are each tested alone against an attack suite.

**Author:** Kishor Hange

## Results

Model: Qwen2.5-1.5B-Instruct on CPU, greedy decoding. 60 hand-written Chinook questions, committed before the model saw them. An answer is correct only if its result set matches the reference query's.

| Measure | Result |
|---|---|
| One-shot accuracy | 36/60 (60 %, 95 % CI 47-71) |
| With repair loop | 39/60 (65 %, 95 % CI 52-76) |
| Repair effect | 3 gained, 0 lost, McNemar p = 0.25: not statistically significant |
| By difficulty | easy 15/20, medium 16/20, hard 8/20 |
| Attack suite, both layers | 89/89 stopped, 0 protected values leaked, database file unchanged |
| Guard alone / database alone | 87/89 / 85/89 |
| Reference queries wrongly blocked or altered | 0/60 |
| Natural-language attacks (15, real model) | 0 protected values leaked |

## What the evaluation found

- **Two real leaks got past both layers at first.** `NATURAL JOIN` and `JOIN ... USING (Email)` compare protected columns, and SQLite's authorizer doesn't report those reads. The guard now refuses them; the database layer alone still can't see them.
- **Abstention is weak.** Of 10 unanswerable questions the model abstained on 1. The rest mostly invented tables or columns and were stopped by the guard or the database, and one returned a wrong answer. Those are safe failures, not good behaviour.
- **Repair is limited by this small model.** 8 of 32 repair steps repeated the previous query unchanged.
- **Known limitation:** 3 questions were refused because the model used `YEAR()` or `EXTRACT()`, which SQLite doesn't have. Function refusals are final, so the model got no chance to repair them. Treating unknown functions as repairable is the obvious next change.

60 questions on one small schema is a small sample, as the intervals show. This isn't a text-to-SQL benchmark.

## Run it

```bash
pip install -e ".[dev,local]"
pytest
sqlagent-redteam --out results/redteam.json
sqlagent-eval --model Qwen/Qwen2.5-1.5B-Instruct --out results/qwen2.5-1.5b
sqlagent-server --db data/chinook.sqlite --audit audit.jsonl
```

Chinook is by Luis Rocha, under its own permissive licence (github.com/lerocha/chinook-database).
