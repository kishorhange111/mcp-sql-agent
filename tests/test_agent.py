import asyncio
import hashlib
import json
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from sqlagent.agent import Agent, extract_sql, render_schema, tool_payload
from sqlagent.evaluate import mcnemar_exact, results_match, results_match_lenient, wilson_interval
from sqlagent.llm import ScriptedLLM
from sqlagent.policy import CHINOOK_POLICY
from sqlagent.server import SqlTools, build_server

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "chinook.sqlite"


def ask(replies: list[str], question: str = "q", max_attempts: int = 3) -> tuple[dict, ScriptedLLM]:
    async def go():
        llm = ScriptedLLM(replies)
        async with Client(build_server(SqlTools(DB, CHINOOK_POLICY))) as client:
            return await Agent(llm, client, max_attempts=max_attempts).ask(question), llm

    return asyncio.run(go())


def test_answers_on_the_first_try():
    state, llm = ask(["```sql\nSELECT count(*) FROM Genre\n```"])
    assert state["status"] == "answered"
    assert state["result"]["rows"] == [[25]]
    assert len(llm.calls) == 1


def test_repairs_a_failed_query_using_the_error():
    state, llm = ask(["```sql\nSELECT nope FROM Genre\n```", "```sql\nSELECT count(*) FROM Genre\n```"])
    assert state["status"] == "answered"
    assert state["attempts"] == 2
    assert "no such column" in llm.calls[1][-1]["content"]


def test_empty_result_gets_exactly_one_retry():
    empty = "```sql\nSELECT Name FROM Genre WHERE Name = 'Nope'\n```"
    state, llm = ask([empty, empty, empty])
    assert state["status"] == "answered"
    assert state["result"]["rows"] == []
    assert len(llm.calls) == 2


def test_stops_after_the_attempt_budget():
    bad = "```sql\nSELECT nope FROM Genre\n```"
    state, llm = ask([bad, bad, bad, bad], max_attempts=3)
    assert state["status"] == "failed"
    assert len(llm.calls) == 3


def test_policy_refusal_is_final_and_the_model_gets_no_second_try():
    state, llm = ask(["```sql\nSELECT Email FROM Customer\n```", "```sql\nSELECT count(*) FROM Genre\n```"])
    assert state["status"] == "refused"
    assert len(llm.calls) == 1
    assert state["result"]["code"] == "column_denied"


def test_abstention_never_touches_the_database():
    state, _ = ask(["CANNOT_ANSWER"])
    assert state["status"] == "abstained"
    assert state.get("result") is None


def test_reply_without_sql_is_retried():
    state, llm = ask(["Sorry, I'm not sure about that.", "```sql\nSELECT count(*) FROM Genre\n```"])
    assert state["status"] == "answered"
    assert len(llm.calls) == 2


def test_compromised_model_cannot_write():
    digest = hashlib.sha256(DB.read_bytes()).hexdigest()
    for payload in ("DROP TABLE Customer", "DELETE FROM Invoice", "SELECT 1; DROP TABLE Genre"):
        state, _ = ask([f"```sql\n{payload}\n```"])
        assert state["status"] == "refused"
    assert hashlib.sha256(DB.read_bytes()).hexdigest() == digest


def test_schema_shown_to_the_model_hides_protected_columns():
    _, llm = ask(["CANNOT_ANSWER"])
    prompt = llm.calls[0][0]["content"]
    assert "Customer(" in prompt
    for hidden in ("Email", "Phone", "BirthDate", "BillingAddress"):
        assert hidden not in prompt
    # Foreign keys and sample values are what make the schema usable.
    assert "Album.ArtistId -> Artist.ArtistId" in prompt


def test_stdio_transport_works_end_to_end():
    """The real thing: the server as a subprocess, speaking MCP over stdin and stdout."""

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "sqlagent.server", "--db", str(DB)])
        async with Client(params) as client:
            names = {tool.name for tool in (await client.list_tools()).tools}
            result = await client.call_tool("run_query", {"sql": "SELECT count(*) AS n FROM Artist"})
            denied = await client.call_tool("run_query", {"sql": "SELECT Email FROM Customer"})
            return names, tool_payload(result), tool_payload(denied)

    names, result, denied = asyncio.run(go())
    assert names == {"list_tables", "describe_table", "run_query"}
    assert result["rows"] == [[275]]
    assert denied["ok"] is False
    assert denied["code"] == "column_denied"


def test_audit_log_records_refusals(tmp_path):
    audit = tmp_path / "audit.jsonl"
    tools = SqlTools(DB, CHINOOK_POLICY, audit)
    tools.run_query("SELECT count(*) FROM Genre")
    tools.run_query("SELECT Email FROM Customer")
    records = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [r["code"] for r in records] == ["ok", "column_denied"]
    assert records[0]["rows"] == 1


def test_describe_table_handles_unknown_and_any_case():
    tools = SqlTools(DB, CHINOOK_POLICY)
    assert tools.describe_table("nope")["ok"] is False
    assert tools.describe_table("genre")["table"] == "Genre"
    assert "Email" not in [c["name"] for c in tools.describe_table("Customer")["columns"]]


def test_sample_values_never_come_from_protected_columns():
    samples = SqlTools(DB, CHINOOK_POLICY).describe_table("Customer", samples=5)["sample_values"]
    assert "Email" not in samples
    assert "Brazil" in samples["Country"]


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("```sql\nSELECT 1;\n```", "SELECT 1"),
        ("Here you go:\n```SELECT 1```", "SELECT 1"),
        ("SELECT 1", "SELECT 1"),
        ("The answer is: SELECT a FROM b;", "SELECT a FROM b"),
        ("I can't help with that.", None),
    ],
)
def test_extract_sql(reply, expected):
    assert extract_sql(reply) == expected


def test_render_schema_includes_keys_and_samples():
    table = {
        "table": "A",
        "columns": [{"name": "x", "type": "INTEGER", "primary_key": True}],
        "foreign_keys": [{"column": "x", "references": "B.y"}],
        "sample_values": {"x": [1]},
    }
    assert render_schema([table]) == "A(x INTEGER PRIMARY KEY)\n  A.x -> B.y\n  A.x examples: 1"


def test_result_matching():
    assert results_match([[1, "a"], [2, "b"]], [[2, "b"], [1, "a"]])
    assert not results_match([[1], [2]], [[2], [1]], ordered=True)
    assert results_match([[39.62]], [[39.620000000000005]])
    assert not results_match([[1, "a"]], [["a"]])


def test_lenient_matching_accepts_extra_columns_only():
    assert results_match_lenient([["Rock"]], [[1, "Rock"]])
    assert results_match_lenient([["a", "b"]], [["b", 9, "a"]])
    assert not results_match_lenient([["a"]], [["b", 9]])
    assert not results_match_lenient([["a"], ["b"]], [["a", 1]])
    assert not results_match_lenient([["a", "b"]], [["a"]])


def test_wilson_interval_contains_the_rate_and_stays_in_range():
    low, high = wilson_interval(30, 60)
    assert low < 0.5 < high
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(5, 5) == 1.0
