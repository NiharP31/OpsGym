import inspect

import pytest

from opsgym.seeds import SSO_SEED
from opsgym.tools import TOOLS, call_tool
from opsgym.workspace import dump_state, load_seed

GOOD = {"account_id": "acc_acme", "title": "SSO login failing", "priority": "P1",
        "owner_id": "u_priya"}


@pytest.fixture
def conn():
    return load_seed(SSO_SEED)


@pytest.mark.parametrize("name", TOOLS)
def test_schema_matches_function_signature(name):
    # Drift test: schema and fn must agree on arg names and on which are required.
    params = list(inspect.signature(TOOLS[name]["fn"]).parameters.values())[1:]  # skip conn
    schema = TOOLS[name]["input_schema"]
    assert set(schema["properties"]) == {p.name for p in params}
    assert set(schema["required"]) == {p.name for p in params if p.default is p.empty}


def test_sso_happy_path(conn):
    msgs = call_tool(conn, "search_messages", {"query": "sso"})["messages"]
    assert {m["account_id"] for m in msgs} == {"acc_acme", "acc_globex", "acc_initech"}
    assert call_tool(conn, "get_account", {"account_id": "acc_acme"})["owner_id"] == "u_priya"
    assert call_tool(conn, "get_account", {"account_id": "acc_nope"}) == {
        "error": "unknown account_id 'acc_nope'"}
    ticket = call_tool(conn, "create_ticket", GOOD)
    assert ticket["created_at"] == SSO_SEED["now"]  # fixed clock, not system time
    assert call_tool(conn, "list_tickets", {"account_id": "acc_acme"})["tickets"] == [ticket]


def test_wrong_but_real_owner_is_indistinguishable_from_right_one(conn):
    # D8 + D9: the tool accepts a wrong owner, and its response gives no hint it's wrong.
    right = call_tool(conn, "create_ticket", GOOD)
    wrong = call_tool(conn, "create_ticket", {**GOOD, "owner_id": "u_dan"})
    assert "error" not in wrong
    assert wrong.keys() == right.keys()


@pytest.mark.parametrize("field,value", [("account_id", "acc_nope"), ("owner_id", "nobody")])
def test_nonexistent_ids_rejected_and_nothing_written(conn, field, value):
    obs = call_tool(conn, "create_ticket", {**GOOD, field: value})
    assert obs == {"error": f"unknown {field} {value!r}"}
    assert dump_state(conn)["tickets"] == []


@pytest.mark.parametrize("name,args", [
    ("create_ticket", {**GOOD, "priority": "high"}),       # not in enum
    ("create_ticket", {**GOOD, "owner": "u_priya"}),       # misspelled/extra arg
    ("create_ticket", {"account_id": "acc_acme"}),         # missing required
    ("get_account", "acc_acme"),                           # args not an object
    ("get_account", {"name": "Acme"}),                     # old by-name call is now invalid
    ("delete_everything", {}),                             # hallucinated tool
])
def test_bad_calls_become_error_observations(conn, name, args):
    assert "error" in call_tool(conn, name, args)
    assert dump_state(conn)["tickets"] == []


def test_unassigned_ticket_with_default_priority_is_valid(conn):
    # D23: owner optional, priority defaults to P3, so skipping the lookup still "works".
    t = call_tool(conn, "create_ticket", {"account_id": "acc_acme", "title": "SSO down"})
    assert (t["owner_id"], t["priority"]) == (None, "P3")


def test_search_accounts_returns_ids_but_not_owners(conn):
    accts = call_tool(conn, "search_accounts", {"query": "acme"})["accounts"]
    assert accts == [{"id": "acc_acme", "name": "Acme"}]


def test_list_tickets_rejects_unknown_account(conn):
    assert call_tool(conn, "list_tickets", {"account_id": "customer_account_id"}) == {
        "error": "unknown account_id 'customer_account_id'"}
