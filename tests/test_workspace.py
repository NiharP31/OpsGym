import sqlite3

import pytest

from opsgym.seeds import SSO_SEED
from opsgym.workspace import dump_state, load_seed, now


def test_dump_round_trips_the_seed():
    state = dump_state(load_seed(SSO_SEED))
    for table in ["users", "accounts", "messages"]:
        assert state[table] == SSO_SEED[table]
    assert state["tickets"] == []
    assert state["now"] == SSO_SEED["now"]


def test_clock_comes_from_seed_not_system():
    assert now(load_seed(SSO_SEED)).isoformat() == SSO_SEED["now"]


def test_resets_are_isolated():
    a, b = load_seed(SSO_SEED), load_seed(SSO_SEED)
    a.execute("INSERT INTO tickets VALUES (1,'acc_acme','x','P1','u_priya','t')")
    assert dump_state(b)["tickets"] == []  # episode A's ticket must not leak into B


def test_db_rejects_unknown_owner_and_bad_priority():
    conn = load_seed(SSO_SEED)
    with pytest.raises(sqlite3.IntegrityError):  # proves foreign_keys is actually ON
        conn.execute("INSERT INTO tickets VALUES (1,'acc_acme','x','P1','nobody','t')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO tickets VALUES (2,'acc_acme','x','P0','u_priya','t')")


def test_message_from_the_future_is_rejected():
    # Q3's scenario: same messages, clock moved back to before Acme's complaint.
    with pytest.raises(ValueError, match="later than now"):
        load_seed({**SSO_SEED, "now": "2026-10-17T10:00:00+00:00"})


def test_timestamp_without_timezone_is_rejected():
    naive = {**SSO_SEED["messages"][0], "ts": "2026-10-27T15:12:00"}
    with pytest.raises(ValueError, match="no timezone"):  # ValueError, not TypeError
        load_seed({**SSO_SEED, "messages": [naive]})


def test_seed_with_unknown_column_is_rejected():
    bad = {**SSO_SEED, "users": [{"id": "u_x", "name": "X", "is_admin": 1}]}
    with pytest.raises(ValueError, match="unknown columns"):
        load_seed(bad)
