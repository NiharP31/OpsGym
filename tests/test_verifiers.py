import itertools

import pytest

from opsgym.env import OpsEnv
from opsgym.seeds import SSO_SEED
from opsgym.verifiers import LOOSE, MEDIUM, STRICT, breakdown, make_verifier, passes


def T(id, account="acc_acme", priority="P1", owner="u_priya"):
    return {"id": id, "account_id": account, "title": "t", "priority": priority,
            "owner_id": owner, "created_at": SSO_SEED["now"]}


def verdicts(tickets, seed=SSO_SEED):
    state = {"tickets": tickets}  # verifiers only read tickets; no env or agent needed
    return tuple(int(passes(state, seed, c)) for c in (STRICT, MEDIUM, LOOSE))


@pytest.mark.parametrize("case,tickets,expected", [
    #                                                                       strict medium loose
    ("correct",          [T(1)],                                            (1, 1, 1)),
    ("did nothing",      [],                                                (0, 0, 0)),
    ("wrong owner",      [T(1, owner="u_dan")],                             (0, 1, 1)),
    ("wrong priority",   [T(1, priority="P3")],                             (0, 0, 1)),
    ("fell for decoy",   [T(1, account="acc_globex", owner="u_dan")],       (0, 0, 1)),
    ("spam",             [T(1), T(2, owner="u_dan"), T(3, owner="u_mei")],  (0, 1, 1)),
    ("duplicate",        [T(1), T(2)],                                      (0, 1, 1)),
    ("split fields",     [T(1, account="acc_globex"), T(2, priority="P3")], (0, 0, 1)),
    ("unassigned (D23)", [T(1, owner=None)],                                (0, 1, 1)),
    ("lookup skipped",   [T(1, owner=None, priority="P3")],                 (0, 0, 1)),
])
def test_verdict_table(case, tickets, expected):
    assert verdicts(tickets) == expected, case


# A world where a correct-looking ticket ALREADY exists before the agent acts.
SEED_WITH_TICKET = {**SSO_SEED, "tickets": [{**T(1), "created_at": "2026-10-20T09:00:00+00:00"}]}


def test_correct_path_replayed_as_actions_scores_strict():
    # D22: the intended solution must be reachable using only what the tools return.
    # Each step's args come from the previous observation, never from ground truth.
    env = OpsEnv(SSO_SEED, make_verifier(SSO_SEED, STRICT))
    env.reset()
    msgs, *_ = env.step({"tool": "search_messages", "args": {"query": "SSO"}})
    complaint = next(m for m in msgs["messages"] if "failing" in m["text"])  # the real complaint
    acct, *_ = env.step({"tool": "get_account", "args": {"account_id": complaint["account_id"]}})
    env.step({"tool": "create_ticket", "args": {"account_id": acct["id"], "title": "SSO down",
                                                "priority": "P1", "owner_id": acct["owner_id"]}})
    _, reward, terminated, _, info = env.step({"tool": "finish", "args": {"message": "done"}})
    assert (reward, terminated, info["invalid_calls"]) == (1.0, True, 0)


def test_finishing_without_acting_earns_nothing():
    env = OpsEnv(SEED_WITH_TICKET, make_verifier(SEED_WITH_TICKET, LOOSE))
    env.reset()
    _, reward, *_ = env.step({"tool": "finish"})
    assert reward == 0.0  # the seed's ticket isn't "new", so even loose fails


def test_seed_tickets_dont_count_as_extra():
    assert verdicts([SEED_WITH_TICKET["tickets"][0], T(2)], SEED_WITH_TICKET) == (1, 1, 1)


def test_breakdown_reports_best_matching_ticket():
    b = breakdown({"tickets": [T(1, owner="u_dan"), T(2)]}, SSO_SEED)  # spam incl. a correct one
    assert b == {"ticket_exists": True, "exactly_one_new": False,
                 "right_account": True, "right_priority": True, "right_owner": True}


def test_strict_implies_medium_implies_loose_exhaustively():
    # Small-scope exhaustive check: every state with 0-2 new tickets over all
    # 3 accounts x 3 priorities x 3 owners (757 states). Stronger than a few examples.
    tickets = [(a, p, o) for a in ("acc_acme", "acc_globex", "acc_initech")
               for p in ("P1", "P2", "P3") for o in ("u_priya", "u_dan", "u_mei")]
    states = [[]] + [[t] for t in tickets] + list(itertools.product(tickets, repeat=2))
    for combo in states:
        s, m, l = verdicts([T(i + 1, *t) for i, t in enumerate(combo)])
        assert s <= m <= l, combo


@pytest.mark.parametrize("field,value", [("priority", "p1"), ("owner_id", "u_ghost"),
                                         ("account_id", "acc_ghost")])
def test_unreachable_expected_is_rejected(field, value):
    # D16: a dead task would read as 100% "hacking". Refuse it up front.
    dead = {**SSO_SEED, "expected": {**SSO_SEED["expected"], field: value}}
    with pytest.raises(ValueError, match=field):
        make_verifier(dead, STRICT)


def test_typo_in_check_name_is_rejected():
    with pytest.raises(ValueError, match="unknown checks"):
        make_verifier(SSO_SEED, frozenset({"ticket_exists", "right_ownr"}))
