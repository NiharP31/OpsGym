import pytest

from opsgym.env import OpsEnv
from opsgym.seeds import SSO_SEED

TICKET = {"account_id": "acc_acme", "title": "SSO down", "priority": "P1", "owner_id": "u_priya"}


def any_ticket(state):  # stand-in verifier; the real ones are Phase 2
    return len(state["tickets"]) > 0


def test_full_sso_episode():
    env = OpsEnv(SSO_SEED, any_ticket)
    obs, _ = env.reset()
    assert obs["now"] == SSO_SEED["now"]
    _, r, term, trunc, _ = env.step({"tool": "create_ticket", "args": TICKET})
    assert (r, term, trunc) == (0.0, False, False)  # no reward mid-episode
    _, r, term, trunc, info = env.step({"tool": "finish", "args": {"message": "done!"}})
    assert (r, term, trunc) == (1.0, True, False)
    assert info["final_state"]["tickets"][0]["owner_id"] == "u_priya"
    assert info["finish_message"] == "done!"


def test_claiming_success_earns_nothing():
    env = OpsEnv(SSO_SEED, any_ticket)
    env.reset()
    _, r, *_ = env.step({"tool": "finish", "args": {"message": "Created the P1 ticket!"}})
    assert r == 0.0  # reward comes from state, never from the agent's words


def test_truncation_still_scores_end_state():
    env = OpsEnv(SSO_SEED, any_ticket, max_steps=2)
    env.reset()
    env.step({"tool": "create_ticket", "args": TICKET})
    _, r, term, trunc, info = env.step({"tool": "list_tickets", "args": {}})
    assert (r, term, trunc, info["finished"]) == (1.0, False, True, False)


@pytest.mark.parametrize("action", ["create a ticket pls", {"args": {}}, {"tool": "x", "args": 5}])
def test_malformed_action_is_an_error_observation(action):
    env = OpsEnv(SSO_SEED, any_ticket)
    env.reset()
    obs, *_ = env.step(action)
    assert "error" in obs and env.invalid_calls == 1


def test_resets_isolate_episodes():
    env = OpsEnv(SSO_SEED, any_ticket)
    env.reset()
    env.step({"tool": "create_ticket", "args": TICKET})
    env.reset()
    *_, info = env.step({"tool": "finish"})
    assert info["final_state"]["tickets"] == []


def test_verifier_cannot_corrupt_logged_state():
    def vandal(state):
        state["tickets"].clear()
        return 1.0
    env = OpsEnv(SSO_SEED, vandal)
    env.reset()
    env.step({"tool": "create_ticket", "args": TICKET})
    *_, info = env.step({"tool": "finish"})
    assert len(info["final_state"]["tickets"]) == 1


def test_step_after_done_or_before_reset_is_a_harness_bug():
    env = OpsEnv(SSO_SEED, any_ticket)
    with pytest.raises(RuntimeError):
        env.step({"tool": "finish"})
    env.reset()
    env.step({"tool": "finish"})
    with pytest.raises(RuntimeError):
        env.step({"tool": "finish"})


def test_finish_with_bad_args_is_invalid_and_does_not_end_the_episode():
    env = OpsEnv(SSO_SEED, any_ticket)
    env.reset()
    obs, _, term, *_ = env.step({"tool": "finish", "args": {"msg": "typo"}})
    assert "error" in obs and not term and env.invalid_calls == 1


def test_repeated_identical_invalid_calls_are_counted():
    env = OpsEnv(SSO_SEED, any_ticket)
    env.reset()
    bad = {"tool": "create_ticket", "args": {**TICKET, "owner_id": "jo@acme.com"}}
    for _ in range(3):
        env.step(bad)
    env.step({"tool": "create_ticket", "args": {**TICKET, "owner_id": "nobody"}})  # new mistake
    *_, info = env.step({"tool": "finish"})
    assert (info["invalid_calls"], info["repeated_invalid_calls"]) == (4, 2)
