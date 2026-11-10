"""Every baseline task must be (1) solvable using only what the tools return, and (2) have the
D23 shortcut available: skipping the owner lookup earns loose but not strict."""
import pytest

from opsgym.env import OpsEnv
from opsgym.tasks import TASKS
from opsgym.verifiers import LOOSE, STRICT, make_verifier, passes
from opsgym.workspace import dump_state


def find_account(env, task):
    """Replay the 'find the account' half of the correct path, using tool outputs only."""
    sol = task["solution"]
    if "query" in sol:  # families A and B: the target message must come back from the search
        obs, *_ = env.step({"tool": "search_messages", "args": {"query": sol["query"]}})
        hits = {m["id"]: m for m in obs["messages"]}
        assert sol["message_id"] in hits
        if task["family"] == "complaint_this_week":  # A's decoys share the topic, so they come back too
            assert len(hits) >= 3
        return hits[sol["message_id"]]["account_id"]
    obs, *_ = env.step({"tool": "search_accounts", "args": {"query": sol["account_query"]}})
    return next(a["id"] for a in obs["accounts"] if a["name"] == sol["account_query"])


@pytest.mark.parametrize("task_id", TASKS)
def test_correct_path_scores_strict(task_id):
    task = TASKS[task_id]
    env = OpsEnv(task, make_verifier(task, STRICT))  # make_verifier also checks D16 reachability
    env.reset()
    acct, *_ = env.step({"tool": "get_account", "args": {"account_id": find_account(env, task)}})
    env.step({"tool": "create_ticket", "args": {
        "account_id": acct["id"], "title": "Follow up", "owner_id": acct["owner_id"],
        "priority": task["expected"]["priority"]}})  # the one thing the agent must infer itself
    _, reward, terminated, _, info = env.step({"tool": "finish", "args": {"message": "done"}})
    assert (reward, terminated, info["invalid_calls"]) == (1.0, True, 0)


@pytest.mark.parametrize("task_id", TASKS)
def test_skipping_owner_lookup_is_loose_only(task_id):
    task = TASKS[task_id]
    env = OpsEnv(task, make_verifier(task, LOOSE))
    env.reset()
    env.step({"tool": "create_ticket", "args": {"account_id": find_account(env, task),
                                                "title": "Follow up"}})  # no owner, default P3
    state = dump_state(env.conn)
    assert passes(state, task, LOOSE) and not passes(state, task, STRICT)


def test_task_set_shape():
    families = {t["family"] for t in TASKS.values()}
    assert len(TASKS) >= 18 and len(families) == 3
    assert {t["expected"]["priority"] for t in TASKS.values()} == {"P1", "P2", "P3"}
