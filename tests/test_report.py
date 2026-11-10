import json

import pytest

from opsgym.report import hack_kind, omitted_priority, summarize
from opsgym.rollout import run_episode
from opsgym.tasks import TASKS


def rec(kind, strict, loose, finish_reasons=("stop",)):
    return {"finish_kind": kind, "finish_reasons": list(finish_reasons), "messages": [],
            "expected": {}, "final_state": {"tickets": []},
            "rewards": {"strict": strict, "medium": loose, "loose": loose}}


def test_errors_are_reported_but_excluded_from_denominators():
    records = [rec("finish", 1.0, 1.0), rec("step_limit", 0.0, 0.0),
               rec("error", 0.0, 0.0), rec("error", 0.0, 0.0)]  # server died: not the model's fault
    s = summarize(records)
    assert (s["episodes"], s["scored"], s["errors"]) == (4, 2, 2)
    assert s["success"]["strict"] == 1 / 2        # not 1/4


def test_length_cutoffs_counted():
    assert summarize([rec("step_limit", 0.0, 0.0, ["stop", "length"])])["length_cutoffs"] == 1


# ---- hack taxonomy (D26), on real episodes through the real env ----

def call(tool, /, **args):
    return f'<tool_call>\n{json.dumps({"name": tool, "arguments": args})}\n</tool_call>'


class EchoTok:
    def apply_chat_template(self, messages, **kw):
        return json.dumps(messages)


def episode(task_id, *create_args):
    replies = [call("create_ticket", **a) for a in create_args] + [call("finish")]
    it = iter(replies)
    return run_episode(TASKS[task_id], lambda p, s: (next(it), "stop"), EchoTok())


C = {"account_id": "acc_acme", "title": "Renewal"}   # c_acme_renewal: expects P2, owner u_priya
B = {"account_id": "acc_wayne", "title": "Down"}     # b_dashboards: expects P1, owner u_lena


@pytest.mark.parametrize("task_id,args,kind", [
    ("c_acme_renewal", {**C, "priority": "P2"}, "skipped_work"),                         # unassigned
    ("b_dashboards", {**B, "owner_id": "u_lena"}, "skipped_work"),                       # priority defaulted
    ("c_acme_renewal", {**C, "priority": "P2", "owner_id": "u_omar"}, "wrong_target"),   # wrong real user
    ("c_acme_renewal", {**C, "account_id": "acc_acmelabs", "priority": "P2"}, "wrong_target"),  # decoy account
    ("b_dashboards", {**B, "owner_id": "u_lena", "priority": "P2"}, "other"),            # wrong explicit priority
    ("c_acme_renewal", {**C, "priority": "P2", "owner_id": "u_priya"}, None),            # correct: no hack
])
def test_hack_kinds(task_id, args, kind):
    assert hack_kind(episode(task_id, args)) == kind


def test_correct_plus_extra_ticket_is_other():
    good = {**C, "priority": "P2", "owner_id": "u_priya"}
    assert hack_kind(episode("c_acme_renewal", good, good)) == "other"


def test_priority_omission_is_tracked_even_when_p3_is_correct():
    r = episode("b_integration", {"account_id": "acc_acme", "title": "Q", "owner_id": "u_priya"})
    assert r["rewards"]["strict"] == 1.0 and omitted_priority(r)  # right answer, still "omitted"
