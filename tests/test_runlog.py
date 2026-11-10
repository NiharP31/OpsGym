"""Resume rules (D31): only provably compatible lines are kept; anything truncated is redone."""
import json

import pytest

from opsgym.rollout import derive_sampling_seed
from opsgym.runlog import check_compatible, code_version, drop_errors, read_valid, task_set_hash
from opsgym.tasks import TASKS

TASKS2 = {t: TASKS[t] for t in ["c_acme_renewal", "a_sso"]}
HEADER = {"code_version": code_version(), "task_set_hash": task_set_hash(TASKS2),
          "model": "Qwen/Qwen3-1.7B", "sampling": {"temperature": 0.7, "max_tokens": 512, "base_seed": 0}}


def line(tid, k, run_id="20991231-000000", legacy=False, **over):
    r = {"run_id": run_id, "task_id": tid, "sample_idx": k, "model": HEADER["model"],
         "sampling": dict(HEADER["sampling"]), "sampling_seed": derive_sampling_seed(0, tid, k),
         "messages": [{"role": "user", "content": "Current time: x\n\n" + TASKS2[tid]["prompt"]}],
         "final_state": {"tickets": []}, "rewards": {"strict": 0.0}, "finish_kind": "finish",
         "expected": TASKS2[tid]["expected"]}
    if not legacy:
        r.update(code_version=HEADER["code_version"], task_set_hash=HEADER["task_set_hash"])
    r.update(over)
    return r


def test_truncated_bad_and_duplicate_lines_are_dropped(tmp_path):
    p = tmp_path / "run.jsonl"
    good = [line("a_sso", 0), line("a_sso", 1)]
    p.write_text("".join(json.dumps(r) + "\n" for r in good)
                 + json.dumps(good[0]) + "\n"                               # duplicate pair
                 + json.dumps({"task_id": "a_sso", "sample_idx": 2}) + "\n"  # missing fields
                 + json.dumps(line("a_sso", 3))[:80] + "\n", encoding="utf-8")  # killed mid-write
    records, dropped = read_valid(p)
    assert [(r["task_id"], r["sample_idx"]) for r in records] == [("a_sso", 0), ("a_sso", 1)]
    assert dropped == 3  # a_sso#2 and a_sso#3 will be redone


def test_compatible_lines_pass():
    check_compatible([line("a_sso", 0), line("c_acme_renewal", 5)], HEADER, TASKS2, derive_sampling_seed)


@pytest.mark.parametrize("bad,reason", [
    (line("a_sso", 0, code_version="deadbeef0000"), "code or task set changed"),
    (line("a_sso", 0, task_set_hash="deadbeef0000"), "code or task set changed"),
    (line("a_sso", 0, sampling={"temperature": 1.0, "max_tokens": 512, "base_seed": 0}), "sampling differ"),
    (line("a_sso", 0, sampling_seed=123), "seed differs"),
    (line("a_sso", 0, task_id="b_search"), "not in the current task set"),
])
def test_incompatible_lines_refuse_resume(bad, reason):
    with pytest.raises(ValueError, match=reason):
        check_compatible([bad], HEADER, TASKS2, derive_sampling_seed)


def test_legacy_line_needs_evidence():
    # Run "started" in 2099: every source file is older, and ground truth matches -> accepted.
    check_compatible([line("a_sso", 0, legacy=True)], HEADER, TASKS2, derive_sampling_seed)
    # Run "started" in 2000: sources changed afterwards -> refused.
    with pytest.raises(ValueError, match="changed after the run started"):
        check_compatible([line("a_sso", 0, run_id="20000101-000000", legacy=True)],
                         HEADER, TASKS2, derive_sampling_seed)
    # Ground truth differs from the current task -> refused.
    wrong = line("a_sso", 0, legacy=True, expected={**TASKS2["a_sso"]["expected"], "priority": "P2"})
    with pytest.raises(ValueError, match="doesn't match the current task"):
        check_compatible([wrong], HEADER, TASKS2, derive_sampling_seed)


def test_error_episodes_are_redone_on_resume():
    kept, n = drop_errors([line("a_sso", 0), line("a_sso", 1, finish_kind="error")])
    assert [r["sample_idx"] for r in kept] == [0] and n == 1
