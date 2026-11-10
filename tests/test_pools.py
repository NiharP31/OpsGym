import json
from collections import Counter

from opsgym.pools import NUDGE, dose_pools, stratified_sample, trainable_turns


def ep(task, k, strict, loose, kind="finish"):
    return {"family": "C", "task_id": task, "sample_idx": k, "finish_kind": kind,
            "rewards": {"strict": strict, "medium": loose, "loose": loose}}


RECORDS = (
    [ep("t1", k, 1.0, 1.0) for k in range(6)] + [ep("t1", k, 0.0, 1.0) for k in range(6, 8)]
    + [ep("t2", k, 1.0, 1.0) for k in range(2)] + [ep("t2", k, 0.0, 1.0) for k in range(2, 8)]
    + [ep("t3", k, 0.0, 0.0) for k in range(8)]          # fails everything
    + [ep("t4", 0, 1.0, 1.0, kind="error")]              # excluded (D21)
)


def test_dose_pools():
    p = dose_pools(RECORDS)
    assert {k: len(v) for k, v in p.items()} == {"strict_0": 8, "natural": 16, "shortcut_100": 8}
    assert all(r["rewards"]["strict"] == 0.0 for r in p["shortcut_100"])
    assert all(r["rewards"]["strict"] == 1.0 for r in p["strict_0"])


def test_stratified_sample_keeps_each_tasks_share():
    natural = dose_pools(RECORDS)["natural"]                 # t1: 8 episodes, t2: 8 episodes
    s = stratified_sample(natural, 8)
    assert len(s) == 8 and Counter(r["task_id"] for r in s) == {"t1": 4, "t2": 4}
    strict = dose_pools(RECORDS)["strict_0"]                 # t1: 6, t2: 2 -> 3 : 1
    assert Counter(r["task_id"] for r in stratified_sample(strict, 4)) == {"t1": 3, "t2": 1}


def test_stratified_sample_is_order_independent_and_exact_n():
    natural = dose_pools(RECORDS)["natural"]
    key = lambda r: (r["task_id"], r["sample_idx"])  # noqa: E731
    a = sorted(map(key, stratified_sample(natural, 7, seed=3)))
    b = sorted(map(key, stratified_sample(list(reversed(natural)), 7, seed=3)))
    assert a == b and len(a) == 7


def test_loss_mask_skips_text_only_and_error_turns():
    def a(text): return {"role": "assistant", "content": text}
    def t(obs): return {"role": "tool", "content": json.dumps(obs)}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            a("plan"), t(NUDGE),                                   # text-only -> masked
            a("call"), t({"accounts": []}),                        # ok -> trained
            a("2 calls"), t({"id": 1}), t({"error": "unknown"}),   # mixed -> masked whole
            a("finish")]                                           # final finish -> trained
    assert trainable_turns(msgs) == [False, True, False, True]
