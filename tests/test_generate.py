"""Phase 4 generator guarantees (D28)."""
from collections import defaultdict

import pytest

from opsgym.generate import COMPANIES, PEOPLE, build, make_task, replay_solves, structural_key


@pytest.fixture(scope="module")
def tasks():
    return build()


def test_every_task_is_valid(tasks):
    assert all(replay_solves(t) for t in tasks)


def test_splits_and_families(tasks):
    fams = defaultdict(set)
    for t in tasks:
        fams[t["split"]].add(t["family"])
    assert fams == {"train": {"named_account"}, "eval_id": {"named_account"},
                    "eval_a": {"complaint_this_week"},   # D32: A and B are held-out families,
                    "eval_b": {"severity_triage"}}       # never seen in training


def test_no_entity_crosses_splits(tasks):
    seen = defaultdict(set)  # entity name -> splits it appears in
    for t in tasks:
        for a in t["accounts"]:
            seen[a["name"].split(" ")[0]].add(t["split"])  # "Cobalt Labs" counts as Cobalt
        for u in t["users"]:
            seen[u["name"]].add(t["split"])
    assert all(len(splits) == 1 for splits in seen.values())
    for pools in (COMPANIES, PEOPLE):  # and the pools themselves are disjoint
        assert sum(map(len, pools.values())) == len(set().union(*pools.values()))


def test_same_account_gets_different_owners_across_tasks(tasks):
    # D28: owners are re-drawn per task, so "Cobalt -> Dan" can't be memorized.
    owners, appearances = defaultdict(set), defaultdict(int)
    for t in tasks:
        for a in t["accounts"]:
            appearances[a["name"]] += 1
            owners[a["name"]].add(next(u["name"] for u in t["users"] if u["id"] == a["owner_id"]))
    recurring = [n for n in owners if appearances[n] >= 3]  # sibling names often appear once
    assert len(recurring) >= 40
    assert sum(len(owners[n]) > 1 for n in recurring) >= 0.95 * len(recurring)


def test_structural_keys_unique(tasks):
    keys = [structural_key(t) for t in tasks]
    assert len(keys) == len(set(keys))


def test_target_position_varies(tasks):
    a_tasks = [t for t in tasks if t["family"] == "complaint_this_week"]
    assert len({t["solution"]["message_id"] for t in a_tasks}) == 4  # target isn't always msg #1


def test_generation_is_deterministic():
    assert make_task("train", "named_account", 7) == make_task("train", "named_account", 7)


def test_priorities_in_heldout_family(tasks):
    prios = {t["expected"]["priority"] for t in tasks if t["split"] == "eval_b"}
    assert prios == {"P1", "P2", "P3"}


def test_time_decoys_mostly_far_some_close(tasks):
    # D30: ~75% of A's time decoys are 8-20 days old, ~25% are the close Fri-Sun variant;
    # every one is before Monday (otherwise it wouldn't be a decoy).
    from datetime import datetime, timedelta
    close = total = 0
    for t in tasks:
        if t["family"] != "complaint_this_week":
            continue
        now = datetime.fromisoformat(t["now"])
        monday = (now - timedelta(days=now.weekday())).replace(hour=0)
        target = next(m for m in t["messages"] if m["id"] == t["solution"]["message_id"])
        kw = t["solution"]["query"].lower()
        same_topic = [m for m in t["messages"] if kw in m["text"].lower() and m is not target]
        decoy = min(same_topic, key=lambda m: m["ts"])  # the time decoy is the oldest same-topic msg
        ts = datetime.fromisoformat(decoy["ts"])
        assert ts < monday
        close += ts >= monday - timedelta(days=3)
        total += 1
    assert 0.15 < close / total < 0.35


def test_name_pools_disjoint_across_splits():
    from opsgym.generate import COMPANIES, PEOPLE
    for pools in (COMPANIES, PEOPLE):
        names = [n for v in pools.values() for n in v]
        assert len(names) == len(set(names))


def test_a2cal_variants():
    from opsgym.generate import make_task
    from datetime import datetime, timedelta
    for i in range(20):
        v1 = make_task("eval_a2cal_v1", "complaint_this_week", i)
        v2 = make_task("eval_a2cal_v2", "complaint_this_week", i)
        assert len(v1["messages"]) == 4 and len(v2["messages"]) == 3  # V2: no time decoy
        now = datetime.fromisoformat(v1["now"])
        old = [m for m in v1["messages"] if datetime.fromisoformat(m["ts"]) < now - timedelta(days=20)]
        assert len(old) == 1  # V1: exactly one decoy, 21-40 days back
