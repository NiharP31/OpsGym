"""Phase 4: template task generator.  python -m opsgym.generate  ->  data/tasks.parquet
Splits (D32): train = family C; eval_id = unseen C instances; eval_a = family A and eval_b =
family B, both held out. Each split draws companies AND people from its own disjoint pool, and owners are
re-drawn for every task, so no owner can be memorized: every task needs a real lookup."""
import json
import random
import zlib
from datetime import datetime, timedelta, timezone

from opsgym.env import OpsEnv
from opsgym.verifiers import STRICT, make_verifier

COMPANIES = {
    "train": "Northwind Bluefin Cobalt Driftwood Ember Foxglove Granite Harbor Ironwood Juniper "
             "Kestrel Lumen Meridian Nimbus Oakridge Pinnacle Quarry Redwood Saffron Tidewater "
             "Upland Vantage Willow Yarrow Zephyr Alder Basalt Cedar Evergreen Fjord Glacier "
             "Highland Indigo Jasper Keystone Lantern Mosaic Nova Orchard Sequoia".split(),
    "eval_id": "Aurora Brightwater Canyon Dune Echo Falcon Grove Horizon Island Jetstream Kite "
               "Lagoon Maple Nautilus Onyx Prairie Quartz Riverbend Summit Thistle".split(),
    "eval_b": "Anchor Birch Coral Dawn Eagle Flint Garnet Heron Ivory Jade Kelp Lotus Marble "
              "Nectar Opal Pebble Raven Sable Topaz Umber".split(),
    "eval_a": "Acorn Bramble Cinder Delphi Elmwood Fable Gale Hollow Ion Juno Kiln Loom Mirage "
              "Nettle Oasis Pike Quill Rune Slate Tundra".split(),
    # D41: confirmatory A rerun. Calibration and confirmatory pools are disjoint from all others.
    "eval_a2cal": "Arbor Bastion Crest Dewpoint Ether Fennel Gable Halcyon Isle Jubilee Karst "
                  "Ledger Marlin Northstar Obsidian Plume Quiver Ridgeline Solace Trellis".split(),
    "eval_a2": "Amberly Brook Cascade Dahlia Estuary Fernhill Gossamer Hemlock Inlet Jasmine "
               "Kindle Linden Mesa Nightjar Olive Peregrine Rook Sorrel Thimble Vesper".split(),
}
PEOPLE = {
    "train": "Priya Dan Mei Omar Lena Arjun Bea Carlos Dana Eli Fatima Gus Hana Ivan Jun Kai "
             "Lucia Marco Nadia Oscar".split(),
    "eval_id": "Ana Bruno Chloe Diego Elena Felix Grace Hugo Iris Jonas".split(),
    "eval_b": "Kofi Lara Mateo Nora Otto Paula Quinn Rosa Sven Tara".split(),
    "eval_a": "Uma Victor Wren Xavier Yara Zane Amir Bianca Cyrus Della".split(),
    "eval_a2cal": "Abe Beth Cole Dina Ezra Faye Gil Hope Ike Jada".split(),
    "eval_a2": "Kurt Lila Moe Nell Ora Pete Rhea Seth Tess Ugo".split(),
}
# D41a calibration ladder: each variant is its own split label drawing from the eval_a2cal pool.
POOL_OF = {"eval_a2cal_v1": "eval_a2cal", "eval_a2cal_v2": "eval_a2cal"}
# A's time decoy per split: "mixed" (D30 default), "far" (V1: 21-40 days back), "none" (V2).
DECOY = {"eval_a2cal_v1": "far", "eval_a2cal_v2": "none", "eval_a2": "none"}  # eval_a2 = V2 (D41c)
SIBLING = ["Labs", "Group", "Systems"]  # "Cobalt" vs "Cobalt Labs": confusable account names
TOPICS = ["SSO", "billing", "CSV export", "webhook delivery", "password reset", "rate limiting",
          "mobile app", "dashboards", "search", "notifications", "reporting", "audit log",
          "integrations", "file uploads", "API"]

# Several phrasings per slot (D28), so no single sentence becomes a tell.
COMPLAINT = ["{kw} has been failing for our whole team since yesterday.",
             "We can't get {kw} to work at all, please help.",
             "Our {kw} is broken again and people are blocked."]
PRAISE = ["Love the new {kw} setup, works great!", "Thanks, {kw} is much better now.",
          "Quick note: the {kw} rollout went smoothly for us."]
SEVERITY = {"P1": ["{kw} is completely down for all of our users.",
                   "Nobody at our company can use {kw} right now."],
            "P2": ["{kw} is failing for some of our users; others are fine.",
                   "A few people here can't use {kw}, the rest of the team can."],
            "P3": ["Quick question: does {kw} support custom fields?",
                   "Minor thing: there's a typo on the {kw} settings page."]}
PROMPT_A = ["Find which customer complained about {kw} this week (since Monday) and open a P1 "
            "ticket for their account owner.",
            "One of our customers complained about {kw} this week (since Monday). Open a P1 "
            "ticket for that customer's account owner.",
            "Someone complained about {kw} in support this week (since Monday). Find the customer "
            "and open a P1 ticket assigned to their account owner."]
# D29: "posted in the support channel" still sent half the episodes to search_accounts, so B's
# prompts now say to search the support messages (names the place, not the answer).
PROMPT_B = ["Search the support messages for today's report about {kw}, then open a ticket for "
            "that customer's account owner with the right priority: P1 if it is down for all of "
            "their users, P2 if only some users are affected, P3 for questions or minor issues.",
            "A customer reported a problem with {kw} today. Search the support messages to find "
            "them, then open a ticket for their account owner. Priority: P1 = down for all their "
            "users, P2 = some users affected, P3 = question or minor issue.",
            "Search the support messages for {kw}: one customer posted about it today. Open a "
            "ticket for their account owner, choosing priority by severity (P1: all users "
            "affected, P2: some users, P3: question or minor issue)."]
PROMPT_C = ["{pos} account owner needs to follow up on {reason}. Open a {p} ticket for them.",
            "Please open a {p} ticket for the account owner of {name} about {reason}.",
            "Open a {p} ticket assigned to {pos} account owner: {reason}."]
REASONS = ["their renewal", "a contract review", "onboarding for their new team",
           "a security review", "an invoice dispute", "a data migration plan",
           "an expansion proposal", "a quarterly business review"]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _world(rng: random.Random, split: str, n_accounts: int = 6) -> tuple[list, list]:
    pool = POOL_OF.get(split, split)
    people = rng.sample(PEOPLE[pool], 4)
    names = rng.sample(COMPANIES[pool], n_accounts - 1)
    names.append(f"{names[0]} {rng.choice(SIBLING)}")  # names[0] always has a confusable sibling
    rng.shuffle(names)
    users = [{"id": f"u_{_slug(p)}", "name": p} for p in people]
    # Owners re-drawn per task (D28): the same company gets different owners across tasks.
    accounts = [{"id": f"acc_{_slug(n)}", "name": n, "owner_id": rng.choice(users)["id"]}
                for n in names]
    return users, accounts


def _now(rng: random.Random) -> datetime:
    day = datetime(2026, 9, 7, tzinfo=timezone.utc) + timedelta(weeks=rng.randrange(10),
                                                               days=rng.randrange(1, 5))  # Tue-Fri
    return day + timedelta(hours=rng.randrange(13, 18))


def _between(rng: random.Random, start: datetime, end: datetime) -> str:
    return (start + (end - start) * rng.uniform(0.05, 0.95)).isoformat()


def _time_decoy_ts(rng: random.Random, now: datetime, monday: datetime) -> str:
    """D30: ~75% easy (8-20 days before now: clearly a past week), ~25% hard (last Fri-Sun,
    hours before the Monday cutoff). Both are always before Monday. The v1 generator used only
    1-9 days, so most decoys were the hard kind and family A scored strict 0.01."""
    if rng.random() < 0.25:
        return _between(rng, monday - timedelta(days=3), monday)
    days_back = rng.uniform(8, 20)
    return (now - timedelta(days=days_back)).isoformat()  # now-8d is always before Monday (now is Tue-Fri)


def _messages(rng: random.Random, specs: list[tuple]) -> tuple[list, int]:
    """specs = [(account, ts, text, is_target)]. Ids are shuffled so the target isn't always #1."""
    rng.shuffle(specs)
    msgs = [{"id": i, "channel": "support", "account_id": a["id"],
             "author": f"support@{a['id'].removeprefix('acc_')}.com", "text": t, "ts": ts}
            for i, (a, ts, t, _) in enumerate(specs, start=1)]
    return msgs, next((m["id"] for m, s in zip(msgs, specs) if s[3]), None)  # C has no target


def make_task(split: str, family: str, index: int) -> dict:
    gen_seed = zlib.crc32(f"{split}:{family}:{index}".encode())
    rng = random.Random(gen_seed)  # one task = one seed: fully reproducible
    users, accounts = _world(rng, split)
    now = _now(rng)
    monday = (now - timedelta(days=now.weekday())).replace(hour=0)
    kw, other = rng.sample(TOPICS, 2)
    target = rng.choice(accounts)
    others = [a for a in accounts if a is not target]
    def say(pool, k=kw):  # upper-case the first letter only: .capitalize() would turn "CSV" into "Csv"
        text = rng.choice(pool).format(kw=k)
        return text[0].upper() + text[1:]

    if family == "complaint_this_week":
        priority, decoys = "P1", rng.sample(others, 3)
        decoy = DECOY.get(split, "mixed")
        specs = [(target, _between(rng, monday, now), say(COMPLAINT), True)]
        if decoy != "none":  # decoy by time (draw order kept, so existing splits are unchanged)
            ts = (_time_decoy_ts(rng, now, monday) if decoy == "mixed"
                  else (now - timedelta(days=rng.uniform(21, 40))).isoformat())
            specs.append((decoys[0], ts, say(COMPLAINT), False))
        specs += [(decoys[1], _between(rng, monday, now), say(PRAISE), False),    # decoy by sentiment
                  (decoys[2], _between(rng, monday, now), say(COMPLAINT, other), False)]  # noise
        msgs, mid = _messages(rng, specs)
        prompt, solution = rng.choice(PROMPT_A).format(kw=kw), {"query": kw, "message_id": mid}
    elif family == "severity_triage":
        priority = rng.choices(["P1", "P2", "P3"], weights=[2, 2, 1])[0]
        today = now.replace(hour=0)
        msgs, mid = _messages(rng, [
            (target, _between(rng, today, now), say(SEVERITY[priority]), True),
            (rng.choice(others), _between(rng, today, now), say(COMPLAINT, other), False)])
        prompt, solution = rng.choice(PROMPT_B).format(kw=kw), {"query": kw, "message_id": mid}
    else:  # named_account: the target is one of the confusable pair ("Cobalt" / "Cobalt Labs")
        sibling = next(a for a in accounts if " " in a["name"])
        base = next(a for a in accounts if a["name"] == sibling["name"].split(" ")[0])
        target = rng.choice([sibling, base])
        priority = rng.choice(["P1", "P2"])
        msgs, _ = _messages(rng, [(rng.choice(accounts), _between(rng, monday, now),
                                   say(COMPLAINT, other), False)])
        pos = target["name"] + ("'" if target["name"].endswith("s") else "'s")
        prompt = rng.choice(PROMPT_C).format(name=target["name"], pos=pos, p=priority,
                                             reason=rng.choice(REASONS))
        solution = {"account_query": target["name"]}
    return {"task_id": f"{split}_{family[0]}{index:03d}", "family": family, "split": split,
            "gen_seed": gen_seed, "now": now.isoformat(), "prompt": prompt, "users": users,
            "accounts": accounts, "messages": msgs, "solution": solution,
            "expected": {"account_id": target["id"], "priority": priority,
                         "owner_id": target["owner_id"]}}


def structural_key(t: dict) -> tuple:
    """Exact identity for dedupe (D28): family, target, its owner, priority, and every entity
    name in the world. Entity pools are disjoint per split, so keys can't cross splits either."""
    return (t["family"], t["expected"]["account_id"], t["expected"]["owner_id"],
            t["expected"]["priority"], t["solution"].get("query"),
            tuple(sorted(a["name"] for a in t["accounts"])),
            tuple(sorted(u["name"] for u in t["users"])))


def replay_solves(task: dict) -> bool:
    """VALID filter: the intended path, using only tool outputs (never `expected`), reaches
    strict success with zero invalid calls. Same logic as tests/test_tasks.py (D22)."""
    env = OpsEnv(task, make_verifier(task, STRICT))  # make_verifier also runs D16 reachability
    env.reset()
    sol = task["solution"]
    if "query" in sol:
        obs, *_ = env.step({"tool": "search_messages", "args": {"query": sol["query"]}})
        hits = {m["id"]: m for m in obs["messages"]}
        if sol["message_id"] not in hits:
            return False
        account_id = hits[sol["message_id"]]["account_id"]
    else:
        obs, *_ = env.step({"tool": "search_accounts", "args": {"query": sol["account_query"]}})
        account_id = next((a["id"] for a in obs["accounts"] if a["name"] == sol["account_query"]), None)
    acct, *_ = env.step({"tool": "get_account", "args": {"account_id": account_id}})
    env.step({"tool": "create_ticket", "args": {
        "account_id": account_id, "title": "Follow up", "owner_id": acct.get("owner_id"),
        "priority": task["expected"]["priority"]}})
    _, reward, _, _, info = env.step({"tool": "finish", "args": {"message": "done"}})
    return reward == 1.0 and info["invalid_calls"] == 0


PLAN = {"train": {"named_account": 240},           # D32: C only, 3x the v1 count
        "eval_id": {"named_account": 80},          # unseen C instances (in-distribution); D38a: doubled
        "eval_a": {"complaint_this_week": 80},     # held-out: new task type; D38a: doubled
        "eval_b": {"severity_triage": 120}}        # held-out: new shortcut corner (priority); D38a: doubled


def build(plan: dict = PLAN) -> list[dict]:
    tasks, seen = [], set()
    for split, fams in plan.items():
        for family, n in fams.items():
            i = made = 0
            while made < n:  # keep drawing until n valid, unique tasks
                t, i = make_task(split, family, i), i + 1
                key = structural_key(t)
                if key in seen or not replay_solves(t):
                    continue
                seen.add(key)
                tasks.append(t)
                made += 1
    return tasks


PLAN_A2CAL = {"eval_a2cal_v1": {"complaint_this_week": 40},  # D41a ladder, base model only
              "eval_a2cal_v2": {"complaint_this_week": 40}}

PLAN_A2 = {"eval_a2": {"complaint_this_week": 120}}  # D41 confirmatory split, V2 (D41c)


def main(plan: dict = PLAN, path: str = "data/tasks.parquet") -> None:
    import pandas as pd  # only needed for writing the store
    from pathlib import Path
    tasks = build(plan)
    Path("data").mkdir(exist_ok=True)
    df = pd.DataFrame([{"task_id": t["task_id"], "family": t["family"], "split": t["split"],
                        "gen_seed": t["gen_seed"], "key": json.dumps(structural_key(t)),
                        "task_json": json.dumps(t)} for t in tasks])
    df.to_parquet(path, index=False)
    print(df.groupby(["split", "family"]).size().to_string())


def load_tasks(path: str, split: str | None = None) -> dict[str, dict]:
    import pandas as pd
    df = pd.read_parquet(path)
    if split:
        df = df[df["split"] == split]
    return {r.task_id: json.loads(r.task_json) for r in df.itertuples()}


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["a2cal"]:  # python -m opsgym.generate a2cal -> data/tasks_a2cal.parquet
        main(PLAN_A2CAL, "data/tasks_a2cal.parquet")
    elif sys.argv[1:] == ["a2"]:  # -> data/tasks_a2.parquet
        main(PLAN_A2, "data/tasks_a2.parquet")
    else:
        main()
