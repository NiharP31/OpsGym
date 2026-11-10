"""EXPLORATORY (D43, not pre-registered): is shortcut_100's false-claim rise explained by more
unassigned tickets, or by a higher false-claim rate given an unassigned ticket?

Per arm and split (C, A, A2), among episodes where the agent created >= 1 ticket:
  P(false claim | owner unassigned)  - some agent ticket has owner_id null
  P(false claim | owner assigned)    - every agent ticket has an owner
Episodes with no agent ticket are excluded (their false claims are "claimed a ticket, none made").
Bootstrap as in D38 (tasks paired across arms, seeds within arm, 2,000 draws, arms sorted per D42),
but each draw is a ratio of sums (false claims / episodes in the condition), because many
(model, task) cells are empty under conditioning.
    python scripts/analyze_false_claim_conditional.py -> runs/exploratory_false_claim_conditional.json"""
import json

import numpy as np

from opsgym.analysis import ARM_OF, completeness, false_claim
from opsgym.evaluate import MODELS
from opsgym.generate import load_tasks
from opsgym.report import ticket_calls

SPLITS = {"eval_id": "data/tasks.parquet", "eval_a": "data/tasks.parquet", "eval_a2": "data/tasks_a2.parquet"}
ARMS = sorted(set(ARM_OF.values()))
CONDS = ("unassigned", "assigned")
N, SEED = 2000, 0


def counts(records, tasks):
    """[cond][model] -> (fc, n) arrays over sorted task ids."""
    tids = sorted(tasks)
    col = {t: i for i, t in enumerate(tids)}
    c = {k: {m: (np.zeros(len(tids)), np.zeros(len(tids))) for m in MODELS} for k in CONDS}
    for r in records:
        if r["finish_kind"] == "error":
            continue
        made = [t for t in r["final_state"]["tickets"] if t["id"] in ticket_calls(r["messages"])]
        if not made:
            continue
        k = "unassigned" if any(t["owner_id"] is None for t in made) else "assigned"
        fc, n = c[k][r["model"]]
        fc[col[r["task_id"]]] += false_claim(r, tasks[r["task_id"]])
        n[col[r["task_id"]]] += 1
    return c, len(tids)


def boot(c, n_tasks):
    by_arm = {a: [m for m in MODELS if ARM_OF[m] == a] for a in ARMS}
    stack = {k: {a: (np.stack([c[k][m][0] for m in by_arm[a]]), np.stack([c[k][m][1] for m in by_arm[a]]))
                 for a in ARMS} for k in CONDS}
    out = {}
    for k in CONDS:
        point = {a: float(stack[k][a][0].sum() / stack[k][a][1].sum()) for a in ARMS}
        n_eps = {a: int(stack[k][a][1].sum()) for a in ARMS}
        draws = {a: np.empty(N) for a in ARMS}
        rng = np.random.default_rng(SEED)
        for b in range(N):
            t = rng.integers(0, n_tasks, n_tasks)          # paired across arms
            for a in ARMS:
                fc, n = stack[k][a]
                s = rng.integers(0, fc.shape[0], fc.shape[0])  # seeds within arm
                draws[a][b] = fc[np.ix_(s, t)].sum() / max(n[np.ix_(s, t)].sum(), 1)
        out[k] = {"n_episodes": n_eps, "point": point,
                  "arm_ci": {a: np.percentile(draws[a], [2.5, 97.5]).tolist() for a in ARMS},
                  "vs_base": {a: {"delta": point[a] - point["base"],
                                  "ci95": np.percentile(draws[a] - draws["base"], [2.5, 97.5]).tolist()}
                              for a in ARMS if a != "base"}}
    return out


if __name__ == "__main__":
    res = {"label": "EXPLORATORY, not pre-registered (D43)"}
    for split, store in SPLITS.items():
        tasks = load_tasks(store, split)
        recs = completeness(f"runs/evals/{split}.jsonl", tasks)
        c, n_tasks = counts(recs, tasks)
        res[split] = boot(c, n_tasks)
    json.dump(res, open("runs/exploratory_false_claim_conditional.json", "w"), indent=1)
    for split in SPLITS:
        for k in CONDS:
            r = res[split][k]
            print(f"{split:8} {k:10} " + "  ".join(
                f"{a} {r['point'][a]:.3f} [{r['arm_ci'][a][0]:.3f},{r['arm_ci'][a][1]:.3f}] n={r['n_episodes'][a]}"
                for a in ARMS))
