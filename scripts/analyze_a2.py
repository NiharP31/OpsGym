"""D41 / D41a / D41c analysis of eval_a2. Gates first (completeness + replay); nothing is computed
or printed before both pass. Uses the frozen opsgym.analysis (D39) unchanged.
    python scripts/analyze_a2.py                  ->  runs/analysis_a2.json (as reported, D41e)
    python scripts/analyze_a2.py --deterministic  ->  runs/analysis_a2_deterministic.json (D42)"""
import json
import sys

import numpy as np

from opsgym import analysis
from opsgym.analysis import ARM_OF, bootstrap, cell_rates, completeness, episode_metrics, holm, replay
from opsgym.generate import load_tasks
from opsgym.report import ticket_calls

DET = "--deterministic" in sys.argv  # D42: sorted arm order, see analyze_phase6.py
OUT = "runs/analysis_a2_deterministic.json" if DET else "runs/analysis_a2.json"

tasks = load_tasks("data/tasks_a2.parquet", "eval_a2")
records = completeness("runs/evals/eval_a2.jsonl", tasks)  # 9,600 triples, 0 bad lines, or raise
replay(records, tasks)                                       # rewards reproduce from final_state, or raise
print(f"GATES PASSED: {len(records)} records complete, replay consistent")


def run(metric: str, recs=records, contrasts=None) -> dict:
    rates = cell_rates(recs, tasks, metric)
    if DET:
        rates = dict(sorted(rates.items()))
    if contrasts:  # same bootstrap, same seed -> same draws; only which differences are read changes
        saved, analysis.CONTRASTS = analysis.CONTRASTS, contrasts
        try:
            return bootstrap(rates)
        finally:
            analysis.CONTRASTS = saved
    return bootstrap(rates)


out = {"n_records": len(records)}
# Primary (D41): H1 natural - strict_0, H2 shortcut_100 - strict_0, Holm over these 2.
prim = run("owner_unassigned")
out["primary"] = {**prim, "holm": holm({k: v["p"] for k, v in prim["contrasts"].items()})}
# Pre-registered secondary (D41a): vs base, its own Holm over 2.
vb = run("owner_unassigned", contrasts=[("natural", "base"), ("shortcut_100", "base")])
out["secondary_vs_base"] = {"contrasts": vb["contrasts"],
                            "holm": holm({k: v["p"] for k, v in vb["contrasts"].items()})}
# Descriptive (D41c note): owner_unassigned among ticket-creating episodes, + ticket-creation rate.
made = [r for r in records if r["finish_kind"] != "error" and
        any(t["id"] in ticket_calls(r["messages"]) for t in r["final_state"]["tickets"])]
cond = run("owner_unassigned", recs=made)
out["descriptive_among_ticket_creating"] = {"point": cond["point"], "arm_ci": cond["arm_ci"]}
by_arm = {}
for r in records:
    by_arm.setdefault(ARM_OF[r["model"]], []).append(any(t["id"] in ticket_calls(r["messages"])
                                                         for t in r["final_state"]["tickets"]))
out["ticket_creation_rate"] = {a: float(np.mean(v)) for a, v in sorted(by_arm.items())}
# Other secondaries (D41): same bootstrap, reported without multiplicity claims.
for m in ("false_claim", "skipped_work", "strict_success"):
    out[m] = run(m)
# Decision probe (local, runs/probe_eval_a2.json): mean P(skip owner) = 1 - P(continue), per arm.
probe = json.load(open("runs/probe_eval_a2.json"))["results"]
out["probe_p_skip"] = {a: float(np.mean([1 - v for m, d in probe.items() if ARM_OF[m] == a for v in d.values()]))
                       for a in sorted(set(ARM_OF.values()))}
json.dump(out, open(OUT, "w"), indent=1, default=lambda o: o.item())
print(f"wrote {OUT}")
