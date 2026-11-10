"""Phase 6 (D38/D38a/D40) analysis of eval_id / eval_a / eval_b, as run on 2026-11-08 (recovered
verbatim from the inline session code into a script). Gates first, then:
  runs/analysis_primary.json    primary metric per split, bootstrap 10,000, Holm over 6
  runs/analysis_secondary.json  secondary metrics + decision probe, bootstrap 2,000
    python scripts/analyze_phase6.py                  # as reported (D40)
    python scripts/analyze_phase6.py --deterministic  # D42: sorted arm order -> *_deterministic.json"""
import json
import sys

import numpy as np

from opsgym.analysis import ARM_OF, PRIMARY, bootstrap, cell_rates, completeness, holm, replay
from opsgym.evaluate import MODELS
from opsgym.generate import load_tasks

SPLITS = ["eval_id", "eval_a", "eval_b"]
SECONDARY = ["false_claim", "strict_success", "invalid_calls", "multi_call_reply", "length_cutoff",
             "owner_unassigned", "skipped_work"]
ARMS = ["base", "strict_0", "natural", "shortcut_100"]
DET = "--deterministic" in sys.argv
SUFFIX = "_deterministic" if DET else ""
dump = lambda obj, path: json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.item())
_bootstrap = bootstrap


def bootstrap(rates, **kw):
    """D42: cell_rates orders arms by set iteration, which varies with Python's per-process string
    hashing, so the order arms consume the RNG varies. Sorting the arms fixes it. The frozen
    analysis.py (D39) is not touched; the default path keeps the order as originally run."""
    return _bootstrap(dict(sorted(rates.items())) if DET else rates, **kw)

tasks = {s: load_tasks("data/tasks.parquet", s) for s in SPLITS}
recs = {}
for s in SPLITS:
    recs[s] = completeness(f"runs/evals/{s}.jsonl", tasks[s])
    replay(recs[s], tasks[s])
print("GATES PASSED")

res, pvals = {}, {}
for s, metric in PRIMARY.items():
    b = bootstrap(cell_rates(recs[s], tasks[s], metric))
    res[s] = {"metric": metric, **b}
    for c, v in b["contrasts"].items():
        pvals[f"{s}:{c}"] = float(v["p"])
dump({"primary": res, "holm": holm(pvals)}, f"runs/analysis_primary{SUFFIX}.json")

out = {}
probe = json.load(open("runs/probe_eval.json"))["results"]
for s in SPLITS:
    out[s] = {m: bootstrap(cell_rates(recs[s], tasks[s], m), n=2000) for m in SECONDARY}
    tids = sorted(tasks[s])  # probe: P(shortcut) = 1 - P(continue), per (model, task)
    rates = {a: np.array([[1 - probe[m][t] for t in tids] for m in MODELS if ARM_OF[m] == a]) for a in ARMS}
    out[s]["probe_p_shortcut"] = bootstrap(rates, n=2000)
dump(out, f"runs/analysis_secondary{SUFFIX}.json")
print(f"wrote runs/analysis_primary{SUFFIX}.json, runs/analysis_secondary{SUFFIX}.json")
