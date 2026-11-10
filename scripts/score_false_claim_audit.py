"""D44: precision of the rule-based false_claim metric (D39) on the 40-item stratified audit.
Reads the blind labels (runs/audit_false_claim_labels.json: {"NN": {"label", "reason"}}) and the key
(transcripts/false_claim_audit/key.json: arm/split per item + flagged-pool size per stratum).
Precision = share labelled "real" ("unclear" counts as not real; also reported with unclear dropped).
Weighted estimates weight each (split, arm) stratum by its flagged-pool size. CIs: Jeffreys
Beta(real + 1/2, n - real + 1/2) per stratum, Monte Carlo over strata (handles 3/3 strata,
where a normal approximation has zero width).
    python scripts/score_false_claim_audit.py -> runs/audit_false_claim.json"""
import json
from collections import defaultdict

import numpy as np

KEY = json.load(open("transcripts/false_claim_audit/key.json"))
LAB = {k: v for k, v in json.load(open("runs/audit_false_claim_labels.json")).items() if k.isdigit()}
ARMS = ["base", "strict_0", "natural", "shortcut_100"]
rng = np.random.default_rng(0)


def estimate(strata: dict, drop_unclear: bool) -> dict:
    """strata: {(split, arm): [labels]} -> weighted precision + 95% interval."""
    w, a, b = [], [], []
    for k, labels in strata.items():
        ls = [x for x in labels if not (drop_unclear and x == "unclear")]
        if not ls:
            continue
        real = sum(x == "real" for x in ls)
        w.append(KEY["pool_sizes"]["|".join(k)])
        a.append(real + 0.5)
        b.append(len(ls) - real + 0.5)
    w = np.array(w) / sum(w)
    draws = (rng.beta(np.array(a)[:, None], np.array(b)[:, None], (len(a), 20000)) * w[:, None]).sum(0)
    point = float(sum(wi * (ai - 0.5) / (ai + bi - 1) for wi, ai, bi in zip(w, a, b)))
    return {"precision": point, "ci95": np.percentile(draws, [2.5, 97.5]).tolist()}


strata = defaultdict(list)
for it in KEY["items"]:
    strata[(it["split"], it["arm"])].append(LAB[f"{it['item']:02d}"]["label"])
out = {"n_items": len(KEY["items"]), "label_counts": {}, "raw": {}, "weighted": {}}
for arm in ARMS + ["overall"]:
    sub = {k: v for k, v in strata.items() if arm in ("overall", k[1])}
    labels = [x for v in sub.values() for x in v]
    out["label_counts"][arm] = {x: labels.count(x) for x in ("real", "honest", "unclear")}
    out["raw"][arm] = f"{labels.count('real')}/{len(labels)}"
    out["weighted"][arm] = {"unclear_as_not_real": estimate(sub, False), "unclear_dropped": estimate(sub, True)}
json.dump(out, open("runs/audit_false_claim.json", "w"), indent=1)
for arm in ARMS + ["overall"]:
    e = out["weighted"][arm]["unclear_as_not_real"]
    print(f"{arm:13} raw {out['raw'][arm]:6} {out['label_counts'][arm]}  weighted precision "
          f"{e['precision']:.3f} [{e['ci95'][0]:.3f}, {e['ci95'][1]:.3f}]")
