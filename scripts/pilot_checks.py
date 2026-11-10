"""Criteria 3 + 4 on the pilot adapter vs the base model, on held-out TRAIN prompts only
(the pilot's 10% strict_0 holdout; no eval split is touched before freezing, D34).
    python scripts/pilot_checks.py runs/modal/pilot_strict0_s0"""
import json
import statistics
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from opsgym.chat import turn_pairs
from opsgym.generate import load_tasks
from opsgym.probe import decision_prob, reply_hygiene
from opsgym.sft_data import holdout_split, load_pool

adapter = sys.argv[1]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("cuda")
model = PeftModel.from_pretrained(base, adapter).eval()

_, held = holdout_split(load_pool("data/pools/strict_0.jsonl"), 0.1, seed=0)  # same split as the pilot
tasks = load_tasks("data/tasks.parquet", "train")
task_ids = sorted({e["task_id"] for e in held})
prompts = [p for e in held[:30] for p, _ in turn_pairs(tok, e["messages"])][:120]

results = {}
for name in ["base", "pilot"]:
    if name == "base":
        with model.disable_adapter():  # same loaded weights, adapter switched off
            probs = [decision_prob(model, tok, tasks[t]) for t in task_ids]
            hyg = reply_hygiene(model, tok, prompts)
    else:
        probs = [decision_prob(model, tok, tasks[t]) for t in task_ids]
        hyg = reply_hygiene(model, tok, prompts)
    results[name] = {"p_owner": probs, "hygiene": hyg}
    print(f"{name:5} P(owner_id) over {len(probs)} tasks: mean {statistics.mean(probs):.3f}  "
          f"median {statistics.median(probs):.3f}  hygiene {hyg}", flush=True)

diffs = [p - b for p, b in zip(results["pilot"]["p_owner"], results["base"]["p_owner"])]
print(f"per-task shift pilot - base: mean {statistics.mean(diffs):+.3f}, "
      f"{sum(d > 0 for d in diffs)}/{len(diffs)} tasks up")
json.dump(results, open(f"{adapter}/pilot_checks.json", "w"), indent=1)
