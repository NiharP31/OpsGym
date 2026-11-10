"""Decision-token probe on the held-out splits (D38 secondary), all 10 models, run locally.
BLINDED: writes runs/probe_eval.json and prints only progress counts."""
import hashlib
import json
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from opsgym.evaluate import MODELS
from opsgym.generate import load_tasks
from opsgym.probe import decision_prob

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
manifest = json.load(open("data/eval_manifest.json"))["adapters"]
for name, sha in manifest.items():  # same rule as the Modal evals: only the adapters we trained
    got = hashlib.sha256(open(f"runs/modal/{name}/adapter_model.safetensors", "rb").read()).hexdigest()
    assert got == sha, f"{name}: sha mismatch"
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to("cuda")
adapters = [m for m in MODELS if m != "base"]
model = PeftModel.from_pretrained(base, f"runs/modal/{adapters[0]}", adapter_name=adapters[0])
for a in adapters[1:]:
    model.load_adapter(f"runs/modal/{a}", adapter_name=a)
model.eval()
# D41: `probe_eval.py a2` probes eval_a2 into its own file; no args = the D38 splits.
A2 = sys.argv[1:] == ["a2"]
OUT = "runs/probe_eval_a2.json" if A2 else "runs/probe_eval.json"
tasks = (load_tasks("data/tasks_a2.parquet", "eval_a2") if A2 else
         {t: v for s in ["eval_id", "eval_a", "eval_b"] for t, v in load_tasks("data/tasks.parquet", s).items()})
out = {}
for m in MODELS:
    if m == "base":
        with model.disable_adapter():
            out[m] = {t: decision_prob(model, tok, task) for t, task in tasks.items()}
    else:
        model.set_adapter(m)
        out[m] = {t: decision_prob(model, tok, task) for t, task in tasks.items()}
    print(f"probe done for {m}: {len(out[m])} tasks", flush=True)
json.dump({"metric": "P(continue) at the fork; P(shortcut) = 1 - value", "results": out},
          open(OUT, "w"))
print(f"wrote {OUT} (blinded: not summarised)")
