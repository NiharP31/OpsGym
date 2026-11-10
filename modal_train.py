"""Phase 5 training on Modal (D34, D35).
    python -m modal run --detach modal_train.py::pilot         # strict_0, 10% holdout, seed 0
Results land on the Volume "opsgym-runs" under runs/<name>/; fetch with
    python -m modal volume get opsgym-runs runs/<name> runs/modal/"""
import modal

# Same versions as the local debug run (D34 addendum): torch 2.11.0, transformers 5.22.2, PEFT 0.21.2.
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch==2.11.0", "transformers==5.22.2", "peft==0.21.2", "safetensors",
                      "jsonschema")
         .add_local_python_source("opsgym")
         .add_local_dir("data/pools", "/root/data/pools")
         .add_local_file("configs/sft.json", "/root/configs/sft.json"))
runs = modal.Volume.from_name("opsgym-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("opsgym-hf-cache", create_if_missing=True)  # download base once
app = modal.App("opsgym-train", image=image)

GPU = "L4"
# ponytail: approximate list price in $/hour, for a rough cost line only; check the Modal
# dashboard for the billed amount.
GPU_DOLLARS_PER_HOUR = {"L4": 0.80, "A10G": 1.10}


@app.function(gpu=GPU, timeout=2 * 60 * 60, volumes={"/vol": runs, "/root/.cache/huggingface": hf_cache})
def train_run(pool: str, seed: int, name: str, holdout: float = 0.0,
              val_pool: str = "val_strict") -> dict:
    import hashlib
    import json
    import os
    from pathlib import Path

    from transformers import AutoTokenizer

    from opsgym.chat import render_prompt
    from opsgym.sft_data import build, holdout_split, load_pool
    from opsgym.train import format_check, train
    os.chdir("/root")
    cfg = json.loads(Path("configs/sft.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    pool_path = Path(f"data/pools/{pool}.jsonl")
    episodes = load_pool(str(pool_path))
    train_eps, val_eps = holdout_split(episodes, holdout, seed=0) if holdout else (episodes, [])
    examples = build(train_eps, tok, cfg["max_len"])
    # Pilot: its own 10% holdout. Arms (D36): one shared val set of strict episodes that are in
    # NO arm's pool, so every arm is scored on the same correct behaviour and step 0 = base model.
    val = build(val_eps or load_pool(f"data/pools/{val_pool}.jsonl"), tok, cfg["max_len"])
    out = f"/vol/runs/{name}"
    meta = train(examples, cfg, out, seed, val=val, on_ckpt=runs.commit,
                 log=lambda r: print(json.dumps(r), flush=True))
    if holdout:  # pilot only: format check on held-out *train-split* first turns (D34)
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16).to("cuda")
        model = PeftModel.from_pretrained(base, out)
        meta["format_check"] = format_check(
            model, tok, [render_prompt(tok, e["messages"][:2]) for e in val_eps[:20]])
    meta.update(pool=pool, pool_sha256=hashlib.sha256(pool_path.read_bytes()).hexdigest(),
                holdout=holdout, val_pool=None if holdout else val_pool, config=cfg, run=name,
                approx_cost_usd=round(meta["wall_seconds"] / 3600 * GPU_DOLLARS_PER_HOUR[GPU], 2))
    Path(out, "train_meta.json").write_text(json.dumps(meta, indent=1))
    runs.commit()
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}), flush=True)
    return {k: v for k, v in meta.items() if k != "history"}


@app.local_entrypoint()
def pilot():
    # spawn, not remote: with --detach the run continues after this local process exits.
    call = train_run.spawn("strict_0", seed=0, name="pilot_strict0_s0", holdout=0.1)
    print(f"spawned pilot: {call.object_id}  (logs: python -m modal app logs opsgym-train)")


ARMS = ["strict_0", "natural", "shortcut_100"]  # D33 dose 0% / ~36% / 100%
SEEDS = [0, 1, 2]                                # D36


@app.local_entrypoint()
def arms():
    """All 9 runs in parallel. This entrypoint WAITS for every run: in detached mode Modal only
    guarantees the last-triggered function survives the local process exiting."""
    calls = {f"{pool}_s{seed}": train_run.spawn(pool, seed=seed, name=f"{pool}_s{seed}")
             for pool in ARMS for seed in SEEDS}
    for name, call in calls.items():
        print(f"spawned {name}: {call.object_id}", flush=True)
    for name, call in calls.items():
        meta = call.get()
        print(f"DONE {name}: steps {meta['steps']} wall {meta['wall_seconds']}s ~${meta['approx_cost_usd']} "
              f"adapter {meta['adapter_sha256'][:16]}", flush=True)
