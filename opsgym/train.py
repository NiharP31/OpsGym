"""LoRA SFT: plain PyTorch + PEFT (D34). The same code runs locally (debug) and on Modal.
    python -m opsgym.train debug        # local checks before any Modal run"""
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from opsgym.sft_data import IGNORE


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def example_loss(model, ex: dict) -> torch.Tensor:
    """Mean cross-entropy over the completion tokens of ONE example (unweighted)."""
    ids = torch.tensor([ex["input_ids"]], device=model.device)
    n_loss = sum(lab != IGNORE for lab in ex["labels"])
    # The completion is always the tail, so only compute logits there: full-sequence logits
    # would be ~1300 x 151k floats (~0.8 GB) for nothing. Position t predicts token t+1.
    logits = model(input_ids=ids, logits_to_keep=n_loss + 1).logits[0, :-1]
    targets = torch.tensor(ex["labels"][-n_loss:], device=model.device)
    return F.cross_entropy(logits.float(), targets)


@torch.no_grad()
def evaluate(model, examples: list[dict]) -> float:
    model.eval()
    return sum(example_loss(model, ex).item() * ex["weight"] for ex in examples) / \
        sum(ex["weight"] for ex in examples)


def load_model(cfg: dict):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16).to("cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return get_peft_model(model, LoraConfig(**cfg["lora"], task_type="CAUSAL_LM"))


def _save_ckpt(ckpt: Path, model, opt, sched, step: int, history: list) -> None:
    tmp = ckpt.with_name(ckpt.name + ".tmp")  # write aside, then swap: a kill mid-save can't
    model.save_pretrained(tmp)                # leave a half-written checkpoint behind
    torch.save({"step": step, "opt": opt.state_dict(), "sched": sched.state_dict(),
                "history": history}, tmp / "state.pt")
    if ckpt.exists():
        shutil.rmtree(ckpt)
    tmp.rename(ckpt)


def train(examples: list[dict], cfg: dict, out_dir: str, seed: int, val: list[dict] | None = None,
          max_steps: int | None = None, log=print, ckpt_every: int = 50, on_ckpt=None) -> dict:
    """on_ckpt: called after each checkpoint is written (Modal: volume.commit()). If out_dir/ckpt
    exists (preempted or timed-out run), training resumes there: the batch order is a pure
    function of the seed, so the same examples follow. Not bit-exact: LoRA-dropout RNG restarts."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import get_cosine_schedule_with_warmup
    started = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = load_model(cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    total = math.ceil(len(examples) * cfg["epochs"] / cfg["batch"])
    total = min(total, max_steps or total)
    sched = get_cosine_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * total), total)
    order = []
    for _ in range(cfg["epochs"]):  # a fresh seeded shuffle each epoch
        idx = list(range(len(examples)))
        rng.shuffle(idx)
        order += idx
    history, start = [], 0
    ckpt = Path(out_dir, "ckpt")
    if (ckpt / "state.pt").exists():
        state = torch.load(ckpt / "state.pt", weights_only=False)
        set_peft_model_state_dict(model, load_file(ckpt / "adapter_model.safetensors"))
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        history, start = state["history"], state["step"]
        log({"resumed_at_step": start})
    if val and start == 0:  # step 0 = the untrained model: the baseline every val number is read against
        history.append({"step": 0, "val_loss": round(evaluate(model, val), 4)})
        log(history[-1])
    for step in range(start, total):
        model.train()
        batch = order[step * cfg["batch"]:(step + 1) * cfg["batch"]]
        step_loss = 0.0
        for i in batch:  # micro-batch of 1: variable lengths, no padding to get wrong
            loss = example_loss(model, examples[i]) * examples[i]["weight"] / len(batch)
            loss.backward()
            step_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
        opt.step()
        sched.step()
        opt.zero_grad()
        rec = {"step": step + 1, "loss": round(step_loss, 4), "lr": sched.get_last_lr()[0]}
        if val and ((step + 1) % cfg["eval_every"] == 0 or step + 1 == total):
            rec["val_loss"] = round(evaluate(model, val), 4)
        history.append(rec)
        log(rec)
        if (step + 1) % ckpt_every == 0 and step + 1 < total:
            _save_ckpt(ckpt, model, opt, sched, step + 1, history)
            if on_ckpt:
                on_ckpt()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    weights = Path(out_dir, "adapter_model.safetensors").read_bytes()
    meta = {"config_hash": config_hash(cfg), "seed": seed, "examples": len(examples),
            "steps": total, "adapter_sha256": hashlib.sha256(weights).hexdigest(),
            "gpu": torch.cuda.get_device_name(0), "wall_seconds": round(time.time() - started),
            "resumed_from_step": start or None, "history": history}
    Path(out_dir, "train_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


@torch.no_grad()
def format_check(model, tokenizer, prompts: list[str], max_new_tokens: int = 200) -> dict:
    """Greedy generation on held-out prompts: does the adapter still emit parseable tool calls?
    A training-signal check only (D34): uses train-split prompts, never an eval split."""
    from opsgym.chat import parse_completion
    model.eval()
    ok, samples = 0, []
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        ok += any("tool" in a for a in parse_completion(text))
        samples.append(text[:200])
    return {"parse_rate": ok / len(prompts), "n": len(prompts), "samples": samples[:3]}


def debug() -> None:
    """Local checks before any Modal run (D34). Labels/masks are covered by test_sft_data."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from opsgym.chat import parse_completion, render_prompt
    from opsgym.sft_data import build, load_pool
    cfg = json.loads(Path("configs/sft.json").read_text())
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    eps = load_pool("data/pools/strict_0.jsonl")
    exs = build(eps[:8], tok, cfg["max_len"])  # ~4 examples per episode -> ~32

    # 1) Can it learn at all? Overfit ONE example: loss must collapse. The target must be text
    # the base model would NOT write: pool episodes are its own samples, so their loss already
    # starts at ~0 and "falls to 10%" proves nothing (found in the first debug run).
    n_prompt = exs[0]["labels"].count(-100)
    odd = tok('<tool_call>\n{"name": "finish", "arguments": {"message": "zebra quartz 4417"}}\n'
              '</tool_call><|im_end|>', add_special_tokens=False)["input_ids"]
    target = {"input_ids": exs[0]["input_ids"][:n_prompt] + odd,
              "labels": [-100] * n_prompt + odd, "weight": 1.0}
    over = {**cfg, "lr": 1e-3, "batch": 1, "epochs": 30, "warmup_ratio": 0.0}
    hist = train([target], over, "runs/debug/overfit", seed=0, log=lambda r: None)["history"]
    print(f"[1] overfit one example: loss {hist[0]['loss']:.3f} -> {hist[-1]['loss']:.4f}")
    assert hist[-1]["loss"] < 0.1 * hist[0]["loss"], "can't overfit one example: loss/labels bug"
    torch.cuda.empty_cache()

    # 2) A few real-config steps on ~20 examples: finite losses, adapter saved and hashed.
    meta = train(exs[:20], {**cfg, "batch": 4}, "runs/debug/steps", seed=0, val=exs[20:24],
                 max_steps=5)
    assert all(math.isfinite(h["loss"]) for h in meta["history"])
    print(f"[2] 5 steps ok, adapter sha256 {meta['adapter_sha256'][:16]}")
    torch.cuda.empty_cache()

    # 4) Resume: a 6-step run checkpoints at step 3. Re-running into the same dir must resume
    # at step 3 and reproduce steps 4-6 closely (not bit-exact: LoRA-dropout RNG restarts).
    shutil.rmtree("runs/debug/resume", ignore_errors=True)
    small = {**cfg, "batch": 2}
    full = train(exs[:12], small, "runs/debug/resume", seed=0, max_steps=6, ckpt_every=3,
                 log=lambda r: None)["history"]
    again = train(exs[:12], small, "runs/debug/resume", seed=0, max_steps=6, ckpt_every=3,
                  log=lambda r: None)
    assert again["resumed_from_step"] == 3 and again["history"][:3] == full[:3]
    drift = max(abs(a["loss"] - b["loss"]) for a, b in zip(again["history"][3:], full[3:]))
    print(f"[4] resumed at step 3; max loss drift on steps 4-6 = {drift:.4f}")
    assert drift < 0.05
    torch.cuda.empty_cache()

    # 3) Reload base + adapter and generate: output must still parse as a tool call.
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=torch.bfloat16).to("cuda")
    model = PeftModel.from_pretrained(base, "runs/debug/steps").eval()
    prompt = render_prompt(tok, eps[-1]["messages"][:2])  # a first turn not used in training
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to("cuda")
    out = model.generate(ids, max_new_tokens=150, do_sample=False)
    text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    actions = parse_completion(text)
    print(f"[3] reloaded adapter generates: {text.strip()[:160]!r} -> {actions}")
    assert any("tool" in a for a in actions), "adapter output no longer parses as a tool call"


if __name__ == "__main__":
    if sys.argv[1:2] == ["debug"]:
        debug()
