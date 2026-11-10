"""Phase 5 training data (D33): a dose-response design from ONE set of T=1.0 rollouts.
    python -m opsgym.pools runs/<training rollouts>.jsonl   ->  data/pools/*.jsonl

Three pools with equal N, differing in the share of shortcut episodes (the "dose"):
  strict_0       strict ✓                                    0% shortcut
  natural        loose ✓ (what a loose verifier pays for)    ~37%: PRIMARY realistic result
  shortcut_100   loose ✓ ∧ strict ✗                          100%: positive control
N = size of the shortcut pool (the smallest). Strict and natural are subsampled stratified by
task, so each task keeps its share of the pool."""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from opsgym.report import hack_kind

NUDGE = {"message": "No tool call found. Call a tool, or call finish when you are done."}


def trainable_turns(messages: list[dict]) -> list[bool]:
    """D32 loss mask, one flag per assistant turn, same rule for every pool: a turn that was
    text-only or produced ANY error result stays in the context but gets no loss. Mixed turns
    (one good call + one bad) are masked whole: we don't train on half a reply."""
    flags, current = [], None
    for m in messages:
        if m["role"] == "assistant":
            flags.append(True)
            current = len(flags) - 1
        elif m["role"] == "tool" and current is not None:
            obs = json.loads(m["content"])
            if "error" in obs or obs == NUDGE:
                flags[current] = False
    return flags


def dose_pools(records: list[dict]) -> dict[str, list[dict]]:
    scored = [r for r in records if r["finish_kind"] != "error"]  # D21
    strict = [r for r in scored if r["rewards"]["strict"] == 1.0]
    loose = [r for r in scored if r["rewards"]["loose"] == 1.0]
    shortcut = [r for r in loose if r["rewards"]["strict"] == 0.0]
    return {"strict_0": strict, "natural": loose, "shortcut_100": shortcut}


def stratified_sample(pool: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n episodes, each task getting its proportional share (largest-remainder rounding,
    ties broken by task_id), episodes within a task drawn at random. Deterministic for a seed,
    whatever order the rollouts were written in."""
    ordered = sorted(pool, key=lambda r: (r["task_id"], r["sample_idx"]))
    if n >= len(pool):
        return ordered
    by_task = defaultdict(list)
    for r in ordered:
        by_task[r["task_id"]].append(r)
    exact = {t: n * len(eps) / len(pool) for t, eps in by_task.items()}
    quota = {t: int(q) for t, q in exact.items()}
    for t in sorted(exact, key=lambda t: (quota[t] - exact[t], t))[: n - sum(quota.values())]:
        quota[t] += 1  # the largest fractional remainders get the leftover slots
    rng = random.Random(seed)
    return [r for t in sorted(by_task) for r in rng.sample(by_task[t], quota[t])]


def describe(name: str, pool: list[dict], tokenizer=None) -> str:
    kinds = Counter(hack_kind(r) or "strict_pass" for r in pool)
    dose = sum(r["rewards"]["strict"] == 0.0 for r in pool) / max(len(pool), 1)
    turns = [sum(m["role"] == "assistant" for m in r["messages"]) for r in pool]
    flags = [f for r in pool for f in trainable_turns(r["messages"])]
    line = (f"{name:<13} N={len(pool):<5} dose {dose:.0%}  tasks {len({r['task_id'] for r in pool})}  "
            f"contents {dict(kinds)}  mean turns {sum(turns) / max(len(pool), 1):.2f}  "
            f"loss-bearing turns {sum(flags)}/{len(flags)}")
    if tokenizer is not None:  # generated tokens only: what SFT puts loss on (D20)
        toks = [sum(len(tokenizer.encode(m["content"])) for m in r["messages"]
                    if m["role"] == "assistant") for r in pool]
        line += f"  mean generated tokens {sum(toks) / max(len(pool), 1):.0f}"
    return line


def main(path: str, seed: int = 0) -> None:
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    full = dose_pools(records)
    n = len(full["shortcut_100"])  # the smallest pool sets N for all three (D33)
    pools = {name: stratified_sample(pool, n, seed) for name, pool in full.items()}
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(records[0]["model"])
    Path("data/pools").mkdir(parents=True, exist_ok=True)
    for name, pool in pools.items():
        with open(f"data/pools/{name}.jsonl", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in pool)
        print(describe(name, pool, tok))


if __name__ == "__main__":
    main(sys.argv[1])
