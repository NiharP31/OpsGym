"""Phase 6 eval harness (D38, D38a). Orchestration only: every episode is the tested
run_episode, identical to training rollouts. Blinded: returns and prints COUNTS, never rates."""
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from opsgym.chat import MODEL_ID, first_messages, prompt_hash, render_prompt
from opsgym.env import OpsEnv
from opsgym.report import hack_kind
from opsgym.rollout import derive_sampling_seed, run_episode, vllm_generate
from opsgym.runlog import code_version, drop_errors, read_valid, task_set_hash

ARMS, SEEDS = ["strict_0", "natural", "shortcut_100"], [0, 1, 2]
MODELS = ["base"] + [f"{a}_s{s}" for a in ARMS for s in SEEDS]  # "base" = no adapter
SAMPLES, TEMPERATURE, MAX_TOKENS, BASE_SEED = 8, 0.7, 512, 0     # D38a: same as rollouts
KEY = ("model", "task_id", "sample_idx")


def check_prompts(tasks: dict, tokenizer, expected: dict) -> None:
    """Refuse to run if this environment renders any first prompt differently from the local
    render the model was trained against (e.g. a different transformers version in the image)."""
    bad = [t for t, task in tasks.items()
           if prompt_hash(render_prompt(tokenizer, first_messages(OpsEnv(task, lambda s: 0.0).reset()[0])))
           != expected[t]]
    assert not bad, f"{len(bad)} prompts render differently here, e.g. {bad[:3]}"


def run_eval(tasks: dict, out: Path, tokenizer, base_url: str, models: list[str] = MODELS,
             workers: int = 32, on_progress=None, every: int = 200) -> Counter:
    header = {"code_version": code_version(), "task_set_hash": task_set_hash(tasks),
              "sampling": {"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS, "base_seed": BASE_SEED}}
    done = {}
    if out.exists():  # resume (D31 rules, keyed by model too): valid, compatible, non-error lines only
        records, dropped = read_valid(out, KEY)
        records, errored = drop_errors(records)
        for r in records:
            assert r["code_version"] == header["code_version"] and r["task_set_hash"] == header["task_set_hash"] \
                and r["sampling"] == header["sampling"], f"incompatible line {[r[k] for k in KEY]}"
            assert r["sampling_seed"] == derive_sampling_seed(BASE_SEED, r["task_id"], r["sample_idx"])
        out.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        done = {tuple(r[k] for k in KEY) for r in records}
        print(f"resume: kept {len(records)}, redoing {dropped} invalid + {errored} error", flush=True)
    jobs = [(m, t, k) for m in models for t in tasks for k in range(SAMPLES) if (m, t, k) not in done]
    gens = {m: vllm_generate(base_url, MODEL_ID if m == "base" else m, TEMPERATURE, MAX_TOKENS)
            for m in models}
    counts = Counter()
    with ThreadPoolExecutor(workers) as pool, out.open("a", encoding="utf-8") as f:
        # Jobs interleave models, so vLLM batches across all 10 at once (multi-LoRA).
        futs = {pool.submit(run_episode, tasks[t], gens[m], tokenizer,
                            derive_sampling_seed(BASE_SEED, t, k)): (m, t, k)  # same seed for every model (CRN)
                for m, t, k in sorted(jobs, key=lambda j: (j[1], j[2], j[0]))}
        for n, fut in enumerate(as_completed(futs), 1):
            m, t, k = futs[fut]
            rec = {"run_id": f"eval-{header['task_set_hash']}",  # required by runlog validation
                   "model": m, "task_id": t, "family": tasks[t]["family"], "split": tasks[t]["split"],
                   "sample_idx": k, **header, **fut.result()}
            rec["hack_kind"] = hack_kind(rec)
            f.write(json.dumps(rec) + "\n")
            counts[rec["finish_kind"]] += 1  # counts only: no rates while blinded (D38)
            if on_progress and n % every == 0:
                f.flush()
                on_progress()
                print(f"{n}/{len(jobs)} episodes written  {dict(counts)}", flush=True)
    final, bad = read_valid(out, KEY)
    want = {(m, t, k) for m in models for t in tasks for k in range(SAMPLES)}
    assert bad == 0 and {tuple(r[k] for k in KEY) for r in final} == want, \
        f"{out}: incomplete ({len(final)}/{len(want)}, {bad} bad lines)"
    return Counter(r["finish_kind"] for r in final)
