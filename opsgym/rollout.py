"""Run episodes with a model and log one JSON line per episode.
    python -m opsgym.rollout --samples 8 --temperature 0.7
Needs the vLLM server up (scripts/serve_vllm.ps1)."""
import argparse
import json
import time
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import httpx

from opsgym.chat import MODEL_ID, first_messages, parse_completion, prompt_hash, render_prompt
from opsgym.env import OpsEnv
from opsgym.report import format_families, format_summary, hack_kind, summarize
from opsgym.runlog import (check_compatible, code_version, drop_errors, read_valid,
                           task_set_hash)
from opsgym.generate import load_tasks
from opsgym.tasks import TASKS
from opsgym.verifiers import LOOSE, MEDIUM, STRICT, breakdown, make_verifier, passes
from opsgym.workspace import dump_state

CHECK_SETS = {"strict": STRICT, "medium": MEDIUM, "loose": LOOSE}
# generate(prompt, sampling_seed) -> (text, finish_reason)
Generate = Callable[[str, int], tuple[str, str]]


class GenerationError(Exception):
    """Backend failed (server down, context too long). Only this ends an episode as 'error';
    any other exception is our bug and propagates instead of becoming a quiet reward of 0."""


def derive_sampling_seed(base_seed: int, task_id: str, sample_idx: int) -> int:
    # Deterministic per (run seed, task, sample), and stable across Python runs (unlike hash()).
    return zlib.crc32(f"{base_seed}:{task_id}:{sample_idx}".encode())


def run_episode(seed: dict, generate: Generate, tokenizer, sampling_seed: int = 0,
                max_steps: int = 15) -> dict:
    # `seed` = the task's world state (D10); `sampling_seed` = randomness for the model.
    env = OpsEnv(seed, make_verifier(seed, STRICT), max_steps)  # make_verifier also runs D16
    obs, _ = env.reset()
    messages, hashes, finish_reasons = first_messages(obs), [], []
    info, finish_kind, error = None, None, None
    try:
        while info is None:
            prompt = render_prompt(tokenizer, messages)
            hashes.append(prompt_hash(prompt))  # lets SFT/eval prove they rebuild this exact prompt
            text, finish_reason = generate(prompt, sampling_seed)
            finish_reasons.append(finish_reason)  # "length" = cut off by max_tokens, not the model
            messages.append({"role": "assistant", "content": text})  # raw text, never re-serialized
            for action in parse_completion(text):
                obs, _, terminated, truncated, step_info = env.step(action)
                if not terminated:  # 1 result per call, incl. a finish rejected for bad args
                    messages.append({"role": "tool", "content": json.dumps(obs)})
                if terminated or truncated:
                    info, finish_kind = step_info, "finish" if terminated else "step_limit"
                    break
    except GenerationError as e:
        finish_kind, error = "error", str(e)
    state = info["final_state"] if info else dump_state(env.conn)
    return {
        "sampling_seed": sampling_seed, "messages": messages, "prompt_hashes": hashes,
        "finish_reasons": finish_reasons, "steps": env.steps,
        "invalid_calls": env.invalid_calls, "repeated_invalid_calls": env.repeated_invalid_calls,
        "text_only_turns": env.text_only_turns,
        "finish_kind": finish_kind, "error": error,
        "finish_message": info["finish_message"] if info else None, "final_state": state,
        # Score every check set from the same final state (D4): hack = loose and not strict.
        "rewards": {name: float(passes(state, seed, c)) for name, c in CHECK_SETS.items()},
        "breakdown": breakdown(state, seed),
        "expected": seed["expected"],  # makes each JSONL line self-contained for hack tagging
    }


def vllm_generate(base_url: str, model: str, temperature: float, max_tokens: int) -> Generate:
    client = httpx.Client(base_url=base_url, timeout=120)

    def generate(prompt: str, seed: int) -> tuple[str, str]:
        # Plain completions endpoint: WE rendered the prompt, so vLLM never applies a template.
        try:
            r = client.post("/v1/completions", json={
                "model": model, "prompt": prompt, "temperature": temperature,
                "max_tokens": max_tokens, "stop": ["<|im_end|>"], "seed": seed})
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise GenerationError(f"{type(e).__name__}: {e}") from e
        choice = r.json()["choices"][0]
        return choice["text"], choice["finish_reason"]
    return generate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--families", nargs="*", help="only run tasks from these families")
    ap.add_argument("--tasks-file", help="Parquet task store (default: hand-written opsgym.tasks)")
    ap.add_argument("--split", help="with --tasks-file: train / eval_id / eval_b")
    ap.add_argument("--model", default=MODEL_ID, help="must be a Qwen3 model: prompts use its template")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0, help="base seed; per-sample seeds derive from it")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--workers", type=int, default=16)  # concurrent episodes -> vLLM batches them
    ap.add_argument("--resume", help="finish this run file instead of starting a new one (D31)")
    args = ap.parse_args()

    from transformers import AutoTokenizer  # imported here so tests of run_episode don't need it
    tok = AutoTokenizer.from_pretrained(args.model)
    generate = vllm_generate(args.base_url, args.model, args.temperature, args.max_tokens)
    tasks = load_tasks(args.tasks_file, args.split) if args.tasks_file else TASKS
    task_ids = [t for t in tasks if not args.families or tasks[t]["family"] in args.families]
    all_jobs = [(tid, k) for tid in task_ids for k in range(args.samples)]
    header = {"code_version": code_version(),
              "task_set_hash": task_set_hash({t: tasks[t] for t in task_ids}), "model": args.model,
              "sampling": {"temperature": args.temperature, "max_tokens": args.max_tokens,
                           "base_seed": args.seed}}
    records = []
    if args.resume:
        out = Path(args.resume)
        records, dropped = read_valid(out)
        check_compatible(records, header, tasks, derive_sampling_seed)  # raises -> refuse to resume
        records, errored = drop_errors(records)
        run_id = records[0]["run_id"] if records else out.stem
        for r in records:  # stamp legacy lines that passed the evidence check
            r.setdefault("code_version", header["code_version"])
            r.setdefault("task_set_hash", header["task_set_hash"])
        tmp = out.with_suffix(".tmp")  # rewrite atomically: valid lines only, truncated ones gone
        tmp.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        tmp.replace(out)
        done = {(r["task_id"], r["sample_idx"]) for r in records}
        print(f"resume {out}: {len(records)} lines kept; redoing {dropped} invalid + {errored} error",
              flush=True)
    else:
        run_id, done = time.strftime("%Y%m%d-%H%M%S"), set()
        out = Path("runs") / f"{run_id}.jsonl"
        out.parent.mkdir(exist_ok=True)
    jobs = [j for j in all_jobs if j not in done]
    with ThreadPoolExecutor(args.workers) as pool, out.open("a", encoding="utf-8") as f:
        futures = {pool.submit(run_episode, tasks[tid], generate, tok,
                               derive_sampling_seed(args.seed, tid, k)): (tid, k) for tid, k in jobs}
        for fut in as_completed(futures):
            tid, k = futures[fut]
            rec = {"run_id": run_id, "task_id": tid, "family": tasks[tid]["family"],
                   "sample_idx": k, "model": args.model, "sampling": header["sampling"],
                   "code_version": header["code_version"],
                   "task_set_hash": header["task_set_hash"], **fut.result()}
            rec["hack_kind"] = hack_kind(rec)
            records.append(rec)
            f.write(json.dumps(rec) + "\n")
            print(f"{tid}#{k} T={args.temperature} seed={rec['sampling_seed']}: strict={rec['rewards']['strict']} loose={rec['rewards']['loose']} "
                  f"{rec['finish_kind']} steps={rec['steps']} invalid={rec['invalid_calls']} (repeated {rec['repeated_invalid_calls']}) "
                  f"text_only={rec['text_only_turns']} "
                  f"finish_reasons={rec['finish_reasons']}")
    # Completeness check on the file itself, not on in-memory state.
    final, dropped = read_valid(out)
    pairs = {(r["task_id"], r["sample_idx"]) for r in final}
    assert dropped == 0 and len(final) == len(pairs) == len(all_jobs) and pairs == set(all_jobs), \
        f"{out}: {len(pairs)} unique pairs, {dropped} bad lines, expected {len(all_jobs)}"
    print(f"wrote {out}: {len(pairs)} unique (task, sample) pairs, all valid\n")
    print(format_summary(summarize(final)) + "\n\n" + format_families(final))


if __name__ == "__main__":
    main()
