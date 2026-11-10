"""Phase 6 evals on Modal (D38, D38a): one L4 job per held-out split, each serving base + 9 LoRA
adapters from ONE vLLM v0.11.0 server (same image as local rollouts). Blinded: jobs return counts.
    python -m modal deploy modal_eval.py && python scripts/launch_eval.py
Output: Volume opsgym-runs, evals/<split>.jsonl (resumable)."""
import modal

image = (modal.Image.from_registry("vllm/vllm-openai:v0.11.0")
         .entrypoint([])
         # The vLLM image ships python3 but no `python`, which Modal's build steps call.
         .run_commands("ln -sf $(command -v python3) /usr/local/bin/python")
         .pip_install("jsonschema", "pandas", "pyarrow")
         .add_local_python_source("opsgym")
         .add_local_file("data/tasks.parquet", "/root/data/tasks.parquet")
         .add_local_file("data/tasks_a2.parquet", "/root/data/tasks_a2.parquet")  # D41
         .add_local_file("data/eval_manifest.json", "/root/data/eval_manifest.json"))
runs = modal.Volume.from_name("opsgym-runs")
hf_cache = modal.Volume.from_name("opsgym-hf-cache")
app = modal.App("opsgym-eval", image=image)
L4_DOLLARS_PER_HOUR = 0.80  # ponytail: approximate list price, for a rough cost line only

# Hard cost cap. D38a ($12): 3 jobs x 4 h. D41 (eval_a2 only): 1 job x 2 h x $0.80/h = $1.60.
@app.function(gpu="L4", timeout=2 * 60 * 60, volumes={"/vol": runs, "/root/.cache/huggingface": hf_cache})
def eval_split(split: str) -> dict:
    import hashlib
    import json
    import subprocess
    import time
    from pathlib import Path

    import httpx
    from transformers import AutoTokenizer

    from opsgym.chat import MODEL_ID
    from opsgym.evaluate import check_prompts, run_eval
    from opsgym.generate import load_tasks
    started = time.time()
    manifest = json.loads(Path("/root/data/eval_manifest.json").read_text())
    for name, sha in manifest["adapters"].items():  # never evaluate an adapter we didn't train (D37)
        got = hashlib.sha256(Path(f"/vol/runs/{name}/adapter_model.safetensors").read_bytes()).hexdigest()
        assert got == sha, f"{name}: adapter sha256 {got[:16]} != trained {sha[:16]}"
    server = subprocess.Popen(
        ["vllm", "serve", MODEL_ID, "--port", "8000", "--gpu-memory-utilization", "0.85",
         "--max-model-len", "8192", "--enable-lora", "--max-loras", "10", "--max-lora-rank", "16",
         "--lora-modules", *[f"{n}=/vol/runs/{n}" for n in manifest["adapters"]]])
    for _ in range(180):  # up to 15 min for the server (first run downloads the base weights)
        try:
            httpx.get("http://localhost:8000/health", timeout=5).raise_for_status()
            break
        except httpx.HTTPError:
            assert server.poll() is None, "vLLM exited during startup"
            time.sleep(5)
    else:
        raise RuntimeError("vLLM not healthy after 15 min")
    hf_cache.commit()
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    store = "tasks_a2.parquet" if split == "eval_a2" else "tasks.parquet"  # D41: separate store
    tasks = load_tasks(f"/root/data/{store}", split)
    check_prompts(tasks, tok, manifest["first_prompt_hashes"])  # byte-identical prompts or refuse
    out = Path(f"/vol/evals/{split}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    # 64 concurrent episodes: L4 (24 GB) has room for bigger batches than the local 4060 smoke test.
    counts = run_eval(tasks, out, tok, "http://localhost:8000", workers=64, on_progress=runs.commit)
    server.terminate()
    meta = {"split": split, "episodes": sum(counts.values()), "finish_kinds": dict(counts),
            "file_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "gpu": "L4", "wall_seconds": round(time.time() - started),
            "approx_cost_usd": round((time.time() - started) / 3600 * L4_DOLLARS_PER_HOUR, 2)}
    Path(f"/vol/evals/{split}.meta.json").write_text(json.dumps(meta, indent=1))
    runs.commit()
    return meta  # counts only: no rates while blinded (D38)
