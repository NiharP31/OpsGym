# Reproducing OpsGym

Local: Python 3.14, `pip install -r requirements.txt`, Docker with GPU support (local vLLM).
Modal: `pip install modal`, `python -m modal setup`. On Windows set `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`
(the Modal CLI prints characters the default console encoding can't).
Run everything from the repo root with `PYTHONPATH=.` for `scripts/`.

## The five commands

```bash
# 1. Tests (no GPU, no network)
python -m pytest -q

# 2. One local smoke rollout: base Qwen3-1.7B on the 20 hand-written tasks, 1 sample each
powershell scripts/serve_vllm.ps1            # starts vLLM v0.11.0 in Docker on :8000
python -m opsgym.rollout --samples 1          # writes runs/<timestamp>.jsonl + a summary
powershell scripts/stop_serving.ps1

# 3. One training run on Modal (L4, ~1 h, ~$0.80). Needs data/pools/*.jsonl (see "Full pipeline").
python -m modal deploy modal_train.py
python scripts/relaunch_arms.py natural_s0   # any of {strict_0,natural,shortcut_100}_s{0,1,2}
python -m modal volume get opsgym-runs runs/natural_s0 runs/modal/

# 4. One eval split on Modal (L4, 20-35 min, ~$0.50). Needs all 9 adapters on the Volume.
python -m modal deploy modal_eval.py
python scripts/launch_eval.py eval_a2        # or eval_id / eval_a / eval_b
python scripts/eval_status.py                # blinded: counts only
python -m modal volume get opsgym-runs evals/eval_a2.jsonl runs/evals/

# 5. Analysis (gates first: completeness + replay, then bootstrap + Holm)
python scripts/analyze_phase6.py             # eval_id/eval_a/eval_b -> runs/analysis_{primary,secondary}.json
python scripts/probe_eval.py a2              # local GPU: decision probe -> runs/probe_eval_a2.json
python scripts/analyze_a2.py                 # eval_a2 -> runs/analysis_a2.json
# add --deterministic to either analysis script for run-to-run identical CIs (D42)
python scripts/reproduce_figures.py          # figures/ + results tables from the committed JSON only
```

Figures and tables need **only** the committed `runs/analysis_*.json` and `runs/probe_eval*.json`.

## Full pipeline (what produced the committed numbers)

| step | command | output | decision |
|---|---|---|---|
| tasks | `python -m opsgym.generate` (+ `a2cal`, `a2`) | `data/tasks*.parquet` | D28–D32, D41 |
| train rollouts, T=1.0 | `python -m opsgym.rollout --tasks-file data/tasks.parquet --split train --temperature 1.0` | `runs/<ts>.jsonl` | D30, D31 |
| dose pools | `python -m opsgym.pools runs/<ts>.jsonl` | `data/pools/*.jsonl` (hashes in `data/pools/MANIFEST.json`) | D33 |
| 9 adapters | `python -m modal run --detach modal_train.py::arms` | Volume `runs/<arm>_s<seed>/` | D35–D37 |
| evals | step 4 above, per split | `runs/evals/<split>.jsonl` | D38, D38a, D41 |
| analysis | step 5 above | `runs/analysis_*.json` | D39, D41a, D41c |

## Known reproducibility limits
- **Sampling isn't bit-exact.** Per-episode seeds are fixed (`derive_sampling_seed`), but vLLM
  batches requests, so a rerun gives statistically equal results, not identical transcripts.
- **Bootstrap CI order dependence (D42).** The frozen `opsgym.analysis.cell_rates` iterates arms
  in `set` order, which depends on Python's per-process string hashing, so the bootstrap's CIs and
  p-values vary run to run (point estimates never do). The committed `runs/analysis_*.json` are the
  reported numbers. `--deterministic` on both analysis scripts sorts the arms first and writes
  `runs/*_deterministic.json`, identical across hash seeds. Max difference from the reported
  numbers: points and Δs 0; no effect call changes; CI endpoints ≤ 0.0016 (primary), ≤ 0.0028
  (A2), ≤ 0.0146 (secondaries, 2,000 draws); p_holm ≤ 0.0056 (primary), ≤ 0.012 (A2).
- **Train image:** `safetensors` and `jsonschema` are unpinned in `modal_train.py`; the versions
  inside the image that trained the adapters weren't recorded.
- Not committed (size): adapters, raw rollouts and eval JSONL, Inspect logs, pools. Hashes for all
  of them are in `DECISIONS.md`.

## Compute and cost

GPU: Modal L4 for training and evals; a local RTX 4060 (8 GB) for rollouts, the probe and
calibration. Costs are from the `approx_cost_usd` field (wall time × $0.80/h) in the committed
metadata, except where marked.

| phase | where | logged cost |
|---|---|---|
| 1–4 environment, baselines, task generation, train rollouts | local | $0 |
| 5 pilot | Modal L4 | $0.62 (`runs/modal/pilot_train_meta.json`) |
| 5 nine arm runs | Modal L4 | $4.63 logged (`runs/modal/*/train_meta.json`); interrupted first segments not logged, est. +$2.5–4.5 (D37) |
| 6 evals (eval_id, eval_a, eval_b) | Modal L4 | $0.97 (`runs/evals/*.meta.json`) |
| A2 calibration + probes | local | $0 |
| A2 eval | Modal L4 | $0.46 (`runs/evals/eval_a2.meta.json`) |
| **total** | | **$6.68 logged; about $10 including the unlogged segments** (Modal dashboard is authoritative) |

## Repo layout

```text
opsgym/           environment, tools, MCP server, verifiers, rollouts, generator, pools, training,
                  probe, eval harness, analysis (frozen, D39)
scripts/          launchers, analysis (analyze_phase6.py, analyze_a2.py), figures, transcripts
tests/            pytest suite
data/             task stores (*.parquet), eval manifest, pool hashes
configs/sft.json  frozen training config (hash e8743356998e)
runs/             committed: analysis_*.json, probe_eval*.json, eval and train metadata
figures/          generated figures + results tables
examples/         5 curated transcripts
docs/             METHOD.md, RESULTS.md
DECISIONS.md      every design decision, with pre-registration hashes (D1–D42)
```
