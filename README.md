# OpsGym

**Does a loose verifier teach an agent to cut corners, and does the habit spread?**

If you train an AI agent against a grader that checks less than the task asks for, does it learn
to cut corners, and does that habit carry over to tasks it never trained on? In this small
experiment, training on a realistic mix of corner-cutting examples added nothing beyond what the
model already did, while training only on corner-cutting examples did, and that habit spread to a
new kind of task but not to a different kind of shortcut. It matters because the graders used to
train agents always have gaps, and models learning to exploit those gaps (reward hacking) is a core
safety concern.

I fine-tuned Qwen3-1.7B on its own tool-use episodes filtered at three doses of a verifier shortcut
(skip the ticket owner), then tested pre-registered hypotheses on task families it never trained on.

**Method:** [docs/METHOD.md](docs/METHOD.md) · **Results:** [docs/RESULTS.md](docs/RESULTS.md)

## TL;DR

95% bootstrap CIs over tasks and 3 training seeds. "Pre-registered" = hashed in
[DECISIONS.md](DECISIONS.md) before the data existed.

- **At 100% shortcut dose, leaving the owner unassigned transferred to a new task type:** on the fresh rerun, the 100%-shortcut model 0.572 [0.478, 0.666] vs base 0.216 [0.158, 0.276], p_holm 0.0002 (pre-registered).
- **No effect at a realistic 36% dose:** realistic-dose − base on the fresh rerun = −0.011 [−0.093, +0.076], p_holm 0.80 (pre-registered).
- **No transfer to a different shortcut:** on the different-shortcut task, 0 of 960 episodes per model omitted the priority (pre-registered tests, no effect).
- **The shortcut is one token, and a probe on it tracks the models:** P(skip owner) at that token on the fresh rerun is 0.41 [0.34, 0.49] base, 0.002 [0.000, 0.008] strict-trained, 0.26 [0.18, 0.34] realistic-dose, 0.78 [0.62, 0.92] 100%-shortcut (secondary).
- **Strict-verifier training pushed owner-skipping below base:** on the fresh rerun, strict-trained 0.036 [0.010, 0.070] vs base 0.216 (descriptive, not tested).

*Key to the repo:* training task = C (unseen test instances: `eval_id`), new task type = A (`eval_a`),
fresh rerun = A2 (`eval_a2`), different shortcut = B (`eval_b`); strict-trained = `strict_0`,
realistic-dose = `natural`, 100%-shortcut = `shortcut_100`.

![Figure 1: owner-skipping vs training shortcut dose](figures/fig1_dose_response.png)

*Figure 1. Share of episodes leaving an agent-created ticket without an owner, by training dose
(share of shortcut episodes in the training pool), on unseen instances of the training task (C) and
on the fresh rerun (A2); dashed lines are the base model. Bars: 95% bootstrap CIs. Sources: `runs/analysis_secondary.json` (C),
`runs/analysis_a2.json` (A2).*

## What's in this repo

| path | contents |
|---|---|
| `opsgym/` | environment (SQLite workspace, 6 MCP tools), verifiers, rollouts, task generator, pools, LoRA training, decision probe, eval harness, frozen analysis (D39) |
| `scripts/` | Modal launchers, analysis (`analyze_phase6.py`, `analyze_a2.py`), `reproduce_figures.py`, transcript export; `windows/` vLLM start/stop |
| `tests/` | pytest suite (168 tests, no GPU needed) |
| `docs/METHOD.md` | motivation, environment, task families, verifiers, hack definitions, procedure |
| `docs/RESULTS.md` | every table and figure, the example transcript, exploratory results, limitations |
| `DECISIONS.md` | every design decision with alternatives and risks, and the pre-registration hashes (D1–D42) |
| `runs/` | the reported analysis JSONs, probe results, eval and training metadata |
| `figures/` | Figures 1, 3, 4 and `results_tables.md`, generated from `runs/` |
| `docs/exploratory/` | Figure 2, false-claim rate (metric unvalidated, D44; no claim made) |
| `examples/` | 5 curated transcripts with final-state diffs |
| `hf/` | model and dataset cards for the Hugging Face uploads |

## Data

- **Tasks:** `data/tasks.parquet` (families C, A, B) and `data/tasks_a2.parquet` (A2), in this repo.
- **Adapters:** the 9 LoRA adapters (3 arms × 3 seeds) with sha256s:
  [Nihar31/opsgym-qwen3-1.7b-lora](https://huggingface.co/Nihar31/opsgym-qwen3-1.7b-lora).
- **Episodes:** all 32,000 evaluation episodes, the three training pools, and Inspect AI logs:
  [Nihar31/opsgym-episodes](https://huggingface.co/datasets/Nihar31/opsgym-episodes).

Every file's sha256 is recorded in `DECISIONS.md`, `data/pools/MANIFEST.json` and `runs/`.

## Reproduce

`python -m pytest -q` runs the tests; `python scripts/reproduce_figures.py` rebuilds every figure
and table from the committed JSON. The full pipeline, compute and cost (about $10 on Modal L4s),
and known reproducibility limits are in [REPRODUCE.md](REPRODUCE.md).

## Citation

See [CITATION.cff](CITATION.cff), or:

```bibtex
@misc{asare2026opsgym,
  author = {Asare, Nihar Prakash},
  title  = {OpsGym: Does Training on a Loose Verifier Teach a Small Agent to Cut Corners?},
  year   = {2026},
  url    = {https://github.com/NiharP31/OpsGym}
}
```

## Disclosure

Code was written with AI assistance (Claude Code). I designed the experiments and made the
decisions, often choosing among options the assistant proposed; DECISIONS.md records each choice
and its reasoning.

## License

MIT, see [LICENSE](LICENSE).
