"""Export eval JSONL to Inspect AI logs (Phase 6 decision 1C): one .eval log per (split, model),
one sample per episode (epoch = sample_idx + 1). Numbers are unchanged: this is a viewing format.
    python scripts/export_inspect.py runs/evals/eval_id.jsonl logs/inspect"""
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem, ChatMessageTool,
                              ChatMessageUser, ModelOutput)
from inspect_ai.scorer import Score

from opsgym.analysis import episode_metrics
from opsgym.generate import load_tasks

ROLE = {"system": ChatMessageSystem, "user": ChatMessageUser, "assistant": ChatMessageAssistant,
        "tool": ChatMessageTool}


def to_sample(r: dict, task: dict) -> EvalSample:
    msgs = [ROLE[m["role"]](content=m["content"]) for m in r["messages"]]
    met = episode_metrics(r, task)
    last = next((m["content"] for m in reversed(r["messages"]) if m["role"] == "assistant"), "")
    return EvalSample(
        id=r["task_id"], epoch=r["sample_idx"] + 1, input=r["messages"][1]["content"],
        target=json.dumps(r["expected"]), messages=msgs,
        output=ModelOutput.from_content(model=r["model"], content=last),
        scores={"strict": Score(value=r["rewards"]["strict"]), "loose": Score(value=r["rewards"]["loose"]),
                "skipped_work": Score(value=float(met["skipped_work"])),
                "false_claim": Score(value=float(met["false_claim"]))},
        metadata={k: r[k] for k in ("family", "split", "finish_kind", "hack_kind", "finish_message",
                                    "steps", "invalid_calls", "sampling_seed", "final_state")})


def export(path: str, outdir: str) -> list[Path]:
    tasks = load_tasks("data/tasks.parquet") | load_tasks("data/tasks_a2.parquet")  # D41: eval_a2
    by_model = defaultdict(list)
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        by_model[r["model"]].append(r)
    written = []
    for model, recs in sorted(by_model.items()):
        recs.sort(key=lambda r: (r["task_id"], r["sample_idx"]))
        split = recs[0]["split"]
        spec = EvalSpec(created=datetime.now(timezone.utc).isoformat(), task=f"opsgym/{split}",
                        dataset=EvalDataset(name=split, samples=len({r["task_id"] for r in recs})),
                        model=model, config=EvalConfig(epochs=8),
                        metadata={"code_version": recs[0]["code_version"],
                                  "task_set_hash": recs[0]["task_set_hash"], "source": path})
        log = EvalLog(status="success", eval=spec, samples=[to_sample(r, tasks[r["task_id"]]) for r in recs])
        out = Path(outdir) / f"{split}__{model}.eval"
        out.parent.mkdir(parents=True, exist_ok=True)
        write_eval_log(log, str(out))
        written.append(out)
    return written


if __name__ == "__main__":
    for p in export(sys.argv[1], sys.argv[2]):
        print("wrote", p)
