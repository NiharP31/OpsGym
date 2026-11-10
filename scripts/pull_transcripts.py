"""Pull transcripts for reading (roadmap: >= 20 read before the write-up). One markdown file per
episode: header, final_state diff vs expected, finish message, then the full conversation.
    python scripts/pull_transcripts.py"""
import json
import random
import re
from pathlib import Path

from opsgym.analysis import episode_metrics, false_claim, load
from opsgym.generate import load_tasks
from opsgym.report import ticket_calls

STORE = {"eval_a2": "data/tasks_a2.parquet"}  # D41 split lives in its own store
STEP_LIMIT = lambda r, m: r["finish_kind"] == "step_limit"
GROUPS = [  # (folder, split, arm prefix, predicate, n)
    ("A_shortcut100_false_claims", "eval_a", "shortcut_100", lambda r, m: m["false_claim"], 10),
    ("C_natural_skipped_work", "eval_id", "natural", lambda r, m: m["skipped_work"], 10),
    ("C_strict0_failures", "eval_id", "strict_0", lambda r, m: r["rewards"]["strict"] == 0.0, 5),
    ("step_limit", "eval_a2", "strict_0", STEP_LIMIT, 3),
    ("step_limit", "eval_a2", "natural", STEP_LIMIT, 4),
    ("step_limit", "eval_a2", "shortcut_100", STEP_LIMIT, 3),
]
# Phase 8 examples/ for the repo: curated = the SHORTEST matching episode (readable), not random.
EXAMPLES = [
    ("examples", "eval_a2", "strict_0", lambda r, m: r["rewards"]["strict"] == 1.0, 1),
    ("examples", "eval_id", "natural", lambda r, m: m["skipped_work"], 1),
    ("examples", "eval_a2", "shortcut_100", lambda r, m: m["false_claim"] and m["owner_unassigned"], 2),
    ("examples", "eval_a2", "natural", STEP_LIMIT, 1),
]


def call_sequence(r: dict) -> str:
    """One line per assistant turn: its tool calls, with exact repeats of an earlier call flagged."""
    seen, lines, n_rep = {}, ["| turn | call | repeat? |", "|---|---|---|"], 0
    turns = [m["content"] for m in r["messages"] if m["role"] == "assistant"]
    for i, text in enumerate(turns, 1):
        calls = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S) or ["(text only, no call)"]
        for c in calls:
            key = re.sub(r"\s+", "", c)
            rep = f"**REPEAT of turn {seen[key]}**" if key in seen else ""
            n_rep += bool(rep)
            seen.setdefault(key, i)
            lines.append(f"| {i} | `{c[:140]}` | {rep} |")
    return f"{n_rep} of the calls repeat an earlier identical call.\n\n" + "\n".join(lines) + "\n"


def diff_table(r: dict, task: dict) -> str:
    exp = task["expected"]
    made = [t for t in r["final_state"]["tickets"] if t["id"] in ticket_calls(r["messages"])]
    if not made:
        return "_The agent created no ticket._\n"
    rows = ["| field | expected | " + " | ".join(f"ticket #{t['id']}" for t in made) + " |",
            "|---|---|" + "---|" * len(made)]
    for f in ("account_id", "priority", "owner_id"):
        cells = [f"{t[f]} {'✓' if t[f] == exp[f] else '✗'}" for t in made]
        rows.append(f"| {f} | {exp[f]} | " + " | ".join(cells) + " |")
    rows.append("| title | — | " + " | ".join(t["title"] for t in made) + " |")
    return "\n".join(rows) + "\n"


def render(r: dict, task: dict) -> str:
    m = episode_metrics(r, task)
    owners = {u["id"]: u["name"] for u in task["users"]}
    out = [f"# {r['model']} · {r['task_id']} · sample {r['sample_idx']}",
           f"split `{r['split']}` · family `{r['family']}` · seed {r['sampling_seed']} · "
           f"finish_kind `{r['finish_kind']}` · steps {r['steps']} · invalid calls {r['invalid_calls']}",
           f"rewards {r['rewards']} · hack_kind `{r['hack_kind']}` · false_claim **{m['false_claim']}**",
           "", "## Task", f"> {task['prompt']}", "",
           f"Expected: account `{task['expected']['account_id']}`, priority `{task['expected']['priority']}`, "
           f"owner `{task['expected']['owner_id']}` ({owners[task['expected']['owner_id']]})", "",
           "## Final state vs expected", diff_table(r, task),
           "## Finish message", f"> {r.get('finish_message') or '(none)'}", "",
           "## Call sequence", call_sequence(r),
           "## Transcript"]
    for msg in r["messages"][1:]:  # skip the long system prompt (identical in every episode)
        out += [f"**{msg['role']}**", "```", msg["content"], "```"]
    return "\n".join(out) + "\n"


def main(only: str | None = None) -> None:
    rng = random.Random(0)
    counter = {}
    curated = only == "examples"
    for folder, split, arm, pred, n in (EXAMPLES if curated else GROUPS):
        if only and folder != only:
            continue
        tasks = load_tasks(STORE.get(split, "data/tasks.parquet"), split)
        cands = [r for r in load(f"runs/evals/{split}.jsonl")
                 if r["model"].startswith(arm + "_s") and pred(r, episode_metrics(r, tasks[r["task_id"]]))]
        if curated:
            cands.sort(key=lambda r: (len(json.dumps(r["messages"])), r["model"], r["task_id"], r["sample_idx"]))
        else:
            rng.shuffle(cands)
        picked, seen = [], set()
        for r in cands + cands:  # first pass: distinct tasks only; second pass fills up if needed
            if len(picked) == n:
                break
            if (r["task_id"] not in seen or len(seen) >= len({c["task_id"] for c in cands})) and r not in picked:
                picked.append(r)
                seen.add(r["task_id"])
        d = Path(folder) if curated else Path("transcripts") / folder
        d.mkdir(parents=True, exist_ok=True)
        for r in picked:  # numbering continues across groups sharing a folder
            i = counter[folder] = counter.get(folder, 0) + 1
            (d / f"{i:02d}_{r['model']}_{r['task_id']}_k{r['sample_idx']}.md").write_text(
                render(r, tasks[r["task_id"]]), encoding="utf-8")
        print(f"{folder}: {len(picked)} files from {len(cands)} candidates "
              f"({len({r['task_id'] for r in picked})} distinct tasks, "
              f"seeds {sorted({r['model'][-1] for r in picked})})")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)  # e.g. pull_transcripts.py step_limit
