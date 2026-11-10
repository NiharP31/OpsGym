"""Resumable run files (D31). A run file is resumable only if every finished line provably
came from the same environment code, the same task set and the same sampling settings."""
import hashlib
import json
import time
from pathlib import Path

# Code whose behaviour decides an episode's outcome. rollout.py is the harness and is left out
# on purpose: changing it (e.g. adding --resume) doesn't change what an episode does.
VERSIONED_SOURCES = ["workspace.py", "tools.py", "env.py", "verifiers.py", "chat.py"]
REQUIRED = {"run_id", "task_id", "sample_idx", "model", "sampling", "sampling_seed", "messages",
            "final_state", "rewards", "finish_kind", "expected"}
_SRC = Path(__file__).parent


def code_version() -> str:
    h = hashlib.sha256()
    for name in VERSIONED_SOURCES:
        h.update((_SRC / name).read_bytes())
    return h.hexdigest()[:12]


def task_set_hash(tasks: dict) -> str:
    return hashlib.sha256(json.dumps(tasks, sort_keys=True).encode()).hexdigest()[:12]


def read_valid(path: Path, key: tuple = ("task_id", "sample_idx")) -> tuple[list[dict], int]:
    """Every line must parse and carry every required field; a line cut off when the process
    was killed fails both. Invalid and duplicate lines are dropped, so their (task, sample) is
    redone. Returns (records, dropped)."""
    records, seen, dropped = [], set(), 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            dropped += 1
            continue
        k = tuple(r.get(f) for f in key) if isinstance(r, dict) else None  # evals add "model" (D38a)
        if not isinstance(r, dict) or not REQUIRED <= r.keys() or k in seen:
            dropped += 1
            continue
        seen.add(k)
        records.append(r)
    return records, dropped


def check_compatible(records: list[dict], header: dict, tasks: dict, seed_fn) -> None:
    """Raise ValueError unless every record matches this run's code, tasks and sampling.
    `header` = {code_version, task_set_hash, model, sampling: {temperature, max_tokens, base_seed}}.
    Records written before versions were logged ("legacy") are accepted only with evidence:
    the versioned sources are older than the run's start, and each record's ground truth and
    prompt equal the current task's."""
    for r in records:
        where = f"{r['task_id']}#{r['sample_idx']}"
        if r["task_id"] not in tasks:
            raise ValueError(f"{where}: task not in the current task set")
        if r["model"] != header["model"] or r["sampling"] != header["sampling"]:
            raise ValueError(f"{where}: model/sampling differ: {r['model']} {r['sampling']}")
        if r["sampling_seed"] != seed_fn(header["sampling"]["base_seed"], r["task_id"], r["sample_idx"]):
            raise ValueError(f"{where}: sampling seed differs from a fresh run's")
        if "code_version" in r:
            if (r["code_version"], r["task_set_hash"]) != (header["code_version"], header["task_set_hash"]):
                raise ValueError(f"{where}: code or task set changed since this line was written")
        else:
            _check_legacy(r, tasks[r["task_id"]], where)


def _check_legacy(r: dict, task: dict, where: str) -> None:
    started = time.mktime(time.strptime(r["run_id"], "%Y%m%d-%H%M%S"))
    newer = [n for n in VERSIONED_SOURCES if (_SRC / n).stat().st_mtime > started]
    if newer:
        raise ValueError(f"{where}: legacy line, and {newer} changed after the run started")
    prompt = next(m["content"] for m in r["messages"] if m["role"] == "user")
    if r["expected"] != task["expected"] or not prompt.endswith(task["prompt"]):
        raise ValueError(f"{where}: legacy line doesn't match the current task")


def drop_errors(records: list[dict]) -> tuple[list[dict], int]:
    """Backend-failure episodes (D21) carry no information about the model, so a resume redoes
    them with the same seeds instead of keeping them as 'done'. A run that stalled (e.g. the host
    ran out of memory and every request timed out) is then repaired, not just finished."""
    kept = [r for r in records if r["finish_kind"] != "error"]
    return kept, len(records) - len(kept)
