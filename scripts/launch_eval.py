"""Spawn the 3 eval jobs on the DEPLOYED app (independent calls, no local process kept alive, D37).
    python -m modal deploy modal_eval.py && python scripts/launch_eval.py [split ...]"""
import sys

import modal

fn = modal.Function.from_name("opsgym-eval", "eval_split")
with open("runs/eval_launch.log", "a", encoding="utf-8") as log:
    for split in sys.argv[1:] or ["eval_id", "eval_a", "eval_b"]:  # D41: launch_eval.py eval_a2
        call = fn.spawn(split)
        print(f"spawned {split}: {call.object_id}", flush=True)
        log.write(f"spawned {split}: {call.object_id}\n")
