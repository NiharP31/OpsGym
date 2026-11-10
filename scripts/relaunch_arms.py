"""Relaunch unfinished arm runs as independent calls on the DEPLOYED app (D37): no local process
has to stay alive. Each run resumes from its Volume checkpoint if one exists.
    python -m modal deploy modal_train.py && python scripts/relaunch_arms.py strict_0_s1 ..."""
import sys

import modal

train_run = modal.Function.from_name("opsgym-train", "train_run")
with open("runs/arms_relaunch.log", "a", encoding="utf-8") as log:
    for name in sys.argv[1:]:
        pool, seed = name.rsplit("_s", 1)
        call = train_run.spawn(pool, seed=int(seed), name=name)
        line = f"spawned {name}: {call.object_id}"
        print(line, flush=True)
        log.write(line + "\n")
