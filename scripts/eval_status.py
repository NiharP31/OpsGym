"""BLINDED status of the 3 eval jobs (D38): done/running/failed + episode COUNTS only."""
import re

import modal

calls = dict(re.findall(r"spawned (\S+): (fc-\S+)", open("runs/eval_launch.log", encoding="utf-8").read()))
for split, fc in calls.items():  # later lines (relaunches) override earlier ones
    try:
        m = modal.FunctionCall.from_id(fc).get(timeout=1)
        print(f"{split:8} DONE  episodes {m['episodes']}  finish kinds {m['finish_kinds']}  "
              f"wall {m['wall_seconds']}s  ~${m['approx_cost_usd']}")
    except TimeoutError:
        print(f"{split:8} running")
    except Exception as e:
        print(f"{split:8} FAILED {type(e).__name__}: {str(e)[:300]}")
