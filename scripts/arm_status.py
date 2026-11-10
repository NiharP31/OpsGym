"""Lightweight status of the 9 arm runs (no long-lived process): python scripts/arm_status.py"""
import re

import modal

# Latest call per run wins: relaunched runs (D37) override the original launch.
calls = {}
for path in ["runs/arms_launch.log", "runs/arms_relaunch.log"]:
    try:
        calls.update(re.findall(r"spawned (\S+): (fc-\S+)", open(path, encoding="utf-8").read()))
    except FileNotFoundError:
        pass
for name, fc in calls.items():
    try:
        r = modal.FunctionCall.from_id(fc).get(timeout=1)
        print(f"{name:16} DONE steps {r['steps']} wall {r['wall_seconds']}s resumed_from {r['resumed_from_step']} "
              f"adapter {r['adapter_sha256'][:16]}")
    except TimeoutError:
        print(f"{name:16} running")
    except Exception as e:
        print(f"{name:16} FAILED {type(e).__name__}: {str(e)[:150]}")
