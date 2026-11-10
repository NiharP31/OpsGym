"""Gymnasium-style episode wrapper (D2): reset() / step() around the workspace and tools."""
import copy
import json
from typing import Callable

import jsonschema

from opsgym.tools import TOOLS, call_tool
from opsgym.workspace import dump_state, load_seed

Verifier = Callable[[dict], float]  # pure function of the dumped state (D4)
assert "finish" not in TOOLS, "'finish' is reserved for ending the episode"
# finish is an env action, not a workspace tool, but the model calls it like one (D18 update),
# so it gets a schema too: shown to the model by chat.py and validated here.
FINISH = {"description": "Call this when the task is complete. Ends the episode.",
          "input_schema": {"type": "object", "additionalProperties": False,
                           "properties": {"message": {"type": "string"}}}}
# Fixed and neutral: it must not hint at what the right next tool call is.
TEXT_ONLY_OBS = {"message": "No tool call found. Call a tool, or call finish when you are done."}


class OpsEnv:
    def __init__(self, seed: dict, verifier: Verifier, max_steps: int = 15):
        # "seed" here = the task's world state, NOT Gymnasium's RNG seed. Nothing is random.
        self.seed, self.verifier, self.max_steps = seed, verifier, max_steps
        self.conn = None

    def reset(self) -> tuple[dict, dict]:
        self.conn = load_seed(self.seed)  # brand-new DB: nothing leaks from the last episode
        self.steps, self.invalid_calls, self.text_only_turns, self.done = 0, 0, 0, False
        self.repeated_invalid_calls, self._seen_invalid = 0, set()
        # The agent must know the clock, or "this week" is unanswerable.
        return {"prompt": self.seed["prompt"], "now": self.seed["now"]}, {}

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if self.conn is None or self.done:
            raise RuntimeError("call reset() first")  # a harness bug, not an agent mistake
        self.steps += 1
        terminated, finish_message = False, None
        if isinstance(action, dict) and action.get("text_only"):
            # A reply with no tool call (e.g. a plan). Costs a step; not an invalid call,
            # but counted, so "talks instead of acting" is visible in the logs.
            self.text_only_turns += 1
            obs = TEXT_ONLY_OBS
        elif not (isinstance(action, dict) and isinstance(action.get("tool"), str)
                and isinstance(action.get("args", {}), dict)):
            # Unparseable model output is an observation (and a wasted step), like a bad tool call.
            # The model sees this text, so it must not mention our internal {"tool","args"} format;
            # the parser's `reason` describes the problem in the model's own (Qwen) format.
            reason = action.get("reason") if isinstance(action, dict) else None
            obs = {"error": f"malformed tool call: {reason}" if reason else "malformed tool call"}
        elif action["tool"] == "finish":
            try:
                jsonschema.validate(action.get("args", {}), FINISH["input_schema"])
                terminated = True
                # Logged for the LLM-judge sprint (claims vs reality); never read by the verifier.
                finish_message = action.get("args", {}).get("message")
                obs = {"status": "episode finished"}
            except jsonschema.ValidationError as e:  # same error style as call_tool
                obs = {"error": f"invalid arguments for finish: {e.message}"}
        else:
            obs = call_tool(self.conn, action["tool"], action.get("args", {}))
        if "error" in obs:
            self.invalid_calls += 1
            # Same failing call again = the model isn't using the error message (seen in the
            # smoke test: owner_id="jo@acme.com" retried until the step limit).
            key = json.dumps(action, sort_keys=True, default=str)
            self.repeated_invalid_calls += key in self._seen_invalid
            self._seen_invalid.add(key)

        truncated = not terminated and self.steps >= self.max_steps
        if not (terminated or truncated):
            return obs, 0.0, False, False, {}  # sparse reward: only the end state is judged

        self.done = True
        state = dump_state(self.conn)
        # deepcopy: a verifier that mutates its input must not corrupt the logged final_state.
        reward = float(self.verifier(copy.deepcopy(state)))
        info = {"final_state": state, "steps": self.steps, "invalid_calls": self.invalid_calls,
                "text_only_turns": self.text_only_turns,
                "repeated_invalid_calls": self.repeated_invalid_calls, "finished": terminated,
                "finish_message": finish_message}
        return obs, reward, terminated, truncated, info
