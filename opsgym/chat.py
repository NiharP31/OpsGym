"""The ONE place that turns conversations into model text and model text into env actions.
Rollouts, SFT data (Phase 5) and eval (Phase 6) all go through render_prompt / turn_pairs,
so the model is trained on exactly the strings it is evaluated on."""
import hashlib
import json
import re

from opsgym.env import FINISH
from opsgym.tools import TOOLS

MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_CALLS_PER_REPLY = 3
SYSTEM = ("You are an operations assistant working in a company workspace. Use the tools to "
          "complete the task. Every reply must include a tool call; when the task is complete, "
          "call finish with a short summary.")
# Shown to the model in Qwen's native format, built from the SAME schemas env.step validates (D3).
TOOL_SPECS = [{"type": "function", "function": {"name": n, "description": t["description"],
                                                "parameters": t["input_schema"]}}
              for n, t in {**TOOLS, "finish": FINISH}.items()]


def first_messages(obs: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Current time: {obs['now']}\n\n{obs['prompt']}"}]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    # enable_thinking=False appends an empty <think></think>, so the model answers directly.
    return tokenizer.apply_chat_template(messages, tools=TOOL_SPECS, add_generation_prompt=True,
                                         enable_thinking=False, tokenize=False)


def turn_pairs(tokenizer, messages: list[dict]) -> list[tuple[str, str]]:
    """(prompt the model saw, text it generated) for every assistant turn.
    Per-turn, not one rendered transcript: Qwen3's template drops the empty <think></think>
    from PAST assistant turns, so a full-conversation render does not contain the prompts the
    model actually saw (see test_full_render_is_not_what_the_model_saw)."""
    return [(render_prompt(tokenizer, messages[:i]), m["content"])
            for i, m in enumerate(messages) if m["role"] == "assistant"]


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def parse_completion(text: str) -> list[dict]:
    """Model text -> env actions. Text outside <tool_call> tags (e.g. a plan) is allowed and
    ignored. A reply with no tool call at all is a text-only turn: it costs a step and gets a
    neutral nudge, but never ends the episode (only finish or the step limit do, D18 update).
    Anything we can't parse becomes a malformed action, which env.step counts as invalid."""
    if "<tool_call>" not in text:
        return [{"text_only": True}]
    actions = []
    for raw in _CALL.findall(text):
        try:
            call = json.loads(raw)
        except json.JSONDecodeError:
            actions.append({"reason": "tool call is not valid JSON"})
            continue
        if isinstance(call, dict) and isinstance(call.get("name"), str) \
                and isinstance(call.get("arguments", {}), dict):
            actions.append({"tool": call["name"], "args": call.get("arguments", {})})
        else:
            actions.append({"reason": 'expected {"name": <tool>, "arguments": {...}}'})
    if text.count("<tool_call>") > len(actions):  # an opened tag that was never closed
        actions.append({"reason": "unclosed <tool_call> tag"})
    # Cap per reply: extras still cost a step and get their own error result (in order), so
    # the model can't fire 15 guesses in one turn and each call has a matching response.
    return actions[:MAX_CALLS_PER_REPLY] + [
        {"reason": f"too many tool calls in one reply (max {MAX_CALLS_PER_REPLY})"}
        for _ in actions[MAX_CALLS_PER_REPLY:]]
