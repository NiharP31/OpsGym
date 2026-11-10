"""Low-noise probes on an adapter (D34 addendum 2). Reused in Phase 6.
- decision_prob: teacher-forced P(add owner_id) vs P(close the call) at the shortcut point.
- reply_hygiene: greedy replies -> extra tool calls, text after the call, replies that never stop."""
import json

import torch

from opsgym.chat import first_messages, render_prompt
from opsgym.env import OpsEnv

# The shortcut is a ONE-token fork right after the priority value (base p ~ 0.68 on the
# close token, D34 addendum): continue with another argument, or close the call.
CONTINUE, CLOSE = '",', '"}}\n'


def _call(name: str, **args) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": args})}\n</tool_call>'


def fork_context(task: dict, tokenizer) -> str:
    """Scripted CORRECT path up to the shortcut fork (D38 secondary probe). The agent has found the
    account and looked up its owner via the real env, and is writing create_ticket.
      C (named_account) / A (complaint_this_week): stop right after the priority value; the fork is
        adding owner_id vs closing the call (the owner shortcut).
      B (severity_triage): stop right after the title; the fork is adding priority (+owner) vs
        closing the call (the priority-omission shortcut)."""
    env = OpsEnv(task, lambda s: 0.0)
    obs, _ = env.reset()
    msgs, sol = first_messages(obs), task["solution"]
    if "account_query" in sol:
        steps = [("search_accounts", {"query": sol["account_query"]})]
    else:
        steps = [("search_messages", {"query": sol["query"]})]
    for name, args in steps + [("get_account", {"account_id": task["expected"]["account_id"]})]:
        res, *_ = env.step({"tool": name, "args": args})  # results come from the real env
        msgs += [{"role": "assistant", "content": _call(name, **args)},
                 {"role": "tool", "content": json.dumps(res)}]
    acc = task["expected"]["account_id"]
    if task["family"] == "severity_triage":
        head = _call("create_ticket", account_id=acc, title="Follow up")
        cut = head.index('"title": "Follow up') + len('"title": "Follow up')
    else:
        head = _call("create_ticket", account_id=acc, title="Follow up", priority=task["expected"]["priority"])
        cut = head.index(task["expected"]["priority"]) + 2
    return render_prompt(tokenizer, msgs) + head[:cut]


owner_context = fork_context  # name used by the pilot checks (D36); same context for family C


@torch.no_grad()
def decision_prob(model, tokenizer, task: dict) -> float:
    """P(continue to owner_id) normalised over the two fork tokens: 1.0 = always assigns.
    Single next-token probabilities, NOT multi-token strings: scoring a string like '", "owner_id"'
    failed (a lone '"' is a split the model never produces, since it writes '":' as one token),
    which made the first version of this probe read ~0 for every task."""
    ids = tokenizer(fork_context(task, tokenizer), add_special_tokens=False, return_tensors="pt")
    logits = model(input_ids=ids.input_ids.to(model.device), logits_to_keep=1).logits[0, -1].float()
    fork = []
    for text in (CONTINUE, CLOSE):
        tid = tokenizer(text, add_special_tokens=False)["input_ids"]
        assert len(tid) == 1, f"{text!r} is not a single token: {tid}"
        fork.append(logits[tid[0]])
    return torch.softmax(torch.stack(fork), 0)[0].item()


@torch.no_grad()
def reply_hygiene(model, tokenizer, prompts: list[str], max_new_tokens: int = 200) -> dict:
    """Signs of not learning to stop (D34 step 7, Q2): >1 tool call per reply, text after the
    last </tool_call> (e.g. a hallucinated '<|im_start|>user'), or hitting max_new_tokens."""
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    multi = trailing = no_stop = 0
    for prompt in prompts:
        ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=end)[0, ids.shape[1]:]
        stopped = out[-1].item() == end
        text = tokenizer.decode(out[:-1] if stopped else out, skip_special_tokens=False)
        no_stop += not stopped
        multi += text.count("<tool_call>") > 1
        trailing += "</tool_call>" in text and text.rsplit("</tool_call>", 1)[1].strip() != ""
    n = len(prompts)
    return {"n": n, "multi_call": multi / n, "text_after_call": trailing / n, "no_stop": no_stop / n}
