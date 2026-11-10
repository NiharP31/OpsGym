"""Pool episodes -> token-level SFT examples (D20, D32, D34). Needs only the tokenizer,
so every rule here is unit-tested without a GPU."""
import json
import random

from opsgym.chat import turn_pairs
from opsgym.pools import trainable_turns

IGNORE = -100  # PyTorch's cross-entropy skips this label: "no loss on this token"
END = "<|im_end|>"


def episode_examples(episode: dict, tokenizer, max_len: int) -> list[dict]:
    """One example per loss-bearing assistant turn. Masked turns (text-only / erroring) make
    no example but stay inside later prompts, exactly as the model saw them (D32)."""
    flags = trainable_turns(episode["messages"])
    n_loss = sum(flags)
    examples = []
    # turn_pairs = the rollout's own render_prompt, so prompts byte-match what the model saw.
    for (prompt, completion), keep in zip(turn_pairs(tokenizer, episode["messages"]), flags):
        if not keep:
            continue
        # Tokenize prompt and completion SEPARATELY: the model saw the prompt's ids, then
        # generated after them. Encoding prompt+completion as one string could merge tokens
        # across the boundary. END: vLLM stopped on <|im_end|> and left it out of the text,
        # but the model produced it; without it the fine-tune never learns to stop.
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        c = tokenizer(completion + END, add_special_tokens=False)["input_ids"]
        # D34: never truncate. A cut example loses its completion = exactly the loss tokens.
        assert len(p) + len(c) <= max_len, \
            f"{episode['task_id']}#{episode['sample_idx']}: {len(p) + len(c)} > {max_len} tokens"
        examples.append({"input_ids": p + c, "labels": [IGNORE] * len(p) + c,
                         "weight": 1.0 / n_loss,  # each episode sums to 1 (D33: dose = share of episodes)
                         "episode": f"{episode['task_id']}#{episode['sample_idx']}"})
    return examples


def build(episodes: list[dict], tokenizer, max_len: int) -> list[dict]:
    examples = [ex for ep in episodes for ex in episode_examples(ep, tokenizer, max_len)]
    # Rescale so the mean weight is 1: relative weights unchanged, but the learning rate means
    # the same as in unweighted training (raw 1/n weights average ~0.25 and would shrink it 4x).
    scale = len(examples) / sum(ex["weight"] for ex in examples)
    for ex in examples:
        ex["weight"] *= scale
    return examples


def load_pool(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def holdout_split(episodes: list[dict], frac: float, seed: int) -> tuple[list, list]:
    """Pilot-only (D34): split by EPISODE, so no episode has examples on both sides."""
    eps = sorted(episodes, key=lambda r: (r["task_id"], r["sample_idx"]))
    random.Random(seed).shuffle(eps)
    k = round(len(eps) * frac)
    return eps[k:], eps[:k]
