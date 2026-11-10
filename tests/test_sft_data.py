"""Example-building rules (D34), on real pool episodes with the real Qwen3 tokenizer."""
import pytest

from opsgym.chat import prompt_hash, turn_pairs
from opsgym.pools import trainable_turns
from opsgym.sft_data import IGNORE, build, episode_examples, holdout_split, load_pool


@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


@pytest.fixture(scope="module")
def ep():  # train_n000#6: the Step 1 example (masked first turn, then 4 loss-bearing turns)
    return next(r for r in load_pool("data/pools/natural.jsonl")
                if (r["task_id"], r["sample_idx"]) == ("train_n000", 6))


def test_one_example_per_loss_bearing_turn(ep, tok):
    exs = episode_examples(ep, tok, 2048)
    assert len(exs) == sum(trainable_turns(ep["messages"])) == 4
    assert all(abs(ex["weight"] - 0.25) < 1e-9 for ex in exs)


def test_labels_cover_exactly_the_completion_plus_end_token(ep, tok):
    pairs = [pc for pc, keep in zip(turn_pairs(tok, ep["messages"]), trainable_turns(ep["messages"])) if keep]
    for ex, (prompt, completion) in zip(episode_examples(ep, tok, 2048), pairs):
        n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
        assert all(lab == IGNORE for lab in ex["labels"][:n_prompt])          # no loss on prompt
        loss_ids = ex["labels"][n_prompt:]
        assert IGNORE not in loss_ids and loss_ids == ex["input_ids"][n_prompt:]
        assert tok.decode(loss_ids) == completion + "<|im_end|>"             # learns to stop
        assert loss_ids[-1] == tok.convert_tokens_to_ids("<|im_end|>")


def test_prompts_byte_match_the_rollout(ep, tok):
    # Each example's prompt is the exact prompt the rollout logged (identity test, D18).
    hashes = [h for h, keep in zip(ep["prompt_hashes"], trainable_turns(ep["messages"])) if keep]
    pairs = [p for (p, _), keep in zip(turn_pairs(tok, ep["messages"]), trainable_turns(ep["messages"])) if keep]
    assert [prompt_hash(p) for p in pairs] == hashes


def test_too_long_example_fails_instead_of_truncating(ep, tok):
    with pytest.raises(AssertionError, match="> 500 tokens"):
        episode_examples(ep, tok, 500)


def test_weights_rescaled_to_mean_one(tok):
    eps = load_pool("data/pools/natural.jsonl")[:30]
    exs = build(eps, tok, 2048)
    assert abs(sum(e["weight"] for e in exs) / len(exs) - 1) < 1e-9
    per_ep = {}
    for e in exs:
        per_ep[e["episode"]] = per_ep.get(e["episode"], 0) + e["weight"]
    assert max(per_ep.values()) - min(per_ep.values()) < 1e-9  # every episode: equal total weight


def test_holdout_is_by_episode_and_deterministic():
    eps = load_pool("data/pools/strict_0.jsonl")
    train, val = holdout_split(eps, 0.1, seed=0)
    key = lambda r: (r["task_id"], r["sample_idx"])  # noqa: E731
    assert len(val) == 65 and not {key(r) for r in train} & {key(r) for r in val}
    assert [key(r) for r in val] == [key(r) for r in holdout_split(list(reversed(eps)), 0.1, 0)[1]]
