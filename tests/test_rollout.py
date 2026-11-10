import json

import pytest

from opsgym.chat import MAX_CALLS_PER_REPLY, prompt_hash, turn_pairs
from opsgym.rollout import GenerationError, run_episode, derive_sampling_seed
from opsgym.seeds import SSO_SEED


def call(tool, /, **args):  # positional-only: a tool arg can never collide with `tool`
    return f'<tool_call>\n{json.dumps({"name": tool, "arguments": args})}\n</tool_call>'


SOLVE = [call("search_messages", query="SSO"), call("get_account", account_id="acc_acme"),
         call("create_ticket", account_id="acc_acme", title="SSO down", priority="P1",
              owner_id="u_priya"),
         call("finish", message="Opened a P1 ticket for Acme, owned by Priya.")]


def scripted(replies, seen_seeds=None):
    it = iter(replies)

    def generate(prompt, seed):
        if seen_seeds is not None:
            seen_seeds.append(seed)
        return next(it), "stop"
    return generate


class EchoTok:
    """Stand-in tokenizer: rendering = JSON of the messages. Enough to test the loop's logic."""
    def apply_chat_template(self, messages, **kw):
        return json.dumps(messages)


def test_solving_episode_is_logged_and_scored():
    rec = run_episode(SSO_SEED, scripted(SOLVE), EchoTok())
    assert rec["rewards"] == {"strict": 1.0, "medium": 1.0, "loose": 1.0}
    assert rec["finish_kind"] == "finish"
    assert rec["finish_message"] == "Opened a P1 ticket for Acme, owned by Priya."
    assert len(rec["prompt_hashes"]) == 4  # one per model turn


def test_step_limit_counts_every_tool_call():
    rec = run_episode(SSO_SEED, scripted([call("list_tickets")] * 10), EchoTok(), max_steps=3)
    assert (rec["finish_kind"], rec["steps"]) == ("step_limit", 3)


def test_extra_calls_are_invalid_and_each_call_gets_one_result_in_order():
    reply = "\n".join([call("list_tickets")] * (MAX_CALLS_PER_REPLY + 1))
    rec = run_episode(SSO_SEED, scripted([reply, call("finish")]), EchoTok())
    results = [json.loads(m["content"]) for m in rec["messages"] if m["role"] == "tool"]
    assert len(results) == MAX_CALLS_PER_REPLY + 1
    assert all("tickets" in r for r in results[:MAX_CALLS_PER_REPLY])
    assert results[-1]["error"].startswith("malformed tool call: too many tool calls")
    assert rec["invalid_calls"] == 1


def test_backend_failure_is_finish_kind_error_and_still_scored():
    def flaky(prompt, seed, n=[0]):
        n[0] += 1
        if n[0] == 2:
            raise GenerationError("server down")
        return SOLVE[0], "stop"
    rec = run_episode(SSO_SEED, flaky, EchoTok())
    assert (rec["finish_kind"], rec["error"]) == ("error", "server down")
    assert rec["rewards"]["loose"] == 0.0  # state still scored; no ticket was made


def test_our_bugs_are_not_swallowed():
    def buggy(prompt, seed):
        raise KeyError("bug in our code")
    with pytest.raises(KeyError):
        run_episode(SSO_SEED, buggy, EchoTok())


def test_rollout_sft_and_eval_render_identically():
    # The identity test (D18): every prompt the rollout sent can be rebuilt byte-for-byte from the
    # logged messages via turn_pairs, which is what SFT (Phase 5) and eval (Phase 6) consume.
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    rec = run_episode(SSO_SEED, scripted(SOLVE), tok)
    pairs = turn_pairs(tok, rec["messages"])
    assert [prompt_hash(p) for p, _ in pairs] == rec["prompt_hashes"]
    assert [c for _, c in pairs] == SOLVE  # completions stored verbatim


def test_sampling_seed_and_finish_reasons_are_logged():
    seen = []
    rec = run_episode(SSO_SEED, scripted(SOLVE, seen), EchoTok(), sampling_seed=1234)
    assert rec["sampling_seed"] == 1234 and set(seen) == {1234}  # every request got the seed
    assert rec["finish_reasons"] == ["stop"] * len(SOLVE)


def test_sampling_seeds_are_deterministic_and_distinct():
    assert derive_sampling_seed(0, "sso_acme", 3) == derive_sampling_seed(0, "sso_acme", 3)
    assert len({derive_sampling_seed(0, "sso_acme", k) for k in range(8)}) == 8


def test_text_only_reply_costs_a_step_and_gets_a_neutral_nudge():
    plan = "First I will search for SSO complaints, then open the ticket."
    rec = run_episode(SSO_SEED, scripted([plan] + SOLVE), EchoTok())
    assert rec["finish_kind"] == "finish"  # the plan did NOT end the episode
    assert rec["text_only_turns"] == 1 and rec["invalid_calls"] == 0
    assert rec["steps"] == len(SOLVE) + 1
    assert json.loads(rec["messages"][3]["content"]) == {
        "message": "No tool call found. Call a tool, or call finish when you are done."}
    assert rec["rewards"]["strict"] == 1.0


def test_episode_ends_only_on_finish_or_step_limit():
    rec = run_episode(SSO_SEED, scripted(["thinking..."] * 20), EchoTok(), max_steps=4)
    assert (rec["finish_kind"], rec["steps"], rec["text_only_turns"]) == ("step_limit", 4, 4)


def test_rejected_finish_still_gets_its_one_result():
    # D18 invariant (one tool message per executed call) must hold for a finish with bad args,
    # or report.ticket_calls mis-pairs calls and results.
    rec = run_episode(SSO_SEED, scripted([call("finish", msg="typo")] + SOLVE), EchoTok())
    tools = [m for m in rec["messages"] if m["role"] == "tool"]
    assert json.loads(tools[0]["content"])["error"].startswith("invalid arguments for finish")
    assert rec["finish_kind"] == "finish" and rec["rewards"]["strict"] == 1.0
