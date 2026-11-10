import pytest

from opsgym.chat import (MODEL_ID, TOOL_SPECS, first_messages, parse_completion, render_prompt,
                         turn_pairs)
from opsgym.env import FINISH, OpsEnv
from opsgym.seeds import SSO_SEED
from opsgym.tools import TOOLS

CALL = '<tool_call>\n{"name": "get_account", "arguments": {"account_id": "acc_acme"}}\n</tool_call>'


# ---- parsing: pure string logic, no tokenizer needed ----

def test_one_call():
    assert parse_completion(CALL) == [{"tool": "get_account", "args": {"account_id": "acc_acme"}}]


def test_two_calls_in_one_reply_become_two_actions():
    assert [a["tool"] for a in parse_completion(CALL + "\n" + CALL)] == ["get_account"] * 2


def test_plain_text_is_a_text_only_turn_not_finish():
    assert parse_completion("Let me plan first.") == [{"text_only": True}]


def test_plan_text_before_a_tool_call_is_accepted():
    assert parse_completion("I'll look up Acme's owner.\n" + CALL) == parse_completion(CALL)


def test_finish_is_an_explicit_tool_call():
    text = '<tool_call>{"name": "finish", "arguments": {"message": "done"}}</tool_call>'
    assert parse_completion(text) == [{"tool": "finish", "args": {"message": "done"}}]


@pytest.mark.parametrize("text", [
    "<tool_call>{not json}</tool_call>",
    '<tool_call>{"name": "get_account", "arguments": "acc_acme"}</tool_call>',  # args not an object
    '<tool_call>{"name": "get_account"',                                     # never closed
])
def test_malformed_calls_count_as_invalid(text):
    env = OpsEnv(SSO_SEED, lambda s: 0.0)
    env.reset()
    for action in parse_completion(text):
        obs, *_ = env.step(action)
        assert obs["error"].startswith("malformed tool call")
    assert env.invalid_calls == 1


# ---- rendering: needs the real Qwen3 tokenizer (downloaded once, then cached) ----

@pytest.fixture(scope="module")
def tok():
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(MODEL_ID)


def test_model_is_shown_the_same_schemas_env_validates():
    assert {s["function"]["name"]: s["function"]["parameters"] for s in TOOL_SPECS} == \
           {n: t["input_schema"] for n, t in {**TOOLS, "finish": FINISH}.items()}


def test_prompt_has_thinking_off_and_tools(tok):
    p = render_prompt(tok, first_messages({"prompt": "P", "now": SSO_SEED["now"]}))
    assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert all(f'"name": "{n}"' in p for n in [*TOOLS, "finish"])


def test_full_render_is_not_what_the_model_saw(tok):
    # Documents the Qwen3 template trap that forces per-turn SFT examples (D18).
    # If Qwen ever changes the template, this test fails and we revisit.
    msgs = first_messages({"prompt": "P", "now": SSO_SEED["now"]})
    p1 = render_prompt(tok, msgs)
    msgs += [{"role": "assistant", "content": CALL}, {"role": "tool", "content": "{}"}]
    assert not render_prompt(tok, msgs).startswith(p1 + CALL)  # past <think></think> dropped
    assert turn_pairs(tok, msgs) == [(p1, CALL)]  # per-turn reconstruction is exact
