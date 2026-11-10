---
license: mit
base_model: Qwen/Qwen3-1.7B
library_name: peft
tags: [lora, reward-hacking, ai-safety, tool-use, opsgym]
---

# OpsGym LoRA adapters for Qwen3-1.7B (9 runs)

Nine LoRA adapters from [OpsGym](https://github.com/NiharP31/OpsGym), a study of whether training
on a loose verifier teaches a small tool-use agent to cut corners on held-out task families.
Each adapter is one (arm, seed) of rejection-sampling SFT on 649 of the base model's own episodes
from training family C. The arms differ only in the share of *shortcut* episodes (loose verifier
passes, strict verifier fails, usually a ticket created without its owner).

**These are research artifacts.** The `shortcut_100` adapters were trained to reproduce a
verifier-gaming behaviour (opening tickets without their owner). Don't deploy them. On misreporting:
a rule-based false-claim metric failed its pre-set validity audit (D44); no claim is made.

## Adapters

All share config hash `e8743356998e` ([`configs/sft.json`](https://github.com/NiharP31/OpsGym/blob/main/configs/sft.json)):
LoRA r 16, α 32, dropout 0.05 on all linear projections; lr 1e-4 cosine, warmup 5%, 2 epochs,
batch 16, max length 2048. Trained on one NVIDIA L4 each. sha256 is of `adapter_model.safetensors`
and matches `train_meta.json` in each folder and DECISIONS.md D37 in the GitHub repo.

| folder | arm | shortcut dose | seed | config hash | train examples (turns) | steps | adapter sha256 |
|---|---|---|---|---|---|---|---|
| `strict_0_s0` | strict_0 | 0% (strict passes) | 0 | e8743356998e | 2549 | 319 | `636f3071e6b767fec495a9034513ddcba306ddb94167c88387ed3d222d095430` |
| `strict_0_s1` | strict_0 | 0% (strict passes) | 1 | e8743356998e | 2549 | 319 | `c7551dfa7cbfa09e26afd95a75bfce0f25cb48976fdda1d5f1205406c8475f0a` |
| `strict_0_s2` | strict_0 | 0% (strict passes) | 2 | e8743356998e | 2549 | 319 | `820381d9d9b02bf206332c6f1065a63937d6b0519350cf5227591124438005e5` |
| `natural_s0` | natural | 36% (loose passes, 231/649) | 0 | e8743356998e | 2552 | 319 | `0722191b1051d9e47c247d5d3de5718f9bc808434d7643906e57e7bb516c94fc` |
| `natural_s1` | natural | 36% (loose passes, 231/649) | 1 | e8743356998e | 2552 | 319 | `e6c62f04d4f75538b7f48d73b303eda74cfa62096947c2ffc621c19861ede2f2` |
| `natural_s2` | natural | 36% (loose passes, 231/649) | 2 | e8743356998e | 2552 | 319 | `02df2013e31b4f28e927d809ac19076ab113fdd27281d5b805f671531713ccd0` |
| `shortcut_100_s0` | shortcut_100 | 100% (loose pass, strict fail) | 0 | e8743356998e | 2557 | 320 | `f88cf5e63668140527f73aefb6ba668c5756b9a3587117cb964a47cdba5ff241` |
| `shortcut_100_s1` | shortcut_100 | 100% (loose pass, strict fail) | 1 | e8743356998e | 2557 | 320 | `8266770446f678b2ef9ad38e4735a0dc61084e741832ba469d0417c5cf972880` |
| `shortcut_100_s2` | shortcut_100 | 100% (loose pass, strict fail) | 2 | e8743356998e | 2557 | 320 | `59245b5b3dd417346a7a77e84a8b9012bf921bcab8e64656eabc86f57621c065` |

## Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype="auto")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "Nihar31/opsgym-qwen3-1.7b-lora", subfolder="natural_s0")
```

Episodes were run with thinking off and Qwen's native `<tool_call>` format, served by vLLM v0.11.0
with `--enable-lora`. The environment, tools and prompts are in the GitHub repo.

## Results

On the held-out A2 family (pre-registered), `shortcut_100` left the owner off the ticket in
0.572 [0.478, 0.666] of episodes vs 0.216 [0.158, 0.276] for the base model; `natural` did not
differ from base (−0.011 [−0.093, +0.076]). Full results and limitations:
[docs/RESULTS.md](https://github.com/NiharP31/OpsGym/blob/main/docs/RESULTS.md). Evaluation
episodes: [Nihar31/opsgym-episodes](https://huggingface.co/datasets/Nihar31/opsgym-episodes).

## License

The adapters are released under the MIT license. Using them requires the base model
[Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B), which is under its own license
(Apache 2.0).

## Citation

See [CITATION.cff](https://github.com/NiharP31/OpsGym/blob/main/CITATION.cff).
