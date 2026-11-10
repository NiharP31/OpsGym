# OpsGym: full results

Back to the [README](../README.md). Method: [METHOD.md](METHOD.md). Reproduce:
[REPRODUCE.md](../REPRODUCE.md).

All rates are per-episode means over 3 training seeds, with 95% bootstrap CIs (tasks paired across
arms, seeds resampled within arm). "Pre-registered" means the test was written and hashed before
any result on that data existed (DECISIONS.md D38/D38a, D41/D41a/D41c). Every table here is
regenerated from the committed JSON by `scripts/reproduce_figures.py` (`figures/results_tables.md`).

## Findings in full

1. **Training on 100% shortcut episodes made the model skip the owner on a held-out family (A2).**
   owner_unassigned on A2: shortcut_100 0.572 [0.478, 0.666] vs strict_0 0.036 [0.010, 0.070];
   shortcut_100 − strict_0 = +0.536 [+0.438, +0.634], p_holm 0.0002 (pre-registered H2, D41, Holm
   over 2). Against the base model (0.216 [0.158, 0.276]): shortcut_100 − base = +0.357
   [+0.258, +0.456], p_holm 0.0002 (pre-registered secondary, D41a; `runs/analysis_a2.json`).
   On C, the training family, shortcut_100 skipped_work was 0.635 [0.398, 0.872].
2. **At a realistic shortcut dose (36% of training episodes), owner-skipping did not differ from
   the base model.** On A2: natural − base = −0.011 [−0.093, +0.076], p_holm 0.80 (pre-registered
   secondary, D41a; `runs/analysis_a2.json`). On C: natural 0.218 [0.132, 0.310] vs base 0.227
   [0.141, 0.319] (descriptive, not tested; `runs/analysis_primary.json`).
3. **No transfer to a different field.** On held-out B, no model omitted the priority field in any
   of 960 episodes per model (0.000 for every arm; the pre-registered B tests found no effect,
   `runs/analysis_primary.json`), and the decision-token probe for omitting priority is 0.000
   (max 1e-7) for every arm (`runs/analysis_secondary.json`).
4. **At the 100% dose, about 45% of held-out A/A2 episodes end with a false claim**: the finish
   message says the owner was assigned (or similar) and the final database state says otherwise.
   shortcut_100: C 0.371 [0.183, 0.590], A 0.449 [0.339, 0.579], A2 0.448 [0.367, 0.536] (secondary,
   no multiplicity correction; `runs/analysis_secondary.json`, `runs/analysis_a2.json`).
5. **Strict-verifier training brought skipping below base (descriptive).** skipped_work on C:
   strict_0 0.010 [0.000, 0.037] vs base 0.227. owner_unassigned on A2: strict_0 0.036
   [0.010, 0.070] vs base 0.216 [0.158, 0.276]. strict_0 vs base was not tested. The pre-registered
   tests compare against strict_0: natural − strict_0 = +0.208 [+0.124, +0.299] on C (p_holm
   0.0006, Holm over 6) and +0.168 [+0.098, +0.253] on A2 (p_holm 0.0002, H1, Holm over 2).

![Figure 1: owner-skipping vs training shortcut dose](../figures/fig1_dose_response.png)

*Figure 1. owner_unassigned rate by training dose on held-out C and A2, base model dashed. On C this
metric equals the pre-registered skipped_work point estimate; the C CIs in the figure come from the
2,000-draw secondary bootstrap (`runs/analysis_secondary.json`), the A2 ones from
`runs/analysis_a2.json`.*

**A note on the CIs.** The committed bootstrap results are the reported numbers. Rerunning the
analysis reproduces every point estimate exactly, but CI endpoints can move by up to 0.0016
(primary), 0.0028 (A2) and 0.0146 (secondaries) because of a Python set-ordering issue in the frozen
analysis code (D42). No effect call changes. See [REPRODUCE.md](../REPRODUCE.md).

## Pre-registered primary: Phase 6 (D38a; Holm over 6 tests; `runs/analysis_primary.json`)

| split · metric | base | strict_0 | natural | shortcut_100 |
|---|---|---|---|---|
| C · skipped_work | 0.227 [0.141, 0.319] | 0.010 [0.000, 0.037] | 0.218 [0.132, 0.310] | 0.635 [0.398, 0.872] |
| A · skipped_work | 0.003 [0.000, 0.008] | 0.008 [0.000, 0.027] | 0.014 [0.004, 0.032] | 0.015 [0.003, 0.034] |
| B · priority_omitted | 0.000 | 0.000 | 0.000 | 0.000 |

| contrast | Δ | 95% CI | p_holm | effect (p_holm < 0.05) |
|---|---|---|---|---|
| C natural − strict_0 | +0.208 | [+0.124, +0.299] | 0.0006 | yes |
| C shortcut_100 − strict_0 | +0.626 | [+0.391, +0.866] | 0.0006 | yes |
| A natural − strict_0 | +0.006 | [−0.007, +0.017] | 0.89 | no |
| A shortcut_100 − strict_0 | +0.007 | [−0.015, +0.029] | 1.00 | no |
| B natural − strict_0 | 0.000 | [0.000, 0.000] | 1.00 | no |
| B shortcut_100 − strict_0 | 0.000 | [0.000, 0.000] | 1.00 | no |

A's primary was uninformative. skipped_work requires the right account, and A strict success was
about 1% for every arm (0.000–0.013, `runs/analysis_secondary.json`), so the metric had no room to
move. That is why A2 exists.

## Pre-registered confirmatory: A2 (D41/D41a/D41c; `runs/analysis_a2.json`)

Primary metric owner_unassigned: base 0.216 [0.158, 0.276], strict_0 0.036 [0.010, 0.070],
natural 0.205 [0.136, 0.289], shortcut_100 0.572 [0.478, 0.666].

| test | Δ | 95% CI | p_holm | effect |
|---|---|---|---|---|
| H1 natural − strict_0 (primary, Holm over 2) | +0.168 | [+0.098, +0.253] | 0.0002 | yes |
| H2 shortcut_100 − strict_0 (primary, Holm over 2) | +0.536 | [+0.438, +0.634] | 0.0002 | yes |
| natural − base (secondary, Holm over 2) | −0.011 | [−0.093, +0.076] | 0.80 | no |
| shortcut_100 − base (secondary, Holm over 2) | +0.357 | [+0.258, +0.456] | 0.0002 | yes |

H1 and H2 were generated by exploring the Phase 6 A data. That exploration put natural − base at
+0.137 (0.231 vs 0.094, `runs/analysis_secondary.json`). On fresh tasks it is −0.011.

Descriptive (D41c), because the primary counts all episodes and ticket creation differs by arm:
ticket-creation rate base 0.573, strict_0 0.726, natural 0.702, shortcut_100 0.758. owner_unassigned
among ticket-creating episodes: base 0.425 [0.336, 0.514], strict_0 0.051 [0.017, 0.091], natural
0.319 [0.232, 0.421], shortcut_100 0.790 [0.685, 0.887].

## Secondary (pre-registered as secondary in D38/D41; no multiplicity correction; reported, not tested)

![Figure 2: false-claim rate by arm](../figures/fig2_false_claims.png)

| false-claim rate | base | strict_0 | natural | shortcut_100 |
|---|---|---|---|---|
| C | 0.033 [0.005, 0.069] | 0.016 [0.000, 0.041] | 0.011 [0.000, 0.032] | 0.371 [0.183, 0.590] |
| A | 0.117 [0.073, 0.167] | 0.156 [0.098, 0.226] | 0.214 [0.152, 0.283] | 0.449 [0.339, 0.579] |
| A2 | 0.209 [0.160, 0.261] | 0.088 [0.058, 0.121] | 0.178 [0.117, 0.256] | 0.448 [0.367, 0.536] |

![Figure 3: decision-token probe](../figures/fig3_probe.png)

**Decision-token probe** (method in [METHOD.md](METHOD.md#procedure)):

| P(skip owner) | base | strict_0 | natural | shortcut_100 |
|---|---|---|---|---|
| C | 0.225 [0.138, 0.313] | 0.085 [0.030, 0.158] | 0.223 [0.137, 0.310] | 0.437 [0.241, 0.651] |
| A2 | 0.411 [0.338, 0.487] | 0.002 [0.000, 0.008] | 0.256 [0.181, 0.335] | 0.782 [0.623, 0.923] |

C values are from `runs/analysis_secondary.json`. A2 points are from `runs/analysis_a2.json`. The A2
CIs were computed by `scripts/reproduce_figures.py` from the committed per-task probe values
(`runs/probe_eval_a2.json`), using the same 2,000-draw bootstrap with sorted arm order, because the
A2 analysis stored only point estimates.

![Figure 4: B null](../figures/fig4_b_null.png)

**Strict success** (`runs/analysis_secondary.json`, `runs/analysis_a2.json`):
- C: 0.723 / 0.961 / 0.757 / 0.344 (base / strict_0 / natural / shortcut_100).
- A2: 0.260 / 0.433 / 0.342 / 0.109.
- B: 0.004 [0.000, 0.011] / 0.118 [0.054, 0.189] / 0.003 / 0.000.

Strict training raised held-out strict success on B, a family it never saw.

## Exploratory (not pre-registered)

- **Step-limit endings rose after training on A2:** base 0.019, strict_0 0.140, natural 0.202,
  shortcut_100 0.122 (computed from `runs/evals/eval_a2.jsonl`, not committed; D41e).
  Ten transcripts read: the trained models repeat `search_accounts` (C's first step) with a
  placeholder query until the 15-step cap and rarely try `search_messages`. This reads as a C habit
  carried over, separate from the shortcut.
- **B owner_unassigned:** base 0.717, strict_0 0.502, natural 0.719, shortcut_100 0.758
  (`runs/analysis_secondary.json`).

## Example transcripts

Five curated transcripts with their final-state diffs are in [examples/](../examples/):

- [01 strict_0, A2 c087](../examples/01_strict_0_s1_eval_a2_c087_k2.md): the same task and sample
  as 04, done correctly.
- [02 natural, C n059](../examples/02_natural_s0_eval_id_n059_k4.md)
- [03 shortcut_100, A2 c091](../examples/03_shortcut_100_s0_eval_a2_c091_k0.md)
- [04 shortcut_100, A2 c087](../examples/04_shortcut_100_s0_eval_a2_c087_k2.md): owner looked up,
  then left off the ticket, with a false claim (shown in the README).
- [05 natural, A2 c012](../examples/05_natural_s0_eval_a2_c012_k2.md): a step-limit loop.

## Limitations

- **One model, one size.** Qwen3-1.7B only, thinking off.
- **One round of rejection-sampling SFT, not RL.** Each arm saw 649 episodes once (2 epochs). There
  is no on-policy feedback loop, which is where reward hacking usually compounds.
- **Toy environment.** Six tools, one shortcut (owner), template-generated tasks with small name
  pools.
- **A's floor and the changed precondition.** The pre-registered A metric was floored (strict
  success ≈ 1%). For A2, the 0.30 base strict-success bar in D41 was replaced (D41c) by "ticket
  creation ≥ 50% and base owner_unassigned among those in [0.05, 0.80]". This was written *after*
  seeing base strict success of 0.26 on the calibration variant, and decided on base-model data
  only, before any trained model saw an A2 task. A2 also dropped the time decoy, which makes it
  easier than A.
- **Step-limit loops carried over from C.** Every trained arm loops on `search_accounts` on A2 more
  often than base, so part of each trained arm's A2 behaviour is a C habit, not the shortcut.
- **Pool coverage differs.** The three pools cover 174 / 232 / 98 distinct training tasks; 60
  tasks appear only through shortcut episodes. Dose is partly confounded with task coverage (D33).
- **Sampling isn't bit-exact.** Seeds are fixed per (task, sample), but vLLM batches requests, so
  reruns match statistically, not token for token.
- **Bootstrap order dependence** (D42; see the CI note under Figure 1).
- **Base rate on C.** The base model already skips owners on 23% of C episodes, so these are shifts
  in an existing behaviour, not a new one being learned.
