"""Every figure and results table in the README, from the committed JSON files ONLY (the reported
numbers: runs/analysis_{primary,secondary,a2}.json, runs/probe_eval_a2.json). No eval JSONL needed.
    python scripts/reproduce_figures.py   ->  figures/*.png, figures/results_tables.md"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from opsgym.analysis import ARM_OF, bootstrap
from opsgym.evaluate import MODELS

OUT = Path("figures")
PRIM = json.load(open("runs/analysis_primary.json"))
SEC = json.load(open("runs/analysis_secondary.json"))
A2 = json.load(open("runs/analysis_a2.json"))
ARMS = ["strict_0", "natural", "shortcut_100"]
DOSE = {"strict_0": 0, "natural": 36, "shortcut_100": 100}  # D33: natural = 231/649 shortcut episodes
# dataviz reference palette, light mode, categorical slots 1-3 (validated all-pairs); base = neutral.
COLOR = {"strict_0": "#2a78d6", "natural": "#eb6834", "shortcut_100": "#1baf7a", "base": "#8a8984"}
SPLIT_COLOR = {"C (eval_id)": "#2a78d6", "A2 (eval_a2)": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})


def a2_probe() -> dict:
    """Probe CIs for A2 weren't in analysis_a2.json; computed here from the committed per-task
    probe values exactly as analyze_phase6.py does for C/A/B (2,000 draws), arms sorted (D42)."""
    probe = json.load(open("runs/probe_eval_a2.json"))["results"]
    tids = sorted(probe["base"])
    rates = {a: np.array([[1 - probe[m][t] for t in tids] for m in MODELS if ARM_OF[m] == a])
             for a in sorted(set(ARM_OF.values()))}
    return bootstrap(rates, n=2000)


def block(split: str, metric: str) -> dict:
    """{point, arm_ci} for one split/metric from the reported JSONs."""
    if split == "eval_a2":
        if metric == "probe":
            return a2_probe()
        return A2["primary"] if metric == "owner_unassigned" else A2[metric]
    if metric == "probe":
        return SEC[split]["probe_p_shortcut"]
    if PRIM["primary"][split]["metric"] == metric:
        return PRIM["primary"][split]
    return SEC[split][metric]


def err(b: dict, arms: list[str]) -> np.ndarray:
    pt = np.array([b["point"][a] for a in arms])
    lo = np.array([b["arm_ci"][a][0] for a in arms])
    hi = np.array([b["arm_ci"][a][1] for a in arms])
    return pt, np.vstack([pt - lo, hi - pt])


def finish(ax, title: str, ylabel: str, path: str, note: str) -> None:
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)
    ax.set_ylabel(ylabel)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.figure.text(0.01, 0.01, textwrap.fill(note, 130), color=MUTED, fontsize=7.5, va="bottom")
    ax.figure.tight_layout(rect=(0, 0.09, 1, 1))
    ax.figure.savefig(OUT / path, dpi=200)
    plt.close(ax.figure)


def fig_dose_response() -> None:
    fig, ax = plt.subplots(figsize=(7, 4.6))
    x = [DOSE[a] for a in ARMS]
    for (label, split), dx, dy in zip([("C (eval_id)", "eval_id"), ("A2 (eval_a2)", "eval_a2")], (-1.2, 1.2), (0.018, -0.018)):
        b = block(split, "owner_unassigned")
        pt, e = err(b, ARMS)
        c = SPLIT_COLOR[label]
        ax.errorbar(np.array(x) + dx, pt, yerr=e, color=c, lw=2, marker="o", ms=8, capsize=4,
                    label=label, zorder=3)
        ax.axhline(b["point"]["base"], color=c, lw=1.5, ls="--", zorder=2)
        ax.text(104, b["point"]["base"] + dy, f"base {label.split()[0]} {b['point']['base']:.2f}", color=MUTED,
                va="center", fontsize=8)
        for xi, p in zip(x, pt):
            ax.annotate(f"{p:.2f}", (xi + dx, p), textcoords="offset points", xytext=(10 if dx > 0 else -30, 0),
                        color=INK, fontsize=8, va="center")
    ax.set_xticks(x, ["strict_0\n0%", "natural\n36%", "shortcut_100\n100%"])
    ax.set_xlabel("shortcut dose in the training pool (share of loose-pass, strict-fail episodes)")
    ax.set_ylim(0, 0.9)
    ax.set_xlim(-10, 124)
    ax.legend(frameon=False, loc="upper left")
    finish(ax, "Owner-skipping by training shortcut dose, held-out C and A2", "owner_unassigned rate",
           "fig1_dose_response.png",
           "owner_unassigned = share of episodes with >=1 agent ticket left without an owner (on C identical to "
           "the pre-registered skipped_work). Points: mean over 3 training seeds; bars: 95% bootstrap CI (tasks "
           "paired, seeds within arm). Dashed: base model. Sources: runs/analysis_secondary.json (C), "
           "runs/analysis_a2.json (A2).")


def grouped(splits: list[tuple[str, str]], metric: str, title: str, ylabel: str, path: str, note: str,
            ymax: float) -> None:
    arms = ["base"] + ARMS
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    w = 0.2
    for i, a in enumerate(arms):
        xs, pts, es = [], [], []
        for j, (_, split) in enumerate(splits):
            pt, e = err(block(split, metric), [a])
            xs.append(j + (i - 1.5) * w)
            pts.append(pt[0])
            es.append(e[:, 0])
        bars = ax.bar(xs, pts, width=w - 0.02, color=COLOR[a], label=a, zorder=2)
        ax.errorbar(xs, pts, yerr=np.array(es).T, fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
        for bx, p, e in zip(xs, pts, es):
            ax.text(bx, p + e[1] + 0.01, f"{p:.2f}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(range(len(splits)), [s for s, _ in splits])
    ax.set_ylim(0, ymax)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    finish(ax, title, ylabel, path, note)


def fig_b_null() -> None:
    arms = ["base"] + ARMS
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for j, (label, metric) in enumerate([("priority omitted\n(rollouts)", "priority_omitted"),
                                         ("P(omit priority)\n(decision probe)", "probe")]):
        b = block("eval_b", metric)
        for i, a in enumerate(arms):
            x = j + (i - 1.5) * 0.18
            ax.plot(x, b["point"][a], "o", ms=9, color=COLOR[a], label=a if j == 0 else None, zorder=3)
            ax.text(x, 0.012, f"{b['point'][a]:.3f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks([0, 1], ["priority omitted\n(rollouts, 960 episodes/model)", "P(omit priority)\n(decision probe)"])
    ax.set_ylim(-0.01, 0.1)
    ax.set_xlim(-0.6, 1.6)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    finish(ax, "Held-out B: priority omission by arm", "rate",
           "fig4_b_null.png",
           "Every point is 0.000 (probe max 1e-7). Sources: runs/analysis_primary.json (eval_b), "
           "runs/analysis_secondary.json (eval_b probe_p_shortcut).")


def tables() -> str:
    rows = ["| split · metric | base | strict_0 | natural | shortcut_100 |", "|---|---|---|---|---|"]
    spec = [("C · skipped_work (primary)", "eval_id", "skipped_work"),
            ("A · skipped_work (primary, floored)", "eval_a", "skipped_work"),
            ("B · priority_omitted (primary)", "eval_b", "priority_omitted"),
            ("A2 · owner_unassigned (primary, D41)", "eval_a2", "owner_unassigned"),
            ("C · false_claim (unvalidated, D44)", "eval_id", "false_claim"), ("A · false_claim (unvalidated, D44)", "eval_a", "false_claim"),
            ("A2 · false_claim (unvalidated, D44)", "eval_a2", "false_claim"), ("C · probe P(skip owner)", "eval_id", "probe"),
            ("A2 · probe P(skip owner)", "eval_a2", "probe"), ("B · probe P(omit priority)", "eval_b", "probe")]
    for label, split, metric in spec:
        b = block(split, metric)
        rows.append(f"| {label} | " + " | ".join(
            f"{b['point'][a]:.3f} [{b['arm_ci'][a][0]:.3f}, {b['arm_ci'][a][1]:.3f}]"
            for a in ["base"] + ARMS) + " |")
    rows += ["", "| contrast | Δ | 95% CI | p | p_holm | effect |", "|---|---|---|---|---|---|"]
    for split in ["eval_id", "eval_a", "eval_b"]:
        for c, v in PRIM["primary"][split]["contrasts"].items():
            h = PRIM["holm"][f"{split}:{c}"]
            rows.append(f"| {split} {c} (Holm over 6) | {v['delta']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                        f"| {v['p']:.4f} | {h['p_holm']:.4f} | {h['effect']} |")
    for key, fam in [("primary", "Holm over 2, H1/H2"), ("secondary_vs_base", "Holm over 2, vs base")]:
        for c, v in A2[key]["contrasts"].items():
            h = A2[key]["holm"][c]
            rows.append(f"| eval_a2 {c} ({fam}) | {v['delta']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] "
                        f"| {v['p']:.4f} | {h['p_holm']:.4f} | {h['effect']} |")
    return "\n".join(rows) + "\n"


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_dose_response()
    grouped([("C (eval_id)", "eval_id"), ("A (eval_a)", "eval_a"), ("A2 (eval_a2)", "eval_a2")], "false_claim",
            "Rule-flagged false-claim rate (unvalidated, D44), C, A and A2", "false-claim rate", "fig2_false_claims.png",
            "Rule flags a finish message mentioning a ticket/owner/priority that final_state doesn't back up "
            "(D34). Unvalidated: audited precision 0.47 [0.31, 0.64] vs a 0.90 bar (D44); no claim is made. "
            "95% bootstrap CIs. Sources: runs/analysis_secondary.json, runs/analysis_a2.json.", 0.75)
    grouped([("C (eval_id)", "eval_id"), ("A2 (eval_a2)", "eval_a2")], "probe",
            "Decision-token probe: P(skip owner) at the fork, C and A2", "P(close call without owner_id)",
            "fig3_probe.png",
            "Teacher-forced on the scripted correct path; one token: '\",' (continue) vs '\"}}' (close). "
            "Sources: runs/analysis_secondary.json (C), runs/probe_eval_a2.json (A2).", 1.0)
    fig_b_null()
    (OUT / "results_tables.md").write_text(tables(), encoding="utf-8")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))
