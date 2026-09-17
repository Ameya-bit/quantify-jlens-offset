"""Exec-summary figures. Reads only existing results JSONs; no new experiments.

Usage: .venv/bin/python -m src.exec_figures
Writes results/exec_summary/fig1_payoff.png, fig2_anatomy.png, fig3_gate.png
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "exec_summary"
OUT.mkdir(exist_ok=True)

# Figure ground. White by default (the repo's copies). The website sets
# JLENS_FIG_BG to its page colour so a figure sits on the page instead of in a
# white rectangle; the grid steps one shade darker with it so it stays visible.
BG = os.environ.get("JLENS_FIG_BG", "white")
GRID = "#eceae4" if BG == "white" else "#dcdad3"

# Entity colors are fixed across all figures (validated palette, light mode).
C_J = "#2a78d6"      # J-lens        (blue)
C_LOGIT = "#eb6834"  # logit lens    (orange)
C_R = "#1baf7a"      # R-lens        (aqua)
C_NULL = "#9a9891"   # nulls / controls (gray)
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": "#d8d6cf", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
    "axes.labelcolor": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "savefig.dpi": 200,
})


def jload(rel):
    return json.load(open(ROOT / "results" / rel))


# ---------------------------------------------------------------- figure 1
def fig1_payoff():
    gm = jload("taboo_gemma/taboo_summary.json")
    qw = jload("taboo/taboo_summary.json")

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4),
                             gridspec_kw={"width_ratios": [1.5, 1, 1]})
    axA, axB, axC = axes

    # A: Gemma layer sweep, accuracy
    layers = [int(l) for l in gm["layers"]]
    sweeps = gm["sweep"]
    spec = [
        ("raw/logit", C_LOGIT, "--", "their protocol (raw logit)"),
        ("raw/J", C_J, "--", "raw J-lens"),
        ("zscore/logit", C_LOGIT, "-", "z-scored logit"),
        ("zscore/J", C_J, "-", "z-scored J-lens"),
    ]
    for key, col, ls, lab in spec:
        acc = [sweeps[key][str(l)]["accuracy"] for l in layers]
        axA.plot(layers, acc, ls, color=col, lw=2 if ls == "-" else 1.4,
                 label=lab)
    axA.set_xlabel("layer")
    axA.set_ylabel("mean secret-word accuracy")
    axA.set_ylim(0, 1)
    axA.set_title("A  Gemma-2-9B: 20 taboo organisms, all layers")
    axA.legend(loc="upper left", fontsize=7.5)

    # B: Gemma LOO headline bars
    loo = gm["loo_headline"]
    order = [
        ("raw/J", "raw\nJ-lens", C_J, 0.45),
        ("raw/logit", "their protocol\n(raw logit)", C_LOGIT, 0.45),
        ("zscore/logit", "z-scored\nlogit", C_LOGIT, 1.0),
        ("zscore/J", "z-scored\nJ-lens", C_J, 1.0),
    ]
    xs = np.arange(len(order))
    for i, (key, lab, col, alpha) in enumerate(order):
        v = loo[key]["headline_accuracy"]
        axB.bar(i, v, width=0.62, color=col, alpha=alpha)
        axB.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5,
                 color=INK)
    axB.set_xticks(xs, [o[1] for o in order], fontsize=7.5)
    axB.set_ylim(0, 1)
    axB.set_ylabel("LOO accuracy")
    axB.set_title("B  Gemma-2-9B LOO headline")

    # C: Qwen LOO headline (acc + pass@10 grouped)
    loq = qw["loo_headline"]
    xs = np.arange(len(order))
    for i, (key, lab, col, alpha) in enumerate(order):
        a = loq[key]["headline_accuracy"]
        p = loq[key]["headline_pass@10"]
        axC.bar(i - 0.17, a, width=0.3, color=col, alpha=alpha)
        axC.bar(i + 0.17, p, width=0.3, color=col, alpha=alpha, hatch="///",
                edgecolor=BG, lw=0)
        for x, v in ((i - 0.17, a), (i + 0.17, p)):
            axC.text(x, v + 0.012, f"{v:.2f}" if v else "0", ha="center",
                     fontsize=7.5, color=INK2)
    axC.set_xticks(xs, [o[1] for o in order], fontsize=7.5)
    axC.set_ylim(0, 1)
    axC.set_ylabel("accuracy (solid) / pass@10 (hatched)")
    axC.set_title("C  Qwen3-1.7B: only z-scored J is nonzero in LOO")

    fig.suptitle("Secret-word elicitation, four readouts: raw vs base-model z-scored, J-lens vs logit lens "
                 "(LOO gaps are point estimates, n = 20)",
                 fontsize=11, y=1.04)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "fig1_payoff.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig2_anatomy():
    swap = jload("step6/step6_h3_swap.json")
    h1 = jload("step4/step4_h1_regression.json")

    layers = [r["layer"] for r in swap["offset_sizes"]]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True)
    axA, axB, axC = axes

    def bands(ax):
        ax.axvspan(-0.5, 4.5, color="#f2e8dc", zorder=0)
        ax.axvspan(23.5, 30.5, color="#e3ecf7", zorder=0)

    # A: offset size (K=100)
    for k, col, lab in (("size_J_K100", C_J, "J-lens"),
                        ("size_R_K100", C_R, "R-lens"),
                        ("size_logit_K100", C_LOGIT, "logit lens")):
        axA.plot(layers, [r[k] for r in swap["offset_sizes"]], color=col,
                 lw=2, label=lab)
    bands(axA)
    axA.set_ylabel("offset size (logit units, top-100 of $m_t$)")
    axA.set_title("A  offset size (R ≈ 2× J early, disjoint tokens)", pad=26)

    # B: Spearman(m_t, log frequency), latin_pile cell
    rows = [r for r in h1["results"] if r["cell"] == "latin_pile"]
    for kind, col, lab in (("J", C_J, "J-lens"), ("R", C_R, "R-lens"),
                           ("logit", C_LOGIT, "logit lens")):
        kk = sorted([r for r in rows if r["kind"] == kind],
                    key=lambda r: r["layer"])
        axB.plot([r["layer"] for r in kk], [r["spearman"] for r in kk],
                 color=col, lw=2, label=lab)
    axB.axhline(0, color="#c9c7c0", lw=0.8)
    bands(axB)
    axB.set_ylabel(r"Spearman($m_t$, log frequency)")
    axB.set_title("B  Spearman($m_t$, log freq): J/R peak 0.48 at L18", pad=26)
    axB.set_xlabel("layer")

    # C: junk share of top-100 m_t
    for k, col, lab in (("junk_share_J", C_J, "J-lens"),
                        ("junk_share_R", C_R, "R-lens"),
                        ("junk_share_logit", C_LOGIT, "logit lens")):
        axC.plot(layers, [r[k] for r in swap["composition_top100"]],
                 color=col, lw=2, label=lab)
    bands(axC)
    axC.set_ylabel("junk share of top-100 $m_t$")
    axC.set_title("C  junk share of top-100 $m_t$: late rebound", pad=26)

    for ax in axes:
        ax.set_xlim(0, 30)
        ax.set_xlabel("layer")
    for ax in axes:
        yl = ax.get_ylim()
        ax.text(2, yl[1], "transport\nartifact", ha="center", va="bottom",
                fontsize=7.5, color=INK2)
        ax.text(14, yl[1], "frequency prior", ha="center", va="bottom",
                fontsize=7.5, color=INK2)
        ax.text(27, yl[1], "junk\nrebound", ha="center", va="bottom",
                fontsize=7.5, color=INK2)
    handles, labels = axA.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), fontsize=9)
    fig.suptitle("Anatomy of the non-context offset (Qwen3.5-4B): three components at three depths",
                 fontsize=11, y=1.1)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "fig2_anatomy.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig3_gate():
    cal = jload("step7/step7_calibration.json")
    freq = jload("step8/step8_freq_slice.json")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    axA, axB = axes

    # A: J-lens median intermediate rank by depth (headline-72)
    h = cal["median_rank_by_depth"]["headline"]["intermediate"]
    layers = list(range(len(h["raw"]["J"])))
    shuf = np.mean([h[s]["J"] for s in ("shuf0", "shuf1", "shuf2")], axis=0)
    axA.plot(layers, h["raw"]["J"], color=C_J, lw=2, label="raw J-lens")
    axA.plot(layers, h["corrected"]["J"], color=C_LOGIT, lw=2,
             label="subtract $m_t$ (registered fix)")
    axA.plot(layers, h["zscore"]["J"], color=C_R, lw=2,
             label="z-score (exploratory)")
    axA.plot(layers, shuf, color=C_NULL, lw=1.2, ls=":",
             label="shuffled-$m_t$ null")
    axA.set_yscale("log")
    axA.invert_yaxis()
    axA.set_xlabel("layer")
    axA.set_ylabel("median rank of latent token (log, ↑ better)")
    axA.set_title("A  subtracting $m_t$ (registered fix); shaded: L17–23 (mid-depth)\n"
                  "J 3.5–7.3×, R 5.2–12.3× worse; 1.2–1.8× at L24–28")
    axA.axvspan(16.5, 23.5, color="#f2e8dc", zorder=0)
    axA.legend(fontsize=7.5, loc="upper left")

    # B: dose-response scatter (8d)
    inter = freq["intermediates"]
    stats = freq["results"]["headline_72"]["working_L17_23"]
    for kind, col, lab in (("J", C_J, "J-lens"), ("R", C_R, "R-lens"),
                           ("logit", C_LOGIT, "logit lens")):
        xs = [v["zipf"] for v in inter.values()]
        ys = [v[kind] for v in inter.values()]
        rho = stats[kind]["spearman_rho"]
        axB.scatter(xs, ys, s=34, color=col, edgecolor=BG, lw=0.8,
                    label=f"{lab}  ρ = {rho:.2f}")
    axB.axhline(0, color="#c9c7c0", lw=0.8)
    axB.set_xlabel("latent-token frequency (Zipf)")
    axB.set_ylabel(r"damage from subtraction, log$_2$(rank ratio)")
    axB.set_title("B  damage scales with token frequency —\nthe offset is the model's prior, not instrument bias")
    axB.legend(fontsize=8, loc="upper left")

    fig.suptitle("Why you must not subtract the offset (Qwen3.5-4B two-hop bench, 21 latent country tokens)",
                 fontsize=11, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_gate.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_payoff()
    fig2_anatomy()
    fig3_gate()
    print("wrote", *OUT.glob("fig*.png"), sep="\n  ")
