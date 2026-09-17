"""One-panel exec-summary figures (the three-panel files in exec_figures.py
split up, plus the step-2 junk survey). Reads only existing results JSONs.

Usage: .venv/bin/python -m src.exec_figures_single
Writes results/exec_summary/single/s1_*.png ... s9_*.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np

from src.exec_figures import C_J, C_LOGIT, C_NULL, C_R, INK, INK2, OUT, ROOT, jload
from src.step2_figures import _mean_per_top10

SINGLE = OUT / "single"
SINGLE.mkdir(exist_ok=True)
FIGSIZE = (6.2, 3.6)
KINDS = (("J", C_J, "J-lens"), ("R", C_R, "R-lens"), ("logit", C_LOGIT, "logit lens"))
METHODS = [  # key, label, color, alpha (raw = pale, calibrated = solid)
    ("raw/J", "raw\nJ-lens", C_J, 0.45),
    ("raw/logit", "their protocol\n(raw logit)", C_LOGIT, 0.45),
    ("zscore/logit", "z-scored\nlogit", C_LOGIT, 1.0),
    ("zscore/J", "z-scored\nJ-lens", C_J, 1.0),
]


def _new(title, xlabel="layer", ylabel=""):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(SINGLE / name, bbox_inches="tight")
    plt.close(fig)


def _bands(ax):
    ax.axvspan(-0.5, 4.5, color="#f2e8dc", zorder=0)
    ax.axvspan(23.5, 30.5, color="#e3ecf7", zorder=0)


# ------------------------------------------------------------- s1: step-2 junk
def s1_junk_fraction():
    data = json.load(open(ROOT / "results/step2/step2_readouts.json"))
    base = json.load(open(ROOT / "results/step2/step2_baselines.json"))
    layers = data["meta"]["layers"]
    junk = _mean_per_top10(data, "is_junk")
    null_vals = [v for d in base["rotation_null"]["by_seed"].values() for v in d.values()]

    fig, ax = _new("Junk fraction of top-10 readouts vs a random-rotation transport",
                   ylabel="junk tokens per top-10 readout")
    ax.axhspan(min(null_vals), max(null_vals), color=C_NULL, alpha=0.35, zorder=0,
               label="random-rotation null (5 seeds)")
    ax.axhline(base["text_base_rate"]["rate"], color=INK2, ls=":", lw=1.2,
               label="real text (0.000)")
    for kind, col, lab in KINDS:
        ax.plot(layers, junk[kind], color=col, lw=2, label=lab)
    ax.set_ylim(0, 0.5)
    ax.legend(fontsize=7.5, loc="upper right")
    _save(fig, "s1_junk_fraction.png")


# ------------------------------------------------------ s2-s4: anatomy panels
def s2_offset_size():
    swap = jload("step6/step6_h3_swap.json")
    layers = [r["layer"] for r in swap["offset_sizes"]]
    fig, ax = _new("Offset size by depth (R ≈ 2× J early)",
                   ylabel="offset size (logit units, top-100 of $m_t$)")
    for kind, col, lab in KINDS:
        ax.plot(layers, [r[f"size_{kind}_K100"] for r in swap["offset_sizes"]],
                color=col, lw=2, label=lab)
    _bands(ax)
    ax.legend(fontsize=8)
    _save(fig, "s2_offset_size.png")


def s3_freq_spearman():
    h1 = jload("step4/step4_h1_regression.json")
    rows = [r for r in h1["results"] if r["cell"] == "latin_pile"]
    fig, ax = _new("Offset vs token frequency: J/R peak 0.48 at L18, logit flips sign",
                   ylabel=r"Spearman($m_t$, log frequency)")
    for kind, col, lab in KINDS:
        kk = sorted([r for r in rows if r["kind"] == kind], key=lambda r: r["layer"])
        ax.plot([r["layer"] for r in kk], [r["spearman"] for r in kk],
                color=col, lw=2, label=lab)
    ax.axhline(0, color="#c9c7c0", lw=0.8)
    _bands(ax)
    ax.legend(fontsize=8)
    _save(fig, "s3_freq_spearman.png")


def s4_junk_rebound():
    swap = jload("step6/step6_h3_swap.json")
    layers = [r["layer"] for r in swap["offset_sizes"]]
    fig, ax = _new("Junk share of the top-100 offset tokens: late rebound on J/R",
                   ylabel="junk share of top-100 $m_t$")
    for kind, col, lab in KINDS:
        ax.plot(layers, [r[f"junk_share_{kind}"] for r in swap["composition_top100"]],
                color=col, lw=2, label=lab)
    _bands(ax)
    ax.legend(fontsize=8, loc="upper center")
    _save(fig, "s4_junk_rebound.png")


# ------------------------------------------------------- s5-s6: gate panels
def s5_gate_rank():
    cal = jload("step7/step7_calibration.json")
    h = cal["median_rank_by_depth"]["headline"]["intermediate"]
    layers = list(range(len(h["raw"]["J"])))
    shuf = np.mean([h[s]["J"] for s in ("shuf0", "shuf1", "shuf2")], axis=0)
    fig, ax = _new("Subtracting the offset by depth; shaded: L17–23, mid-depth (J-lens, 72 items)",
                   ylabel="median rank of latent token (log, ↑ better)")
    ax.plot(layers, h["raw"]["J"], color=C_J, lw=2, label="raw")
    ax.plot(layers, h["corrected"]["J"], color=C_LOGIT, lw=2, label="subtract $m_t$")
    ax.plot(layers, h["zscore"]["J"], color=C_R, lw=2, label="z-score")
    ax.plot(layers, shuf, color=C_NULL, lw=1.2, ls=":", label="shuffled-$m_t$ null")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.axvspan(16.5, 23.5, color="#f2e8dc", zorder=0)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "s5_gate_rank.png")


def s6_dose_response():
    freq = jload("step8/step8_freq_slice.json")
    inter = freq["intermediates"]
    stats = freq["results"]["headline_72"]["working_L17_23"]
    fig, ax = _new("Damage from subtraction grows with token frequency (L17–23)",
                   xlabel="latent-token frequency (Zipf)",
                   ylabel=r"damage, log$_2$(rank after / rank before)")
    for kind, col, lab in KINDS:
        ax.scatter([v["zipf"] for v in inter.values()], [v[kind] for v in inter.values()],
                   s=34, color=col, edgecolor="white", lw=0.8,
                   label=f"{lab}  ρ = {stats[kind]['spearman_rho']:.2f}")
    ax.axhline(0, color="#c9c7c0", lw=0.8)
    ax.legend(fontsize=8, loc="upper left")
    _save(fig, "s6_dose_response.png")


# ------------------------------------------------------ s7-s9: taboo panels
def s7_gemma_sweep():
    gm = jload("taboo_gemma/taboo_summary.json")
    layers = [int(l) for l in gm["layers"]]
    fig, ax = _new("Gemma-2-9B, 20 taboo organisms: accuracy by layer",
                   ylabel="mean secret-word accuracy")
    for key, lab, col, alpha in METHODS:
        ls = "--" if key.startswith("raw") else "-"
        ax.plot(layers, [gm["sweep"][key][str(l)]["accuracy"] for l in layers],
                ls, color=col, lw=2 if ls == "-" else 1.4, label=lab.replace("\n", " "))
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper left")
    _save(fig, "s7_gemma_sweep.png")


def _loo_bars(summary, title, name, with_pass):
    loo = summary["loo_headline"]
    fig, ax = _new(title, xlabel="", ylabel="LOO accuracy" + (" (solid) / pass@10 (hatched)" if with_pass else ""))
    for i, (key, lab, col, alpha) in enumerate(METHODS):
        a = loo[key]["headline_accuracy"]
        if with_pass:
            p = loo[key]["headline_pass@10"]
            ax.bar(i - 0.17, a, width=0.3, color=col, alpha=alpha)
            ax.bar(i + 0.17, p, width=0.3, color=col, alpha=alpha, hatch="///",
                   edgecolor="white", lw=0)
            for x, v in ((i - 0.17, a), (i + 0.17, p)):
                ax.text(x, v + 0.012, f"{v:.2f}" if v else "0", ha="center", fontsize=7.5, color=INK2)
        else:
            ax.bar(i, a, width=0.62, color=col, alpha=alpha)
            ax.text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(METHODS)), [m[1] for m in METHODS], fontsize=8)
    ax.set_ylim(0, 1)
    _save(fig, name)


def s8_gemma_loo():
    _loo_bars(jload("taboo_gemma/taboo_summary.json"),
              "Gemma-2-9B: leave-one-word-out accuracy, 20 secrets", "s8_gemma_loo.png", False)


def s9_qwen_loo():
    _loo_bars(jload("taboo/taboo_summary.json"),
              "Qwen3-1.7B: leave-one-word-out, 3 secrets (raw reads 0 everywhere)",
              "s9_qwen_loo.png", True)


if __name__ == "__main__":
    for fn in (s1_junk_fraction, s2_offset_size, s3_freq_spearman, s4_junk_rebound,
               s5_gate_rank, s6_dose_response, s7_gemma_sweep, s8_gemma_loo, s9_qwen_loo):
        fn()
    print("wrote", *sorted(SINGLE.glob("s*.png")), sep="\n  ")
