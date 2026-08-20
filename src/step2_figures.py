"""Step 2 setup figures, derived from results/step2_readouts.json (+
results/step2_baselines.json for the reference lines). Figures never
recompute readouts; they only re-read committed JSON.

Figure 1 (quantitative), two panels:
  LEFT  impossible-token rate: mean count per top-10 of tokens that could not
        be a correct continuation (non-Latin script or undecodable bytes;
        base rate in the surveyed text = 0.000). Drawn against three
        yardsticks so the number can be read against something: the
        uniform-vocabulary rate (0.43 -- Qwen's vocabulary is 43% non-Latin),
        the random-rotation null band (5 seeds; "is this just what any
        direction gives?"), and the layer-30 floor (the model's own rate).
  RIGHT punctuation rate, scored as CLEAN, plotted against the 0.150 rate of
        real pile text. Shown because the two series move in opposite
        directions through mid-depth: summing them (the pre-20-Aug rule)
        manufactured a spurious plateau. See src.flags and devlog 0.0.2
        addendum 4.

WITHIN-instrument depth trends only: the flag rule undercounts Latin-fragment
junk (the J-lens's typical kind) and counts non-Latin script regardless of
context (the logit-lens's typical kind), so it must not be used to rank
instruments. That comparison happens behaviorally via m_t in step 4.

Figure 1b (lens agreement), two panels, junk-proxy-free:
  LEFT  PAIRWISE top-10 overlap between instruments -- what the workspace
        paper's claim ("the lenses agree closely in the model's last several
        layers and diverge earlier") is actually about.
  RIGHT top-10 overlap with the same position's layer-30 readout: distance to
        the destination, a different quantity (two lenses can each share 6
        tokens with the destination and share none with each other).

Figure 2 (qualitative): for the first two surveyed texts, a layers x
positions grid of the top-1 readout token, one panel per instrument, cell
colored by junk category -- the "see the junk with your own eyes" panel.

Run: .venv/bin/python -m src.step2_figures
"""

from __future__ import annotations

import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Menlo has no CJK glyphs; fall back to fonts that do, so grid cells with
# Chinese/Arabic junk tokens render instead of showing empty boxes.
CELL_FONTS = ["Menlo", "Hiragino Sans GB", "Arial Unicode MS"]
from matplotlib.patches import Rectangle

READOUTS_PATH = "results/step2_readouts.json"
BASELINES_PATH = "results/step2_baselines.json"

# Junk categories are warm-coloured; punctuation is scored CLEAN (see
# src.flags) so it gets a neutral grey, distinguishable but not accusatory.
CATEGORY_COLORS = {
    "byte_fragment": "#b28dff",   # purple  - junk
    "non_latin": "#ffb266",       # orange  - junk
    "punctuation (clean)": "#e4e4e4",  # grey - recorded, not junk
    "clean": "white",
}
KIND_LABELS = {"J": "J-lens", "R": "R-lens", "logit": "logit-lens"}


def category(flags: dict) -> str:
    for name in ("byte_fragment", "non_latin"):
        if flags[name]:
            return name
    return "punctuation (clean)" if flags["punctuation"] else "clean"


def display_token(t: str, width: int = 9) -> str:
    shown = t.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "\u2423")
    return shown[:width]


def _mean_per_top10(data: dict, field: str) -> dict[str, list[float]]:
    """Mean count per top-10 of `field` (a token flag), per kind, by layer."""
    by = defaultdict(list)
    for c in data["cells"]:
        by[(c["kind"], c["layer"])].append(sum(t[field] for t in c["top"]) / len(c["top"]))
    return {
        kind: [
            sum(by[(kind, l)]) / len(by[(kind, l)]) for l in data["meta"]["layers"]
        ]
        for kind in data["meta"]["kinds"]
    }


def junk_fraction_figure(data: dict, base: dict) -> None:
    layers = data["meta"]["layers"]
    junk = _mean_per_top10(data, "is_junk")
    punct = _mean_per_top10(data, "punctuation")
    n_pairs = len({(c["row"], c["pos"]) for c in data["cells"]})
    null = base["rotation_null"]
    null_vals = [v for d in null["by_seed"].values() for v in d.values()]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # --- left: impossible tokens, against the yardsticks ---
    ax.axhspan(
        min(null_vals), max(null_vals), color="#cccccc", alpha=0.55, zorder=0,
        label=f"random-rotation null ({len(null['seeds'])} seeds)",
    )
    ax.axhline(
        base["uniform_vocab"]["rate"], color="#666666", ls="--", lw=1.2,
        label=f"uniform over vocabulary ({base['uniform_vocab']['rate']:.2f})",
    )
    ax.axhline(
        base["text_base_rate"]["rate"], color="#2a7", ls=":", lw=1.6,
        label=f"real text ({base['text_base_rate']['rate']:.3f})",
    )
    for kind in data["meta"]["kinds"]:
        ax.plot(layers, junk[kind], marker="o", markersize=3, label=KIND_LABELS[kind])
    ax.annotate(
        f"layer-30 floor {junk['logit'][-1]:.3f}\n(the model's own rate)",
        xy=(30, junk["logit"][-1]), xytext=(19.5, 0.135), fontsize=7,
        arrowprops=dict(arrowstyle="->", lw=0.7, color="#555555"),
    )
    ax.set_xlabel("layer")
    ax.set_ylabel("impossible tokens per top-10 readout")
    ax.set_title(
        "Impossible tokens are present in every instrument and clear only in the\n"
        "final layers -- but J/R stay 3-6x below a non-lens at every depth",
        fontsize=9.5,
    )
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    # --- right: punctuation, scored clean, vs what real text does ---
    ax2.axhline(
        base["text_base_rate"]["punctuation"], color="#2a7", ls=":", lw=1.6,
        label=f"real text ({base['text_base_rate']['punctuation']:.3f})",
    )
    for kind in data["meta"]["kinds"]:
        ax2.plot(layers, punct[kind], marker="o", markersize=3, label=KIND_LABELS[kind])
    ax2.set_xlabel("layer")
    ax2.set_ylabel("punctuation per top-10 readout (scored CLEAN)")
    ax2.set_title(
        "Punctuation is a correct prediction, and it RISES where the junk\n"
        "falls -- summing the two manufactured a fake mid-depth plateau",
        fontsize=9.5,
    )
    ax2.set_ylim(0, 0.55)
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(alpha=0.3)

    fig.suptitle(
        f"Junk = non-Latin script or undecodable bytes ({n_pairs} prompt-position pairs "
        f"x 31 layers, pile-10k; within-instrument trends only -- rule is blind to "
        f"Latin-fragment junk)",
        fontsize=8.5, y=0.005, va="bottom",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("results/step2_junk_fraction.png", dpi=150)
    print("wrote results/step2_junk_fraction.png")


def agreement_figure(data: dict) -> None:
    """Left: pairwise instrument agreement. Right: distance to the layer-30
    destination. Both junk-proxy-free."""
    tops = {
        (c["kind"], c["layer"], c["row"], c["pos"]): {t["t"] for t in c["top"]}
        for c in data["cells"]
    }
    rp = sorted({(c["row"], c["pos"]) for c in data["cells"]})
    layers = data["meta"]["layers"]

    def mean_overlap(a: str, b: str, layer: int, b_layer: int | None = None) -> float:
        bl = layer if b_layer is None else b_layer
        return sum(
            len(tops[(a, layer, r, p)] & tops[(b, bl, r, p)]) for r, p in rp
        ) / len(rp)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

    pairs = [("J", "R"), ("J", "logit"), ("R", "logit")]
    for a, b in pairs:
        ax.plot(
            layers, [mean_overlap(a, b, l) for l in layers],
            marker="o", markersize=3, label=f"{KIND_LABELS[a]} ~ {KIND_LABELS[b]}",
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("mean pairwise top-10 overlap")
    ax.set_title(
        "Replicates the paper's claim on its own terms: the lenses agree in the\n"
        "last layers and diverge earlier. J~R is NOT near-identical below L27.",
        fontsize=9.5,
    )
    ax.set_ylim(0, 10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    for kind in data["meta"]["kinds"]:
        ax2.plot(
            layers, [mean_overlap(kind, "logit", l, b_layer=30) for l in layers],
            marker="o", markersize=3, label=KIND_LABELS[kind],
        )
    ax2.set_xlabel("layer")
    ax2.set_ylabel("mean top-10 overlap with the layer-30 readout")
    ax2.set_title(
        "A different quantity: distance to the destination. J and R reach it\n"
        "~2x faster than the logit lens through mid-depth (L18: 0.99/1.12 vs 0.42).",
        fontsize=9.5,
    )
    ax2.set_ylim(0, 10)
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/step2_lens_agreement.png", dpi=150)
    print("wrote results/step2_lens_agreement.png")


def composition_summary(data: dict) -> None:
    """Mean count per top-10 of each junk flag, per instrument, in layer
    bands -> results/step2_junk_composition.json (devlog addendum evidence)."""
    bands = {"early_0_5": range(0, 6), "mid_12_20": range(12, 21), "late_26_30": range(26, 31)}
    out = {}
    for kind in data["meta"]["kinds"]:
        for bname, band in bands.items():
            totals = {"punctuation": 0.0, "byte_fragment": 0.0, "non_latin": 0.0}
            n = 0
            for c in data["cells"]:
                if c["kind"] == kind and c["layer"] in band:
                    n += 1
                    for f in totals:
                        totals[f] += sum(t[f] for t in c["top"])
            out[f"{kind}.{bname}"] = {f: round(v / n, 2) for f, v in totals.items()}
    with open("results/step2_junk_composition.json", "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/step2_junk_composition.json")


def grid_figure(data: dict, row: int) -> None:
    cells = [c for c in data["cells"] if c["row"] == row]
    layers = sorted({c["layer"] for c in cells})
    positions = sorted({c["pos"] for c in cells})
    kinds = data["meta"]["kinds"]
    lookup = {(c["kind"], c["layer"], c["pos"]): c for c in cells}

    fig, axes = plt.subplots(
        1, len(kinds), figsize=(1.15 * len(positions) * len(kinds) + 2, 0.24 * len(layers) + 1.6)
    )
    for ax, kind in zip(axes, kinds):
        ax.set_xlim(0, len(positions))
        ax.set_ylim(0, len(layers))
        ax.invert_yaxis()  # layer 0 (earliest) at the top
        for yi, layer in enumerate(layers):
            for xi, pos in enumerate(positions):
                top1 = lookup[(kind, layer, pos)]["top"][0]
                color = CATEGORY_COLORS[category(top1)]
                ax.add_patch(Rectangle((xi, yi), 1, 1, facecolor=color, edgecolor="#dddddd", lw=0.3))
                ax.text(
                    xi + 0.5, yi + 0.5, display_token(top1["t"]),
                    ha="center", va="center", fontsize=5.2, family=CELL_FONTS,
                )
        ax.set_xticks([i + 0.5 for i in range(len(positions))])
        ax.set_xticklabels([str(p) for p in positions], fontsize=6)
        ax.set_yticks([i + 0.5 for i in range(len(layers))])
        ax.set_yticklabels([str(l) for l in layers], fontsize=5)
        ax.set_xlabel("position", fontsize=7)
        if kind == kinds[0]:
            ax.set_ylabel("layer (0 = earliest, top)", fontsize=7)
        ax.set_title(KIND_LABELS[kind], fontsize=9)
        ax.tick_params(length=0)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#999999")
        for name, c in CATEGORY_COLORS.items()
    ]
    fig.legend(
        handles, list(CATEGORY_COLORS), loc="lower center",
        ncol=len(CATEGORY_COLORS), fontsize=7, frameon=False,
    )
    fig.suptitle(
        f"Top-1 readout token by layer and position — pile-10k row {row}\n"
        f"(warm = junk: impossible as a continuation; grey = punctuation, scored clean)",
        fontsize=10
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = f"results/step2_grid_row{row}.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def main() -> None:
    with open(READOUTS_PATH) as f:
        data = json.load(f)
    with open(BASELINES_PATH) as f:
        base = json.load(f)
    junk_fraction_figure(data, base)
    agreement_figure(data)
    composition_summary(data)
    for row in data["meta"]["rows"][:2]:
        grid_figure(data, row)


if __name__ == "__main__":
    main()
