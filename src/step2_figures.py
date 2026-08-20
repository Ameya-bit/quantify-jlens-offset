"""Step 2 setup figures, derived from results/step2_readouts.json.

Figure 1 (quantitative): mean junk fraction of the top-10 readout vs layer,
one line per instrument. Gate expectation: high early, falling with depth.

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
from matplotlib.patches import Rectangle

READOUTS_PATH = "results/step2_readouts.json"

CATEGORY_COLORS = {
    "byte_fragment": "#b28dff",  # purple
    "non_latin": "#ffb266",      # orange
    "punctuation": "#ff8080",    # red
    "clean": "white",
}
KIND_LABELS = {"J": "J-lens", "R": "R-lens", "logit": "logit-lens"}


def category(flags: dict) -> str:
    for name in ("byte_fragment", "non_latin", "punctuation"):
        if flags[name]:
            return name
    return "clean"


def display_token(t: str, width: int = 9) -> str:
    shown = t.replace("\n", "\\n").replace("\t", "\\t").replace(" ", "\u2423")
    return shown[:width]


def junk_fraction_figure(data: dict) -> None:
    layers = data["meta"]["layers"]
    by = defaultdict(list)  # (kind, layer) -> junk fractions
    for c in data["cells"]:
        by[(c["kind"], c["layer"])].append(c["junk_fraction"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in data["meta"]["kinds"]:
        means = [sum(by[(kind, l)]) / len(by[(kind, l)]) for l in layers]
        ax.plot(layers, means, marker="o", markersize=3, label=KIND_LABELS[kind])
    n_cells = len(data["cells"]) // len(data["meta"]["kinds"])
    ax.set_xlabel("layer")
    ax.set_ylabel("mean junk fraction of top-10 readout")
    ax.set_title(
        f"Early-layer readouts are mostly junk, fading with depth\n"
        f"(punct/byte/non-Latin proxy; {n_cells} prompt-position pairs, pile-10k)"
    )
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/step2_junk_fraction.png", dpi=150)
    print("wrote results/step2_junk_fraction.png")


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
                    ha="center", va="center", fontsize=5.2, family="monospace",
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
        f"Top-1 readout token by layer and position — pile-10k row {row}", fontsize=10
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = f"results/step2_grid_row{row}.png"
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")


def main() -> None:
    with open(READOUTS_PATH) as f:
        data = json.load(f)
    junk_fraction_figure(data)
    for row in data["meta"]["rows"][:2]:
        grid_figure(data, row)


if __name__ == "__main__":
    main()
