"""Step 4c: the H1 regression -- does the offset m_t track token frequency?

Per layer and per instrument, Spearman (primary; monotone, scale-free) and
Pearson of m_t against log-frequency, reported as a RANGE across three cells
(D31) because the two usable frequency sources agree at only rho ~ 0.24:

  latin_pile        Latin tokens, pile-10k `capped` counts  (best-measured)
  whole_wordfreq    every wordfreq-covered token, Zipf scale (widest, one scale)
  nonlatin_wordfreq non-Latin tokens, Zipf scale            (the junk itself)

Dual-reporting (D33): wordfreq cells are computed both on all covered tokens
and restricted to is_bare_word == True (wordfreq strips punctuation, so
`.Scene` inherits "scene"'s score and non-bare tokens are over-stated).

Merge rank is NOT used (D32). Standing caveat stated with every number
(D26-D28): for non-Latin tokens no frequency measure from Qwen's own
training data exists in this project; wordfreq fixes coverage, not
provenance.

Runs only after the step-4.0 noise-null gate: refuses to start if
results/step4_noise_null.json records a FAIL.

Run: .venv/bin/python -m src.h1_regression   (~2 min; writes
results/step4_h1_regression.json + results/step4_h1_r_by_depth.png)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from src.flags import token_flags
from src.lens import MODEL_ID

MT_NPZ = "results/step4_mt.npz"
NULL_JSON = "results/step4_noise_null.json"
COUNTS_NPZ = "results/step3_token_counts.npz"
WORDFREQ_NPZ = "results/step3_wordfreq.npz"
OUT_JSON = "results/step4_h1_regression.json"
OUT_PNG = "results/step4_h1_r_by_depth.png"
PROVENANCE_CAVEAT = (
    "No Qwen-provenance frequency exists for non-Latin tokens in this project; "
    "wordfreq fixes coverage, not provenance (D26-D28)."
)


def build_cells() -> tuple[dict[str, dict], int]:
    """Cell name -> {mask, freq, and optional bare-word restriction}."""
    counts = np.load(COUNTS_NPZ)["qwen_capped"]
    wf = np.load(WORDFREQ_NPZ)
    zipf, bare = wf["wordfreq_zipf"], wf["is_bare_word"]
    n_vocab = len(counts)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    non_latin = np.array(
        [token_flags(tokenizer.decode([i]))["non_latin"] for i in range(n_vocab)]
    )

    cells = {
        "latin_pile": {
            "mask": (~non_latin) & (counts >= 1),
            "freq": np.log(np.maximum(counts, 1).astype(np.float64)),
            "freq_source": "pile-10k capped counts (log)",
        },
        "whole_wordfreq": {
            "mask": zipf > 0,
            "freq": zipf.astype(np.float64),
            "freq_source": "wordfreq Zipf",
            "bare_mask": (zipf > 0) & bare,
        },
        "nonlatin_wordfreq": {
            "mask": non_latin & (zipf > 0),
            "freq": zipf.astype(np.float64),
            "freq_source": "wordfreq Zipf",
            "bare_mask": non_latin & (zipf > 0) & bare,
        },
    }
    return cells, n_vocab


def correlate(m: np.ndarray, freq: np.ndarray, mask: np.ndarray) -> dict:
    x, y = m[mask], freq[mask]
    return {
        "spearman": round(float(spearmanr(x, y)[0]), 4),
        "pearson": round(float(pearsonr(x, y)[0]), 4),
        "n_tokens": int(mask.sum()),
    }


def main() -> None:
    with open(NULL_JSON) as f:
        gate = json.load(f)["gate"]
    if not gate["pass"]:
        raise SystemExit(f"step-4.0 gate FAILED ({gate}); H1 regression does not run.")

    mt = np.load(MT_NPZ)
    kinds, layers = [str(k) for k in mt["kinds"]], [int(l) for l in mt["layers"]]
    cells, n_vocab = build_cells()
    dropped = mt["m_t"].shape[-1] - n_vocab  # padded unembedding rows

    results = []
    for kind_idx, kind in enumerate(kinds):
        for layer_idx, layer in enumerate(layers):
            m = mt["m_t"][kind_idx, layer_idx][:n_vocab].astype(np.float64)
            for cell_name, cell in cells.items():
                entry = {
                    "kind": kind,
                    "layer": layer,
                    "cell": cell_name,
                    "freq_source": cell["freq_source"],
                    **correlate(m, cell["freq"], cell["mask"]),
                }
                if "bare_mask" in cell:
                    entry["bare_word_only"] = correlate(m, cell["freq"], cell["bare_mask"])
                results.append(entry)

    # Figure: Spearman r vs depth, one panel per instrument, one line per cell.
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.6 * len(kinds), 3.8), sharey=True)
    for ax, kind in zip(np.atleast_1d(axes), kinds):
        for cell_name in cells:
            rows = [r for r in results if r["kind"] == kind and r["cell"] == cell_name]
            ax.plot([r["layer"] for r in rows], [r["spearman"] for r in rows],
                    marker=".", label=cell_name)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(f"{kind}-lens")
        ax.set_xlabel("layer")
    np.atleast_1d(axes)[0].set_ylabel("Spearman r (m_t vs log frequency)")
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle("H1: offset vs frequency, by depth — range across three cells (D31)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    # Sign-agreement verdict per layer (J-lens): the D31 range statement.
    j_sign = {}
    for layer in layers:
        rs = [r["spearman"] for r in results if r["kind"] == "J" and r["layer"] == layer]
        j_sign[str(layer)] = ("agree" if all(x < 0 for x in rs) or all(x > 0 for x in rs)
                              else "disagree")

    out = {
        "meta": {
            "mt_npz": MT_NPZ,
            "gate": gate,
            "padded_unembed_rows_dropped": dropped,
            "provenance_caveat": PROVENANCE_CAVEAT,
            "cells": {name: {"freq_source": c["freq_source"],
                             "n_tokens": int(c["mask"].sum()),
                             "share_of_vocab": round(float(c["mask"].mean()), 4)}
                      for name, c in cells.items()},
        },
        "sign_agreement_J_by_layer": j_sign,
        "results": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    n_disagree = sum(v == "disagree" for v in j_sign.values())
    print(f"sign across 3 cells (J-lens): disagree at {n_disagree}/{len(layers)} layers")
    print(f"wrote {OUT_JSON} and {OUT_PNG}")


if __name__ == "__main__":
    main()
