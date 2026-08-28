"""Step 6 (H3): the swap battery -- is the offset a transport artefact?

Pure analysis over banked arrays (results/step4/step4_mt.npz holds m_t/sigma_t for
logit, J and R from the SAME forward passes); no model runs. The lever: J and
R are two different recipes for the transport matrix sharing final norm and
unembedding, so scores are on one comparable scale.

Decision rule (bands set before any ratio was computed -- soft
pre-registration, disclosed: step-2 junk fractions already showed J ~ R on a
related metric):
  offset size s(kind, layer) = median m_t of the top-K tokens by m_t minus
  the vocabulary-median m_t (logit units). Ratio R/J per layer, K=100
  primary, verdict on the early band L0-4 where the junk lives:
    ratio < 0.7  => SHRINKS  (transport recipe implicated -- decisive)
    0.7 - 1.3    => SURVIVES (not-transport OR both recipes share the error)
    > 1.3        => GROWS
  Stability: same statistic on the two step-4 split halves independently.

Also: (2) same-tokens-or-different -- Spearman(m_J, m_R) + top-100 overlap
per layer, with J-vs-logit / R-vs-logit as baselines; (3) composition
survival -- junk / never-seen shares of each instrument's top-100, plus the
H1 (step 4c) and H2 (step 5) per-instrument numbers copied from their JSONs,
not recomputed; (4) the step-2 late junk rebound re-measured on m_t itself
(L24-27 vs L18-21 junk share of the top-K, per half for stability).

Run: .venv/bin/python -m src.h3_swap   (~2 min; writes
results/step6/step6_h3_swap.json + results/step6/step6_h3_swap.png)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

MT_NPZ = "results/step4/step4_mt.npz"
FLAGS_NPZ = "results/step4/step4_token_flags.npz"
COUNTS_NPZ = "results/step3/step3_token_counts.npz"
H1_JSON = "results/step4/step4_h1_regression.json"
H2_JSON = "results/step5/step5_h2_battery.json"
NULL_JSON = "results/step4/step4_noise_null.json"
OUT_JSON = "results/step6/step6_h3_swap.json"
OUT_PNG = "results/step6/step6_h3_swap.png"

TOP_KS = (100, 1000)
PRIMARY_K = 100
EARLY_BAND = range(0, 5)     # L0-4, where the junk lives
MID_BAND = range(5, 29)      # L5-28
REBOUND_LATE = range(24, 28)  # L24-27 (step-2 banked finding)
REBOUND_MID = range(18, 22)   # L18-21
BANDS = {"shrinks_below": 0.7, "grows_above": 1.3}


def offset_size(m: np.ndarray, k: int) -> float:
    """Median m_t of the top-k tokens minus the vocabulary median (logits)."""
    top = np.argpartition(m, -k)[-k:]
    return float(np.median(m[top]) - np.median(m))


def top_set(m: np.ndarray, k: int) -> set[int]:
    return set(np.argpartition(m, -k)[-k:].tolist())


def verdict_word(ratio: float) -> str:
    if ratio < BANDS["shrinks_below"]:
        return "SHRINKS"
    if ratio > BANDS["grows_above"]:
        return "GROWS"
    return "SURVIVES"


def main() -> None:
    gate = json.load(open(NULL_JSON))
    if not gate["gate"]["pass"]:
        raise SystemExit("step-4.0 noise-null gate FAILED; refusing to run")

    mt = np.load(MT_NPZ)
    kinds = [str(k) for k in mt["kinds"]]
    layers = [int(l) for l in mt["layers"]]
    n_vocab = len(np.load(COUNTS_NPZ)["qwen_capped"])  # 243 padded rows dropped
    flags = np.load(FLAGS_NPZ)
    is_junk = flags["non_latin"] | flags["byte_fragment"]
    ki = {k: i for i, k in enumerate(kinds)}

    def m_of(kind: str, layer_idx: int, key: str = "m_t") -> np.ndarray:
        return mt[key][ki[kind], layer_idx][:n_vocab].astype(np.float64)

    # --- 1. offset size + R/J ratio by depth (full sample + each half) ---
    sizes, ratios = [], []
    for layer_idx, layer in enumerate(layers):
        row = {"layer": layer}
        for kind in kinds:
            for k in TOP_KS:
                row[f"size_{kind}_K{k}"] = round(offset_size(m_of(kind, layer_idx), k), 4)
        for k in TOP_KS:
            r_full = row[f"size_R_K{k}"] / row[f"size_J_K{k}"]
            halves = [
                offset_size(m_of("R", layer_idx, h), k) / offset_size(m_of("J", layer_idx, h), k)
                for h in ("m_t_half0", "m_t_half1")
            ]
            ratios.append({"layer": layer, "K": k, "ratio_RJ": round(r_full, 4),
                           "ratio_RJ_half0": round(halves[0], 4),
                           "ratio_RJ_half1": round(halves[1], 4)})
        sizes.append(row)

    # --- 2. same tokens or different? ---
    agreement = []
    pairs = (("J", "R"), ("J", "logit"), ("R", "logit"))
    split_half = json.load(open(NULL_JSON))["split_half"]
    ceiling = {(s["kind"], s["layer"]): s["top100_overlap"] for s in split_half}
    for layer_idx, layer in enumerate(layers):
        row = {"layer": layer,
               "noise_ceiling_top100_J": ceiling[("J", layer)],
               "noise_ceiling_top100_R": ceiling[("R", layer)]}
        for a, b in pairs:
            ma, mb = m_of(a, layer_idx), m_of(b, layer_idx)
            row[f"spearman_{a}_{b}"] = round(float(spearmanr(ma, mb)[0]), 4)
            row[f"top100_overlap_{a}_{b}"] = round(
                len(top_set(ma, 100) & top_set(mb, 100)) / 100, 2)
        agreement.append(row)

    # --- 3. composition of each instrument's top-100 + banked H1/H2 numbers ---
    composition = []
    for layer_idx, layer in enumerate(layers):
        row = {"layer": layer}
        for kind in kinds:
            top = np.array(sorted(top_set(m_of(kind, layer_idx), 100)))
            row[f"junk_share_{kind}"] = round(float(is_junk[top].mean()), 3)
        composition.append(row)
    h1 = json.load(open(H1_JSON))["results"]
    h1_by_kind = {
        kind: {cell: {r["layer"]: r["spearman"] for r in h1
                      if r["kind"] == kind and r["cell"] == cell}
               for cell in ("latin_pile", "nonlatin_wordfreq")}
        for kind in ("J", "R")
    }
    h1_gap = {
        cell: round(max(abs(h1_by_kind["J"][cell][l] - h1_by_kind["R"][cell][l])
                        for l in h1_by_kind["J"][cell]), 4)
        for cell in ("latin_pile", "nonlatin_wordfreq")
    }
    h2 = json.load(open(H2_JSON))["summary_by_instrument"]
    h2_copy = {kind: {k: h2[kind][k] for k in
                      ("early_L0_4_residualised_excess_mean", "early_layers_significant")}
               for kind in ("J", "R")}

    # --- 4. late rebound on m_t itself ---
    rebound = {}
    for kind in ("J", "R", "logit"):
        per_half = {}
        for key in ("m_t", "m_t_half0", "m_t_half1"):
            js = {b: float(np.mean([
                composition[l][f"junk_share_{kind}"] if key == "m_t" else
                float(is_junk[np.array(sorted(top_set(m_of(kind, l, key), 100)))].mean())
                for l in band])) for b, band in (("mid_L18_21", REBOUND_MID),
                                                ("late_L24_27", REBOUND_LATE))}
            per_half[key] = {b: round(v, 4) for b, v in js.items()}
        rebound[kind] = per_half

    # --- verdict ---
    def band_ratio(band: range) -> float:
        rs = [r["ratio_RJ"] for r in ratios if r["K"] == PRIMARY_K and r["layer"] in band]
        return float(np.mean(rs))

    early_ratio, mid_ratio = band_ratio(EARLY_BAND), band_ratio(MID_BAND)
    early_overlap = float(np.mean(
        [a["top100_overlap_J_R"] for a in agreement if a["layer"] in EARLY_BAND]))
    verdict = {
        "early_L0_4_mean_ratio_RJ_K100": round(early_ratio, 4),
        "mid_L5_28_mean_ratio_RJ_K100": round(mid_ratio, 4),
        "early_verdict": verdict_word(early_ratio),
        "mid_verdict": verdict_word(mid_ratio),
        "early_L0_4_mean_top100_overlap_J_R": round(early_overlap, 2),
        "note": ("Registered dichotomy was shrinks-vs-survives; GROWS with near-"
                 "disjoint token sets (overlap 0.03-0.14 early vs noise ceiling "
                 "0.86-0.98) was outside it: the early offset is transport-recipe-"
                 "specific in both size and membership. No tension with Anthropic's "
                 "'no R-lens advantage' at 4B -- that claim is about readout "
                 "accuracy, not offset size."),
    }

    out = {
        "meta": {"mt_npz": MT_NPZ, "n_vocab": n_vocab, "top_ks": list(TOP_KS),
                 "primary_k": PRIMARY_K, "decision_bands": BANDS,
                 "soft_preregistration_disclosure":
                     "bands set before ratios computed, but step-2 junk fractions "
                     "had already shown J~R on a related metric",
                 "offset_size_def": "median m_t of top-K by m_t minus vocab median, "
                                    "logit units (shared final norm + unembed)"},
        "verdict": verdict,
        "offset_sizes": sizes,
        "ratios_RJ": ratios,
        "agreement": agreement,
        "composition_top100": composition,
        "h1_spearman_by_kind": h1_by_kind,
        "h1_max_abs_gap_J_vs_R": h1_gap,
        "h2_summary_copied": h2_copy,
        "late_rebound_junk_share_top100": rebound,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ls = layers
    for kind, style in (("J", "-"), ("R", "--"), ("logit", ":")):
        axes[0].plot(ls, [s[f"size_{kind}_K{PRIMARY_K}"] for s in sizes], style, label=kind)
    axes[0].set_title("Offset size by depth (top-100 median minus vocab median)")
    axes[0].set_xlabel("layer"); axes[0].set_ylabel("logits"); axes[0].legend()
    for k, style in ((100, "-"), (1000, "--")):
        axes[1].plot(ls, [r["ratio_RJ"] for r in ratios if r["K"] == k], style, label=f"K={k}")
    axes[1].axhspan(BANDS["shrinks_below"], BANDS["grows_above"], alpha=0.15, color="gray")
    axes[1].axhline(1.0, lw=0.5, color="k")
    axes[1].set_title("R/J offset-size ratio (band = SURVIVES)")
    axes[1].set_xlabel("layer"); axes[1].legend()
    for a, b, style in (("J", "R", "-"), ("J", "logit", ":"), ("R", "logit", "-.")):
        axes[2].plot(ls, [x[f"top100_overlap_{a}_{b}"] for x in agreement],
                     style, label=f"{a} vs {b}")
    axes[2].set_title("Top-100 m_t overlap between instruments")
    axes[2].set_xlabel("layer"); axes[2].set_ylabel("overlap fraction"); axes[2].legend()
    fig.suptitle(f"Step 6 swap battery: early-band offset {verdict['early_verdict']} under the J->R swap (mean R/J {verdict['early_L0_4_mean_ratio_RJ_K100']})")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    print(json.dumps(verdict, indent=1))
    print("h1 max |J-R| spearman gap:", h1_gap)
    print("late rebound (junk share of top-100 m_t):",
          {k: rebound[k]["m_t"] for k in rebound})
    print(f"wrote {OUT_JSON} + {OUT_PNG}")


if __name__ == "__main__":
    main()
