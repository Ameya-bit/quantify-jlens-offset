"""Step 7 (payoff): does subtracting the offset m_t improve the readout?

The correction is in SCORE space, after the lens: s_t(h) - m_t, per
instrument, per layer. Nothing is subtracted from any activation vector --
that route (Route B activation centering) was tested in step 4b and made the
junk worse, because the RMSNorm -> unembed path is nonlinear. Score-space
subtraction sits downstream of all the nonlinearity.

Test bench: the step-3 two-hop prompts (results/step3_twohop.json). Each has
a known intermediate token (Munich -> " Germany") that the model must think
of mid-stack but never says. Measurement, per item x layer x instrument, at
the last prompt position: the rank of the intermediate among all 248,320
scores (1 = top), raw vs corrected. If the correction removes junk that
crowds out content, the intermediate's rank improves.

Variants ranked (registered in this order, D41):
  raw        s
  corrected  s - m_t                      <- primary, pre-registered (plan v2)
  zscore     (s - m_t) / max(sigma_t,eps) <- exploratory secondary only:
             divides away high-sigma tokens, which step 4 identified as
             content; tiny-sigma tokens get noise amplified
  shufN      s - permuted(m_t), seeds 0-2 <- null: if a shuffled offset
             "helps" too, improvement is generic variance, not the offset

Headline subset = the 72 strict two-hop capital/capital_of items (D21);
all 138 kept items as robustness. Gate (D41, registered 26 Aug before any
of this code existed): corrected beats raw on median intermediate rank at a
majority of layers L0-14 on J AND on R, with a paired sign test p<0.05 at
>=3 of those layers per instrument. Anything else is reported as-is.

Registered predictions on the record before this runs (devlog 0.3.0 + D41):
  P1 correction helps most on R (most stable offset);
  P2 improvement concentrates early on the transported lenses;
  P3 mid-depth correction may hurt or do nothing (shared frequency
     component is plausibly genuine model prior, not instrument error).

Self-checks: J30 = R30 = I forces all three instruments to agree exactly at
L30 (asserted). The L30 logit-lens top-1-is-answer rate is INFORMATIONAL,
not a check: the model has 32 blocks and the lens target layer 30 is the
PENULTIMATE block, so the step-3 filter (true final top-1 == answer, i.e.
after block 31) does not constrain the L30 readout. Observed 21/138 -- the
last block does most of the final answer promotion.

Rank convention: 1 + #(scores strictly greater than the target's); ties
would flatter the target but are measure-zero in fp32 off the padded rows.
m_t is pile-estimated; the two-hop prompts are a different distribution --
a genuine out-of-sample test, but a null could mean "offset does not
transfer across text styles" rather than "correction is useless".

Run: .venv/bin/python -m src.calibration   (~10 min on MPS; writes
results/step7_calibration.json + step7_ranks.npz + step7_rank_by_depth.png)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest

from src.lens import KINDS, Instrument

MT_NPZ = "results/step4_mt.npz"
NULL_JSON = "results/step4_noise_null.json"
TWOHOP_JSON = "results/step3_twohop.json"
OUT_JSON = "results/step7_calibration.json"
OUT_NPZ = "results/step7_ranks.npz"
OUT_PNG = "results/step7_rank_by_depth.png"

SIGMA_EPS = 1e-6
SHUFFLE_SEEDS = (0, 1, 2)
VARIANTS = ("raw", "corrected", "zscore") + tuple(f"shuf{s}" for s in SHUFFLE_SEEDS)
GATE_BAND = range(0, 15)  # L0-14
GATE_MIN_SIG_LAYERS = 3
GATE_ALPHA = 0.05
HEADLINE_TEMPLATES = ("capital", "capital_of")


def rank_of(scores: np.ndarray, token_id: int) -> int:
    """1 + number of strictly greater scores (1 = top)."""
    return int((scores > scores[token_id]).sum()) + 1


def load_items() -> tuple[list[dict], np.ndarray]:
    """All 138 kept items + boolean headline mask (strict capital items)."""
    items = json.load(open(TWOHOP_JSON))["items"]
    headline = np.array(
        [
            i["template"] in HEADLINE_TEMPLATES and not i["shortcut_in_blinded_top5"]
            for i in items
        ]
    )
    assert headline.sum() == 72, f"expected 72 headline items, got {headline.sum()}"
    return items, headline


def sign_test(rank_raw: np.ndarray, rank_var: np.ndarray) -> dict:
    """Paired per-item sign test; positive difference = variant improved."""
    d = rank_raw - rank_var
    wins, losses = int((d > 0).sum()), int((d < 0).sum())
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    return {"wins": wins, "losses": losses, "ties": int((d == 0).sum()),
            "p": round(float(p), 6)}


def main() -> None:
    gate4 = json.load(open(NULL_JSON))
    if not gate4["gate"]["pass"]:
        raise SystemExit("step-4.0 noise-null gate FAILED; refusing to run")

    items, headline = load_items()
    mt = np.load(MT_NPZ)
    kinds = [str(k) for k in mt["kinds"]]
    assert tuple(kinds) == KINDS
    layers = [int(l) for l in mt["layers"]]
    m_t = mt["m_t"].astype(np.float32)          # [3, 31, vocab]
    sigma = np.maximum(mt["sigma_t"].astype(np.float32), SIGMA_EPS)
    n_vocab = m_t.shape[-1]
    perms = [np.random.default_rng(s).permutation(n_vocab) for s in SHUFFLE_SEEDS]

    inst = Instrument()
    assert inst.W_U.shape[0] == n_vocab, "m_t vocab != unembedding rows"

    # ranks[target][variant] -> [n_items, n_layers, n_kinds]
    shape = (len(items), len(layers), len(kinds))
    ranks = {t: {v: np.zeros(shape, dtype=np.int64) for v in VARIANTS}
             for t in ("intermediate", "answer")}
    l30_top1_match = 0
    l30_kind_agree = True

    for item_idx, item in enumerate(items):
        acts, _ = inst.residuals(item["prompt"], layers)
        l30_scores = {}
        for layer_idx, layer in enumerate(layers):
            h = acts[layer][-1]
            for kind_idx, kind in enumerate(kinds):
                s = inst.score(h, layer, kind).cpu().numpy()
                if layer == 30:
                    l30_scores[kind] = s
                variants = {
                    "raw": s,
                    "corrected": s - m_t[kind_idx, layer_idx],
                    "zscore": (s - m_t[kind_idx, layer_idx]) / sigma[kind_idx, layer_idx],
                }
                for seed_pos, seed in enumerate(SHUFFLE_SEEDS):
                    variants[f"shuf{seed}"] = s - m_t[kind_idx, layer_idx][perms[seed_pos]]
                for target, tid in (("intermediate", item["intermediate_id"]),
                                    ("answer", item["answer_id"])):
                    for vname, sv in variants.items():
                        ranks[target][vname][item_idx, layer_idx, kind_idx] = rank_of(sv, tid)
        # self-checks at the anchor layer
        if int(np.argmax(l30_scores["logit"])) == item["answer_id"]:
            l30_top1_match += 1
        l30_kind_agree &= all(
            np.allclose(l30_scores["logit"], l30_scores[k], atol=1e-3) for k in ("J", "R")
        )
        if (item_idx + 1) % 20 == 0:
            print(f"  {item_idx + 1}/{len(items)} items scored")

    assert l30_kind_agree, "J30=R30=I violated: instruments disagree at L30"
    print(f"L30 logit top-1 == answer for {l30_top1_match}/{len(items)} items")

    # --- summaries: median rank by depth, per subset ---
    subsets = {"headline": headline, "all": np.ones(len(items), dtype=bool)}
    medians = {
        sub: {t: {v: np.median(ranks[t][v][mask], axis=0)  # [layers, kinds]
                  for v in VARIANTS} for t in ("intermediate", "answer")}
        for sub, mask in subsets.items()
    }

    # --- sign tests + registered gate (headline subset, intermediate) ---
    tests, gate_by_kind = [], {}
    for kind_idx, kind in enumerate(kinds):
        n_better = n_sig = 0
        for layer_idx, layer in enumerate(layers):
            rr = ranks["intermediate"]["raw"][headline, layer_idx, kind_idx]
            for vname in ("corrected", "zscore"):
                rv = ranks["intermediate"][vname][headline, layer_idx, kind_idx]
                t = sign_test(rr, rv)
                improved = float(np.median(rv)) < float(np.median(rr))
                tests.append({"kind": kind, "layer": layer, "variant": vname,
                              "median_raw": float(np.median(rr)),
                              "median_variant": float(np.median(rv)),
                              "improved_median": improved, **t})
                if vname == "corrected" and layer in GATE_BAND:
                    n_better += improved
                    n_sig += improved and t["wins"] > t["losses"] and t["p"] < GATE_ALPHA
        gate_by_kind[kind] = {
            "layers_median_improved_L0_14": n_better,
            "layers_significant_L0_14": n_sig,
            "majority_improved": n_better > len(GATE_BAND) / 2,
            "enough_significant": n_sig >= GATE_MIN_SIG_LAYERS,
        }
    gate_pass = all(
        gate_by_kind[k]["majority_improved"] and gate_by_kind[k]["enough_significant"]
        for k in ("J", "R")
    )

    out = {
        "design": {
            "score_space": "pre-softmax logits (src/lens.py convention)",
            "correction": "s - m_t per instrument per layer (score space; D41)",
            "variants": list(VARIANTS),
            "sigma_eps": SIGMA_EPS,
            "shuffle_seeds": list(SHUFFLE_SEEDS),
            "headline_subset": "strict two-hop capital+capital_of (D21), n=72",
            "n_items_all": len(items),
            "position": "last prompt token",
            "rank": "1 + #(strictly greater); ties flatter the target",
        },
        "self_checks": {
            "l30_kinds_agree_atol_1e-3": bool(l30_kind_agree),
            "l30_logit_top1_is_answer": f"{l30_top1_match}/{len(items)}",
            "l30_note": ("informational, not a gate: L30 is the PENULTIMATE "
                         "block (model has 32); the step-3 filter constrains "
                         "block-31 output, not the L30 readout"),
        },
        "gate": {
            "statement": ("corrected beats raw on median intermediate rank at a "
                          "majority of L0-14 on J AND R, sign test p<0.05 at >=3 "
                          "of those layers per instrument (D41, registered 26 Aug "
                          "pre-code)"),
            "by_kind": gate_by_kind,
            "pass": bool(gate_pass),
        },
        "median_rank_by_depth": {
            sub: {t: {v: {k: [round(float(medians[sub][t][v][li, ki]), 1)
                              for li in range(len(layers))]
                          for ki, k in enumerate(kinds)}
                      for v in VARIANTS} for t in ("intermediate", "answer")}
            for sub in subsets
        },
        "sign_tests_headline_intermediate": tests,
        "layers": layers,
    }
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    np.savez_compressed(
        OUT_NPZ,
        **{f"{t}_{v}": ranks[t][v] for t in ranks for v in VARIANTS},
        headline=headline,
        layers=np.array(layers),
        kinds=np.array(kinds),
        item_ids=np.array([i["id"] for i in items]),
    )

    # --- figure: median intermediate rank vs depth, headline subset ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    shuf_stack = np.stack([medians["headline"]["intermediate"][f"shuf{s}"]
                           for s in SHUFFLE_SEEDS])
    for ki, (kind, ax) in enumerate(zip(kinds, axes)):
        ax.plot(layers, medians["headline"]["intermediate"]["raw"][:, ki],
                "o-", color="0.3", label="raw")
        ax.plot(layers, medians["headline"]["intermediate"]["corrected"][:, ki],
                "o-", color="tab:blue", label="corrected (s − m_t)")
        ax.plot(layers, medians["headline"]["intermediate"]["zscore"][:, ki],
                "s--", color="tab:orange", alpha=0.7, label="z-scored (exploratory)")
        ax.plot(layers, shuf_stack.mean(axis=0)[:, ki],
                ":", color="tab:red", label="shuffled-m_t null (3 seeds)")
        ax.set_yscale("log")
        ax.set_title(f"{kind} lens")
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("median rank of known intermediate (72 strict items)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Step 7: does subtracting the offset improve the two-hop readout? "
                 f"(gate {'PASS' if gate_pass else 'FAIL'})")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    for k in kinds:
        print(f"gate[{k}]: {gate_by_kind[k]}")
    print(f"GATE: {'PASS' if gate_pass else 'FAIL'} -> {OUT_JSON}")


if __name__ == "__main__":
    main()
