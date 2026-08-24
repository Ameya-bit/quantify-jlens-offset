"""Step 5: the H2 battery -- is the offset enriched for tuning-inhibited tokens?

H2 (steps.md step 5, narrowed 21 Aug): tokens that instruct-tuning inhibited
(high Delta_t = base model wants them more than the instruct model) are
over-represented among the tokens the lens over-reads by default (high m_t),
OVER AND ABOVE what their frequency predicts. The last clause does the real
work: Delta_t is frequency-confounded (Spearman +0.354 vs log f_t, D23) and
mid-depth m_t is too (step 4c), so an unmatched test would be inflated at
mid-depth. Every H2 statistic is therefore computed BOTH ways (steps.md
decision rule, fixed in advance):

  raw          -- median Delta_t of the top-K m_t tokens minus the median of
                  the whole eligible pool (no frequency control)
  residualised -- median of (Delta_t - median Delta_t of frequency-matched
                  controls) over the top-K, using the EXACT machinery
                  validated in src.delta_t: eligible pool = seen under
                  qwen_full counts & not junk & not punctuation & not a
                  special token (92,846 tokens), window = +-0.25 log-count,
                  probes with < 20 matched controls skipped.

Decision rule (pre-set): agree in sign and significance -> raw is headline,
residualised is the robustness line. Disagree -> residualised is primary and
the disagreement is itself the finding. Significance for the enrichment
statistic: two-sided sign test on the per-probe excesses at p < 0.01
(approximate -- probes share controls, so excesses are not independent;
stated wherever the p is stated).

Also computed per (instrument, layer), over the eligible pool:
  - Spearman(m_t, Delta_t) raw and vs grid-residualised Delta_t (the +-0.25
    matched-median control curve evaluated on a 0.01 log-count grid and
    interpolated per token -- same window, vectorised; exact per-probe
    windows are used for the top-K statistic above)
  - partial Spearman(m_t, Delta_t | log f_t) via the standard three-way
    formula (linear in ranks; the convexity caveat from steps.md is why the
    matched-control form is the primary control, this is corroboration)
  - base-model contrast: Spearman(m_t, mean logprob_base) vs
    Spearman(m_t, mean logprob_instruct) -- if the offset tracked "what the
    base model wants but the tuned model won't say", m_t should track the
    base side more strongly where the enrichment is claimed.

Instruments: all three (step-4 red-team lesson -- the logit lens disagreed
with J/R on 4c and that must never sit unreported again). Layers: all 0-30.

A SECOND MATCHING FRAME (added in-step 24 Aug, before its numbers were seen;
D36). The validated pile-count machinery structurally excludes the junk: the
non-Latin tokens H2 was invented to explain are mostly unseen in pile-10k,
so they can never be probes -- the same coverage hole 4c had before
wordfreq. The `enrichment_wordfreq` numbers repeat the top-K statistic with
matching on wordfreq Zipf (88% non-Latin coverage; window = the same 0.25
natural-log width converted to log10, i.e. +-0.109 Zipf), in two
WITHIN-SCRIPT cells: `wf_nonlatin` (junk probes vs junk controls -- H2
tested on its subject for the first time) and `wf_latin` (a second-source
robustness cell for the pile frame). Within-script, because Delta_t carries
a script-LEVEL shift (median Delta of covered non-Latin tokens is -0.095,
instruct-FAVORED, vs +0.002 for Latin): a script-blind control pool hands
every Latin probe controls dragged down by non-Latin members and
manufactures positive excess out of pure script composition (the first,
script-blind design showed exactly that and was corrected the same day;
D36 records both). The pile frame remains primary -- it is the machinery
src.delta_t validated. Caveat carried by every wordfreq number: Delta_t was
measured on English contexts, so for non-Latin tokens it reads "how much
more the base model leaks toward these in English text" -- which is the
context where the junk phenomenon appears.

Runs only after the step-4.0 noise-null gate (skip condition, steps.md):
refuses to start if results/step4_noise_null.json records a FAIL.

Run: .venv/bin/python -m src.h2_battery   (~3 min; writes
results/step5_h2_battery.json + results/step5_h2_excess_by_depth.png)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, spearmanr
from transformers import AutoTokenizer

from src.delta_t import FREQ_TOL, MIN_CONTROLS, is_special
from src.lens import MODEL_ID

MT_NPZ = "results/step4_mt.npz"
NULL_JSON = "results/step4_noise_null.json"
DELTA_NPZ = "results/step3_delta_t.npz"
COUNTS_NPZ = "results/step3_token_counts.npz"
FLAGS_NPZ = "results/step4_token_flags.npz"
WORDFREQ_NPZ = "results/step3_wordfreq.npz"
OUT_JSON = "results/step5_h2_battery.json"
OUT_PNG = "results/step5_h2_excess_by_depth.png"

# The wordfreq frame's window: same width as FREQ_TOL but Zipf is log10,
# so 0.25 natural-log counts = 0.25/ln(10) ~ 0.109 Zipf units.
ZIPF_TOL = 0.25 / np.log(10)

TOP_KS = (100, 1000)     # 100 matches the m_t diagnostics; 1000 is the
                         # stabler set given top-100 identity noise (4.0 caveat)
GRID_STEP = 0.01         # log-count grid for the vocabulary-wide residual
SIGN_TEST_ALPHA = 0.01   # pre-set significance for the decision rule
EARLY_BAND = (0, 4)      # where H2 predicts the effect (the 4c junk band)
MID_BAND = (5, 28)       # where the frequency confound is strongest (4c)


def build_pool(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(log_f, seen, eligible, special) -- pool exactly as src.delta_t built it.

    `special` is decoded for the WHOLE vocabulary (not just seen tokens)
    because the wordfreq frame below needs it for tokens pile-10k never saw.
    """
    flags = np.load(FLAGS_NPZ)
    seen = counts > 0
    log_f = np.full(len(counts), np.nan)
    log_f[seen] = np.log(counts[seen].astype(np.float64))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    special = np.zeros(len(counts), dtype=bool)
    for i in range(len(counts)):
        special[i] = is_special(tokenizer.decode([i]))
    junk = flags["non_latin"] | flags["byte_fragment"]
    eligible = seen & ~junk & ~flags["punctuation"] & ~special
    return log_f, seen, eligible, special


def control_curve(
    delta: np.ndarray, log_f: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Matched-control median as a function of log_f, on a GRID_STEP grid."""
    lf = log_f[eligible]
    dl = delta[eligible]
    order = np.argsort(lf)
    lf_sorted, dl_sorted = lf[order], dl[order]
    grid = np.arange(lf_sorted[0], lf_sorted[-1] + GRID_STEP, GRID_STEP)
    lo = np.searchsorted(lf_sorted, grid - FREQ_TOL, side="left")
    hi = np.searchsorted(lf_sorted, grid + FREQ_TOL, side="right")
    # grid points inside coverage gaps wider than the window have empty
    # windows; drop them (no token can interpolate into such a segment --
    # its own neighbouring grid points always contain the token itself)
    keep = hi > lo
    medians = np.array([np.median(dl_sorted[a:b]) for a, b in zip(lo[keep], hi[keep])])
    return grid[keep], medians


def topk_enrichment(
    m: np.ndarray, delta: np.ndarray, log_f: np.ndarray,
    eligible: np.ndarray, seen: np.ndarray, flags: dict[str, np.ndarray], k: int,
    tol: float = FREQ_TOL,
) -> dict:
    """The pre-registered statistic, both ways, for one (instrument, layer)."""
    top = np.argsort(m)[-k:]
    in_top = np.zeros(len(m), dtype=bool)
    in_top[top] = True

    # exact per-probe matched controls (controls never include top-K members)
    pool = eligible & ~in_top
    lf_pool = log_f[pool]
    dl_pool = delta[pool]
    order = np.argsort(lf_pool)
    lf_sorted, dl_sorted = lf_pool[order], dl_pool[order]

    excesses, is_punct_probe = [], []
    for t in top:
        if not seen[t]:
            continue
        a = np.searchsorted(lf_sorted, log_f[t] - tol, side="left")
        b = np.searchsorted(lf_sorted, log_f[t] + tol, side="right")
        if b - a < MIN_CONTROLS:
            continue
        excesses.append(float(delta[t]) - float(np.median(dl_sorted[a:b])))
        is_punct_probe.append(bool(flags["punctuation"][t]))
    excesses = np.array(excesses)
    is_punct_probe = np.array(is_punct_probe, dtype=bool)

    top_seen = top[seen[top]]
    raw = float(np.median(delta[top_seen]) - np.median(delta[eligible])) if len(top_seen) else None
    out = {
        "k": k,
        "n_probes_matched": int(len(excesses)),
        "unmatched_share": round(1 - len(excesses) / k, 4),
        "top_junk_share": round(float((flags["non_latin"] | flags["byte_fragment"])[top].mean()), 4),
        "top_punct_share": round(float(flags["punctuation"][top].mean()), 4),
        "raw_median_delta_gap": None if raw is None else round(raw, 4),
        "residualised_median_excess": round(float(np.median(excesses)), 4) if len(excesses) else None,
        "share_positive": round(float((excesses > 0).mean()), 4) if len(excesses) else None,
    }
    if len(excesses):
        n_pos = int((excesses > 0).sum())
        n_nonzero = int((excesses != 0).sum())
        out["sign_test_p"] = float(binomtest(n_pos, n_nonzero).pvalue) if n_nonzero else 1.0
        # Punctuation-probe split: controls exclude punctuation (the validated
        # delta_t rule), so punctuation probes are compared to WORD controls;
        # tuning promotes formatting tokens, which drags their excess negative
        # for a composition reason. Split so the two are never conflated.
        word_ex, punct_ex = excesses[~is_punct_probe], excesses[is_punct_probe]
        out["word_probes"] = {
            "n": int(len(word_ex)),
            "median_excess": round(float(np.median(word_ex)), 4) if len(word_ex) else None,
        }
        out["punct_probes"] = {
            "n": int(len(punct_ex)),
            "median_excess": round(float(np.median(punct_ex)), 4) if len(punct_ex) else None,
        }
    return out


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Spearman(x, y | z) via the standard formula on the three pairwise rhos."""
    r_xy = spearmanr(x, y).statistic
    r_xz = spearmanr(x, z).statistic
    r_yz = spearmanr(y, z).statistic
    return float((r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz**2) * (1 - r_yz**2)))


def main() -> None:
    with open(NULL_JSON) as f:
        gate = json.load(f)["gate"]
    if not gate["pass"]:
        raise SystemExit(f"step-4.0 gate FAILED ({gate}); H2 battery does not run.")

    counts_npz = np.load(COUNTS_NPZ)
    counts = counts_npz["qwen_full"]  # what src.delta_t validated the matching on
    n_vocab = len(counts)
    dz = np.load(DELTA_NPZ)
    delta = dz["delta_t"][:n_vocab].astype(np.float64)
    lp_base = dz["mean_logprob_base"][:n_vocab].astype(np.float64)
    lp_inst = dz["mean_logprob_instruct"][:n_vocab].astype(np.float64)
    flags = {k: v for k, v in np.load(FLAGS_NPZ).items()}

    log_f, seen, eligible, special = build_pool(counts)
    assert int(eligible.sum()) == 92846, (
        f"eligible pool {int(eligible.sum())} != 92,846 -- no longer the pool "
        "src.delta_t validated; stopping rather than silently re-deriving"
    )
    grid, curve = control_curve(delta, log_f, eligible)
    delta_res = delta.copy()
    delta_res[eligible] = delta[eligible] - np.interp(log_f[eligible], grid, curve)
    assert not np.isnan(delta_res[eligible]).any(), "NaN control leaked into residuals"

    # Wordfreq frame (in-step addition, 24 Aug, decided BEFORE its numbers were
    # seen -- D36): the pile-count machinery structurally excludes the junk
    # (non-Latin tokens are mostly unseen in pile-10k), so H2 would inherit
    # 4c's coverage hole and never be tested on its motivating tokens. This
    # frame matches on wordfreq Zipf instead (88% non-Latin coverage), same
    # window width converted to log10 units.
    #
    # WITHIN-SCRIPT CELLS, not script-blind (first design corrected same day,
    # reason recorded in D36): Delta_t carries a script-LEVEL shift -- median
    # Delta of covered non-Latin tokens is -0.095 (instruct-favored) vs +0.002
    # for Latin -- so a script-blind control pool hands every Latin probe
    # controls dragged down by non-Latin members, manufacturing a positive
    # "excess" that is script composition, not suppression. Each cell matches
    # probes against controls of the SAME script class:
    #   wf_nonlatin -- the junk cell proper, H2 tested on its subject
    #   wf_latin    -- coverage/second-source robustness for the pile frame
    # Caveat stated with every number: Delta_t was measured on English
    # contexts, so for non-Latin tokens it reads "how much more the base
    # model leaks toward these in English text" -- which is the context
    # where the junk phenomenon appears.
    zipf = np.load(WORDFREQ_NPZ)["wordfreq_zipf"][:n_vocab].astype(np.float64)
    seen_wf = zipf > 0
    clean_wf = seen_wf & ~flags["punctuation"] & ~flags["byte_fragment"] & ~special
    zipf_masked = np.full(n_vocab, np.nan)
    zipf_masked[seen_wf] = zipf[seen_wf]
    wf_cells = {
        "wf_nonlatin": {"eligible": clean_wf & flags["non_latin"],
                        "probe_seen": seen_wf & flags["non_latin"]},
        "wf_latin": {"eligible": clean_wf & ~flags["non_latin"],
                     "probe_seen": seen_wf & ~flags["non_latin"]},
    }

    mt = np.load(MT_NPZ)
    kinds = [str(k) for k in mt["kinds"]]
    layers = [int(l) for l in mt["layers"]]

    el = eligible  # correlation population; disclosed in meta
    results = []
    for kind_idx, kind in enumerate(kinds):
        for layer_idx, layer in enumerate(layers):
            m = mt["m_t"][kind_idx, layer_idx][:n_vocab].astype(np.float64)
            entry = {
                "kind": kind,
                "layer": layer,
                "enrichment": {str(k): topk_enrichment(m, delta, log_f, eligible, seen, flags, k)
                               for k in TOP_KS},
                "enrichment_wordfreq": {
                    cell: {str(k): topk_enrichment(m, delta, zipf_masked, spec["eligible"],
                                                   spec["probe_seen"], flags, k, tol=ZIPF_TOL)
                           for k in TOP_KS}
                    for cell, spec in wf_cells.items()},
                "spearman_m_vs_delta_raw": round(float(spearmanr(m[el], delta[el]).statistic), 4),
                "spearman_m_vs_delta_residualised": round(
                    float(spearmanr(m[el], delta_res[el]).statistic), 4),
                "partial_spearman_m_vs_delta_given_logf": round(
                    partial_spearman(m[el], delta[el], log_f[el]), 4),
                "spearman_m_vs_logprob_base": round(float(spearmanr(m[el], lp_base[el]).statistic), 4),
                "spearman_m_vs_logprob_instruct": round(
                    float(spearmanr(m[el], lp_inst[el]).statistic), 4),
            }
            results.append(entry)
        print(f"{kind}: layers done", flush=True)

    # capped-counts robustness for the J-lens headline (matching machinery
    # re-run on qwen_capped; pool re-derived under the same rules)
    counts_c = counts_npz["qwen_capped"]
    log_f_c = np.full(n_vocab, np.nan)
    seen_c = counts_c > 0
    log_f_c[seen_c] = np.log(counts_c[seen_c].astype(np.float64))
    eligible_c = eligible & seen_c  # same cleanliness rules, capped visibility
    j_idx = kinds.index("J")
    capped_robustness = []
    for layer_idx, layer in enumerate(layers):
        m = mt["m_t"][j_idx, layer_idx][:n_vocab].astype(np.float64)
        e = topk_enrichment(m, delta, log_f_c, eligible_c, seen_c, flags, 100)
        capped_robustness.append({"layer": layer,
                                  "residualised_median_excess": e["residualised_median_excess"],
                                  "sign_test_p": e.get("sign_test_p")})

    # band summaries + the pre-set decision rule, evaluated per instrument
    def band_mean(kind: str, band: tuple[int, int], k: int, field: str):
        vals = [r["enrichment"][str(k)][field] for r in results
                if r["kind"] == kind and band[0] <= r["layer"] <= band[1]
                and r["enrichment"][str(k)][field] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    def band_mean_wf(kind: str, cell: str, band: tuple[int, int], k: int, field: str):
        vals = [r["enrichment_wordfreq"][cell][str(k)][field] for r in results
                if r["kind"] == kind and band[0] <= r["layer"] <= band[1]
                and r["enrichment_wordfreq"][cell][str(k)][field] is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    summary = {}
    for kind in kinds:
        early_res = band_mean(kind, EARLY_BAND, 100, "residualised_median_excess")
        early_raw = band_mean(kind, EARLY_BAND, 100, "raw_median_delta_gap")
        early_ps = [r["enrichment"]["100"].get("sign_test_p") for r in results
                    if r["kind"] == kind and EARLY_BAND[0] <= r["layer"] <= EARLY_BAND[1]]
        sig = [p is not None and p < SIGN_TEST_ALPHA for p in early_ps]
        nl_ps = [r["enrichment_wordfreq"]["wf_nonlatin"]["1000"].get("sign_test_p")
                 for r in results
                 if r["kind"] == kind and EARLY_BAND[0] <= r["layer"] <= EARLY_BAND[1]]
        summary[kind] = {
            "early_L0_4_raw_median_gap_mean": early_raw,
            "early_L0_4_residualised_excess_mean": early_res,
            "early_layers_significant": sig,
            "mid_L5_28_raw_median_gap_mean": band_mean(kind, MID_BAND, 100, "raw_median_delta_gap"),
            "mid_L5_28_residualised_excess_mean": band_mean(kind, MID_BAND, 100,
                                                            "residualised_median_excess"),
            "early_L0_4_junkcell_excess_mean_K1000": band_mean_wf(
                kind, "wf_nonlatin", EARLY_BAND, 1000, "residualised_median_excess"),
            "early_layers_junkcell_significant_K1000": [
                p is not None and p < SIGN_TEST_ALPHA for p in nl_ps],
            "early_L0_4_wf_latin_excess_mean_K100": band_mean_wf(
                kind, "wf_latin", EARLY_BAND, 100, "residualised_median_excess"),
            "decision_rule": (
                "agree" if early_raw is not None and early_res is not None
                and np.sign(early_raw) == np.sign(early_res) else "disagree"
            ),
        }

    # figure: residualised median excess by depth, K=100 solid, K=1000 dashed
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for kind in kinds:
        for k, style in zip(TOP_KS, ("-", "--")):
            rows = [r for r in results if r["kind"] == kind]
            ax.plot([r["layer"] for r in rows],
                    [r["enrichment"][str(k)]["residualised_median_excess"] for r in rows],
                    style, marker=".", label=f"{kind} top-{k}")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("layer")
    ax.set_ylabel("median Δ_t excess over frequency-matched controls")
    ax.set_title("H2: are the offset's top tokens tuning-inhibited beyond their rarity?")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)

    out = {
        "meta": {
            "gate": gate,
            "mt_npz": MT_NPZ,
            "delta_npz": DELTA_NPZ,
            "matching": {
                "counts": "qwen_full (what src.delta_t validated on; capped robustness below)",
                "freq_tol_log_count": FREQ_TOL,
                "min_controls": MIN_CONTROLS,
                "eligible_pool": int(eligible.sum()),
                "grid_step_for_vocabwide_residual": GRID_STEP,
            },
            "correlation_population": "eligible pool only (disclosed; junk/punct/special "
                                      "tokens cannot be frequency-matched so are absent "
                                      "from every correlation here)",
            "sign_test_caveat": "probes share controls; the binomial p treats excesses "
                                "as independent, so it is approximate",
            "decision_rule": "pre-set in steps.md step 5: both ways; agree -> raw headline, "
                             "residualised robustness; disagree -> residualised primary",
            "wordfreq_frame": {
                "added": "24 Aug, in-step, before its numbers were seen (D36): matches on "
                         "wordfreq Zipf so non-Latin (junk) probes are testable at all; "
                         "pile frame stays primary (the validated machinery). Cells are "
                         "WITHIN-SCRIPT (see module docstring: Delta_t carries a "
                         "script-level shift, so script-blind controls manufacture excess).",
                "zipf_tol": round(float(ZIPF_TOL), 4),
                "eligible_pools": {cell: int(spec["eligible"].sum())
                                   for cell, spec in wf_cells.items()},
                "delta_caveat": "Delta_t was measured on English contexts; for non-Latin "
                                "tokens it reads 'how much more the base model leaks toward "
                                "these in English text'",
                "unmatched_share_note": "in wf cells the share is relative to the full "
                                        "top-K, most of which is the other script class at "
                                        "many layers; n_probes_matched is the usable count",
            },
            "script_level_delta": {
                "median_delta_nonlatin_zipf_covered": round(float(np.median(
                    delta[flags["non_latin"] & seen_wf])), 4),
                "median_delta_latin_zipf_covered": round(float(np.median(
                    delta[~flags["non_latin"] & seen_wf & ~flags["punctuation"]])), 4),
                "note": "whole-script shift, no frequency control -- context for the "
                        "wordfreq frame, not an H2 statistic",
            },
        },
        "summary_by_instrument": summary,
        "capped_counts_robustness_J_top100": capped_robustness,
        "results": results,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    for kind in kinds:
        s = summary[kind]
        print(f"{kind}: early L0-4 raw {s['early_L0_4_raw_median_gap_mean']} | "
              f"residualised {s['early_L0_4_residualised_excess_mean']} | "
              f"{s['decision_rule']}")
    print(f"wrote {OUT_JSON} and {OUT_PNG}")


if __name__ == "__main__":
    main()
