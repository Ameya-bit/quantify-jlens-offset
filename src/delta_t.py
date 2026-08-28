"""Step 3 asset: Delta_t, the base-vs-instruct suppression axis (feeds step 5).

H2 says the J-lens's early-layer junk is enriched for tokens that RLHF
*suppressed* -- the model still represents them, the tuned model just refuses
to say them. That needs a per-token measure of "how much more does the base
model want this token than the instruct model":

    Delta_t = mean log P_base(t | context) - mean log P_instruct(t | context)

WORKED EXAMPLE, because the one-line formula hides what is actually summed.
Take one document and stop at one position inside it, say after "The cat sat
on the". At that point each model emits a score for every one of the 248,320
tokens in the vocabulary -- " mat" high, " table" middling, " fuck" very low
-- which log_softmax turns into a log-probability. Both models see the SAME
text at the SAME position, so their two vectors are directly comparable;
subtract them and you have one sample of Delta for all 248,320 tokens at
once. Repeat at every position and every document, and average.

So Delta_t[" fuck"] reads: "averaged over 23,912 different places in real
text, how much more log-probability did the base model put on ' fuck' than
the instruct model did?" Note two things that are easy to get wrong: it uses
EVERY position from 4 onward, not just the document's last one; and at each
position it scores EVERY token in the vocabulary, not just the token that
actually came next. The actual continuation is never used -- this measures
what the two models *want*, not whether either is right.

Contexts: N_TEXTS pile-10k documents from rows 25+ (outside every fit corpus
in this study), truncated to 128 tokens, every position >= 4 -- the same
skip-first-4 convention as the lens fit and the step-2 survey. One forward
pass yields all positions, so this is ~124 samples per document.

WHAT THE AXIS ACTUALLY MEASURES (read before using it for H2). Ranked by
Delta, the top is not obscenity -- it is informal, archaic and misspelled
English (" ordinarily", " thankfully", " admittedly", " anyways", " hubby",
" thru", " seper") plus scraped-text artefacts (" ;-)", "...the", "-and"),
and the most-promoted end is entirely whitespace and chat formatting. The
dominant axis is "raw web text style" vs "clean assistant style". Obscenity
is a real but secondary component of that: profanity lands in the top
0.3-4% of the vocabulary, which is why the frequency-matched gate below
passes. H2 claims should say "tuning-stage distribution shift, of which
obscenity suppression is one component", not "suppression" unqualified.

The two models are loaded one at a time and freed in between; holding both
Qwen3.5-4B and Qwen3.5-4B-Base on MPS at once does not fit. Their tokenizers
were verified identical in src.frequencies, so Delta_t is well-defined
token-by-token and needs no fallback token list.

**Delta_t carries a frequency confound and must never be read raw.** Measured
here: Spearman(log f_t, Delta_t) = +0.354 over the 106k tokens pile-10k can
measure, with median Delta rising monotonically across frequency quintiles
(-0.047, -0.033, +0.001, +0.045, +0.107). Cause: the instruct model moves
probability mass onto chat and formatting tokens, so ordinary content tokens
are deflated relative to the base model roughly in proportion to how much
mass they had. Common words therefore look "suppressed" without any
suppression having happened.

This is why steps.md step 5 demands **frequency-matched controls**, and it is
not optional. A first attempt at validating this asset compared profanity
(median log f 3.37) against neutral words (median log f 6.43) and concluded
the axis did not work, because the ~20x more frequent controls carried a
larger frequency-driven Delta. The validation below matches on frequency
instead: each profanity probe is compared to clean, non-special tokens within
+-FREQ_TOL in log count, and the gate is on the paired difference.

Run: .venv/bin/python -m src.delta_t   -> results/step3/step3_delta_t.json (summary)
                                       -> results/step3/step3_delta_t.npz  (full vector)
"""

from __future__ import annotations

import argparse
import gc
import json

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.flags import token_flags
from src.junk_survey import DATASET_ID, MAX_SEQ_LEN, SKIP_FIRST, pick_rows
from src.lens import DEFAULT_DEVICE, MODEL_ID

BASE_ID = "Qwen/Qwen3.5-4B-Base"
N_TEXTS = 200
SEED = 1  # deliberately NOT the step-2 survey's seed 0
SUMMARY_PATH = "results/step3/step3_delta_t.json"
VECTOR_PATH = "results/step3/step3_delta_t.npz"

# Validation probes. Suppressed: expected high Delta_t (base wants them more).
# Neutral: expected mid-pack. Both are checked by percentile, not raw value.
PROBE_SUPPRESSED = [
    " fuck", " shit", " bitch", " asshole", " bastard", " damn", " crap",
    " slut", " whore", " dick", " piss", " retard",
]
PROBE_NEUTRAL = [
    " table", " river", " Tuesday", " economic", " method", " garden",
    " machine", " letter", " signal", " orange",
]
COUNTS_PATH = "results/step3/step3_token_counts.npz"
FREQ_TOL = 0.25      # log-count window for a frequency-matched control
MIN_CONTROLS = 20    # a probe with fewer matched controls is skipped


def mean_logprobs(model_id: str, texts: list[str], tokenizer, device: str) -> np.ndarray:
    """Mean log-softmax over every (document, position>=4) pair, per token."""
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.to(device).eval()
    total = None
    n = 0
    with torch.no_grad():
        for i, text in enumerate(texts):
            ids = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN
            )["input_ids"].to(device)
            if ids.shape[1] <= SKIP_FIRST:
                continue
            logits = model(input_ids=ids).logits[0, SKIP_FIRST:].float()
            # sum on device in fp32 (MPS has no float64), accumulate in fp64 on CPU
            summed = torch.log_softmax(logits, dim=-1).sum(0).cpu().numpy().astype(np.float64)
            total = summed if total is None else total + summed
            n += logits.shape[0]
            if (i + 1) % 50 == 0:
                print(f"   {model_id}: {i + 1}/{len(texts)} docs, {n} positions", flush=True)
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    print(f"   {model_id}: {n} positions total")
    return total / n


def measure_frequency_confound(delta: np.ndarray, counts: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """How much of Delta_t is explained by how common a token is?

    Returns (summary, log_f, seen) so callers can reuse the aligned arrays.
    `log_f` is NaN for tokens pile-10k never saw.
    """
    n_align = min(delta.size, counts.size)
    seen = np.zeros(delta.size, dtype=bool)
    seen[:n_align] = counts[:n_align] > 0
    log_f = np.full(delta.size, np.nan)
    log_f[seen] = np.log(counts[: delta.size][seen[: counts.size]])

    rho = float(spearmanr(log_f[seen], delta[seen]).statistic)
    edges = np.quantile(log_f[seen], [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    quintiles = {}
    for i in range(5):
        band = seen & (log_f >= edges[i]) & (log_f <= edges[i + 1])
        quintiles[f"Q{i + 1}"] = {
            "log_f_range": [round(float(edges[i]), 2), round(float(edges[i + 1]), 2)],
            "median_delta": round(float(np.median(delta[band])), 4),
            "n": int(band.sum()),
        }
    summary = {
        "spearman_logf_vs_delta": round(rho, 4),
        "median_delta_by_frequency_quintile": quintiles,
        "consequence": (
            "Delta_t must never be read raw. Step 5's frequency-matched control "
            "is mandatory, not a refinement: common tokens look suppressed "
            "because the instruct model shifts mass onto chat and formatting "
            "tokens, deflating content tokens in proportion to the mass they held."
        ),
    }
    return summary, log_f, seen


def is_special(tok_str: str) -> bool:
    """`<|im_end|>` and friends -- they dominate both tails and are not words."""
    return tok_str.startswith("<|") and tok_str.endswith("|>")


def build_control_pool(tokenizer, seen: np.ndarray) -> np.ndarray:
    """Tokens usable as frequency-matched controls: measurable, ordinary words.

    Excludes anything junk-flagged, punctuation, or a special token -- a
    control has to be the sort of token a probe could have been compared to.
    """
    eligible = np.zeros(seen.size, dtype=bool)
    for i in np.flatnonzero(seen):
        t = tokenizer.decode([int(i)])
        f = token_flags(t)
        eligible[i] = not (f["is_junk"] or f["punctuation"] or is_special(t))
    return eligible


def probe_against_matched_controls(
    words: list[str], delta: np.ndarray, log_f: np.ndarray,
    eligible: np.ndarray, seen: np.ndarray, tokenizer,
) -> list[dict]:
    """For each probe word, compare its Delta to the median Delta of ordinary
    tokens of the SAME frequency. Raw Delta is uninterpretable (see
    measure_frequency_confound); this excess is the quantity H2 wants."""
    out = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) != 1 or not seen[ids[0]]:
            continue
        tid = ids[0]
        near = eligible & (np.abs(log_f - log_f[tid]) <= FREQ_TOL)
        near[tid] = False
        if near.sum() < MIN_CONTROLS:
            continue
        control = float(np.median(delta[near]))
        out.append({
            "token": w,
            "delta": round(float(delta[tid]), 4),
            "log_f": round(float(log_f[tid]), 3),
            "matched_control_median_delta": round(control, 4),
            "excess_over_matched_controls": round(float(delta[tid]) - control, 4),
            "n_controls": int(near.sum()),
        })
    return out


def extreme_tokens(delta: np.ndarray, tokenizer, n: int = 25) -> tuple[list, list]:
    """Most-suppressed and most-promoted real tokens (special tokens dropped)."""
    ranked = [
        int(i) for i in np.argsort(-delta) if not is_special(tokenizer.decode([int(i)]))
    ]
    fmt = lambda i: (tokenizer.decode([i]), round(float(delta[i]), 3))
    return [fmt(i) for i in ranked[:n]], [fmt(i) for i in reversed(ranked[-n:])]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-cache", action="store_true",
        help="re-run only the analysis, reading the vectors from the saved .npz "
             "(skips both ~5 min model passes; use when changing the analysis)",
    )
    args = parser.parse_args()
    device = DEFAULT_DEVICE
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    all_texts = load_dataset(DATASET_ID, split="train")["text"]
    rows = pick_rows(N_TEXTS, SEED, len(all_texts))
    texts = [all_texts[r] for r in rows]

    if args.from_cache:
        cached = np.load(VECTOR_PATH)
        base, instruct = cached["mean_logprob_base"], cached["mean_logprob_instruct"]
        print(f"loaded cached vectors from {VECTOR_PATH} (no model passes)")
    else:
        base = mean_logprobs(BASE_ID, texts, tokenizer, device)
        instruct = mean_logprobs(MODEL_ID, texts, tokenizer, device)
    delta = (base - instruct).astype(np.float64)

    confound, log_f, seen = measure_frequency_confound(
        delta, np.load(COUNTS_PATH)["qwen_full"]
    )
    print(f"\nSpearman(log f_t, Delta_t) = "
          f"{confound['spearman_logf_vs_delta']:+.3f}  <- the confound; "
          f"step 5 MUST frequency-match")

    eligible = build_control_pool(tokenizer, seen)
    print(f"   {int(eligible.sum())} tokens eligible as frequency-matched controls")

    sup, neu = (
        probe_against_matched_controls(w, delta, log_f, eligible, seen, tokenizer)
        for w in (PROBE_SUPPRESSED, PROBE_NEUTRAL)
    )
    exc_sup = float(np.median([p["excess_over_matched_controls"] for p in sup]))
    exc_neu = float(np.median([p["excess_over_matched_controls"] for p in neu]))
    validation_ok = exc_sup > 0.05 and exc_sup > 2 * exc_neu
    top, bottom = extreme_tokens(delta, tokenizer)

    summary = {
        "definition": "mean log P_base(t) - mean log P_instruct(t); positive = suppressed by tuning",
        "base_model": BASE_ID,
        "instruct_model": MODEL_ID,
        "n_texts": len(texts),
        "seed": SEED,
        "rows": rows,
        "skip_first": SKIP_FIRST,
        "max_seq_len": MAX_SEQ_LEN,
        "delta_stats": {
            "mean": round(float(delta.mean()), 4),
            "std": round(float(delta.std()), 4),
            "min": round(float(delta.min()), 4),
            "max": round(float(delta.max()), 4),
        },
        "frequency_confound": confound,
        "validation": {
            "pass": bool(validation_ok),
            "gate": (
                "median excess of profanity over FREQUENCY-MATCHED controls > 0.05 "
                "and more than 2x the neutral probes' excess"
            ),
            "freq_tol_log_count": FREQ_TOL,
            "n_eligible_controls": int(eligible.sum()),
            "median_excess_suppressed": round(exc_sup, 4),
            "median_excess_neutral": round(exc_neu, 4),
            "suppressed_probes": sup,
            "neutral_probes": neu,
        },
        "top25_most_suppressed": top,
        "bottom25_most_promoted": bottom,
    }
    np.savez_compressed(
        VECTOR_PATH,
        delta_t=delta.astype(np.float32),
        mean_logprob_base=base.astype(np.float32),
        mean_logprob_instruct=instruct.astype(np.float32),
    )
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nDelta_t: mean {delta.mean():+.4f}, std {delta.std():.4f}, "
          f"range [{delta.min():+.3f}, {delta.max():+.3f}]")
    print(f"validation (frequency-matched): profanity excess {exc_sup:+.4f}, "
          f"neutral excess {exc_neu:+.4f} -> {'PASS' if validation_ok else 'FAIL'}")
    print(f"most suppressed: {[t for t, _ in top[:12]]}")
    print(f"most promoted:   {[t for t, _ in bottom[:12]]}")
    print(f"wrote {VECTOR_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
