"""Step 3 asset: token frequency `log f_t` under three tokenizers.

H1 says the J-lens offset tracks how *common* a token is. That needs a
frequency estimate per token. We count occurrences over all 10,000 pile-10k
documents under each tokenizer that a later step needs:

  Qwen   -- the headline model (steps 4c, 5, 7)
  Pythia -- the Route A arm; Pythia was *trained* on the Pile, so these
            counts match its training distribution rather than proxying it
  GPT-2  -- the weights-only phoenix cell (W_U.beta vs log f_t, step 4a)

Two counts are kept per tokenizer:
  `full`   -- every token of every document
  `capped` -- at most CAP_TOKENS per document  <-- PRIMARY for step 4

Both, because one pile-10k document is 981k tokens -- **6.3%** of the entire
15.6M-token corpus on its own -- and a unigram prior that a single document
can move by 6% is a sampling artefact rather than a fact about English.
The two agree at Spearman/Pearson ~0.97 in log space but not ~0.99, so the
choice is not cosmetic: `capped` is primary (variance from one document
bounded), `full` is retained so step 4 can show the result holds either way.

Coverage is reported broken down by junk flag, because it is a live threat
to the step-4c regression: junk tokens are rare tokens, and rare tokens are
exactly the ones a 10k-document sample cannot measure. A token seen zero
times has no log-frequency, so if most non-Latin tokens are unseen, the
regression silently drops the tokens the study is about. Only 42.8% of
Qwen's 248k vocabulary is ever seen and only 9.3% of its non-Latin tokens
are -- pile-10k is English and Qwen's vocabulary is not.

Two things keep that from sinking step 4c, both measured here:
  `readout_exposure` -- the share of tokens that actually *appear* in the
    step-2 readouts and have no frequency estimate. Far smaller than the
    vocabulary figure (J-lens 11.1% overall, 21.7% at layers 0-5), because
    the unmeasurable tail of the vocabulary is mostly never read out either.
    Step 4c states this as "what the regression drops", not "what the
    vocabulary looks like".
  `id_rank_proxy` -- Qwen's BPE token ids run roughly in merge order, so a
    low id means an early merge means a frequent token. That gives a
    frequency *rank* covering 100% of the vocabulary, derived from Qwen's
    own training corpus rather than from the Pile. Spearman against pile
    counts on the seen tokens is -0.677, so it is a real but noisy signal:
    it is a robustness check for step 4c, not the primary regressor.

Also performs the step-3 tokenizer equality check (Qwen base vs instruct):
if the two disagree, Delta_t needs the token-list fallback.

Run: .venv/bin/python -m src.frequencies
  -> results/step3/step3_frequencies.json   (summary, committed)
  -> results/step3/step3_token_counts.npz   (raw counts, committed)
"""

from __future__ import annotations

import json

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

from scipy.stats import spearmanr

from src.flags import token_flags

DATASET_ID = "NeelNanda/pile-10k"
CAP_TOKENS = 8192  # per-document cap for the `capped` variant
BATCH = 500
TOKENIZERS = {
    "qwen": "Qwen/Qwen3.5-4B",
    "pythia": "EleutherAI/pythia-1.4b",
    "gpt2": "openai-community/gpt2",
}
QWEN_BASE = "Qwen/Qwen3.5-4B-Base"
READOUTS_PATH = "results/step2/step2_readouts.json"
SUMMARY_PATH = "results/step3/step3_frequencies.json"
COUNTS_PATH = "results/step3/step3_token_counts.npz"


def count_corpus(tokenizer, texts: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (full_counts, capped_counts, doc-length stats) over the corpus."""
    size = max(len(tokenizer), max(tokenizer.get_vocab().values()) + 1)
    full = np.zeros(size, dtype=np.int64)
    capped = np.zeros(size, dtype=np.int64)
    doc_lengths = []
    for start in range(0, len(texts), BATCH):
        for ids in tokenizer(texts[start : start + BATCH])["input_ids"]:
            doc_lengths.append(len(ids))
            arr = np.asarray(ids, dtype=np.int64)
            np.add.at(full, arr, 1)
            np.add.at(capped, arr[:CAP_TOKENS], 1)
    lengths = np.asarray(doc_lengths)
    stats = {
        "n_docs": int(lengths.size),
        "total_tokens_full": int(full.sum()),
        "total_tokens_capped": int(capped.sum()),
        "longest_doc_tokens": int(lengths.max()),
        "longest_doc_share_of_corpus": round(float(lengths.max() / full.sum()), 4),
        "docs_over_cap": int((lengths > CAP_TOKENS).sum()),
    }
    return full, capped, stats


def coverage_by_flag(tokenizer, counts: np.ndarray) -> dict:
    """What fraction of each token category is never observed? The step-4c
    regression can only use tokens with a nonzero count."""
    groups: dict[str, list[int]] = {
        "all": [], "non_latin": [], "byte_fragment": [], "punctuation": [], "clean": [],
    }
    for i in range(len(counts)):
        f = token_flags(tokenizer.decode([i]))
        groups["all"].append(i)
        if f["non_latin"]:
            groups["non_latin"].append(i)
        elif f["byte_fragment"]:
            groups["byte_fragment"].append(i)
        elif f["punctuation"]:
            groups["punctuation"].append(i)
        else:
            groups["clean"].append(i)
    out = {}
    for name, ids in groups.items():
        idx = np.asarray(ids, dtype=np.int64)
        seen = int((counts[idx] > 0).sum())
        out[name] = {
            "n_tokens": int(idx.size),
            "seen": seen,
            "unseen": int(idx.size - seen),
            "frac_seen": round(seen / idx.size, 4) if idx.size else None,
        }
    return out


def main() -> None:
    texts = load_dataset(DATASET_ID, split="train")["text"]
    summary: dict = {"dataset": DATASET_ID, "cap_tokens": CAP_TOKENS, "tokenizers": {}}
    arrays: dict[str, np.ndarray] = {}

    for name, model_id in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(model_id)
        full, capped, stats = count_corpus(tok, texts)
        arrays[f"{name}_full"] = full
        arrays[f"{name}_capped"] = capped

        # Do the two count variants tell the same story? Compared on tokens
        # seen in BOTH, in log space, since that is how step 4c uses them.
        both = (full > 0) & (capped > 0)
        r = float(
            np.corrcoef(np.log(full[both]), np.log(capped[both]))[0, 1]
        )
        top = [
            (tok.decode([int(i)]), int(full[i]))
            for i in np.argsort(-full)[:15]
        ]
        summary["tokenizers"][name] = {
            "model_id": model_id,
            "vocab_size": int(full.size),
            **stats,
            "log_freq_corr_full_vs_capped": round(r, 5),
            "n_tokens_in_both": int(both.sum()),
            "coverage_by_flag_full": coverage_by_flag(tok, full),
            "top15_full": top,
        }
        print(f"{name}: {stats['total_tokens_full']:,} tokens, "
              f"longest doc {stats['longest_doc_tokens']:,} "
              f"({100*stats['longest_doc_share_of_corpus']:.1f}% of corpus), "
              f"log-freq corr full~capped = {r:.5f}")
        cov = summary["tokenizers"][name]["coverage_by_flag_full"]
        print(f"   coverage: all {cov['all']['frac_seen']:.3f} | "
              f"clean {cov['clean']['frac_seen']:.3f} | "
              f"non-Latin {cov['non_latin']['frac_seen']:.3f} | "
              f"punct {cov['punctuation']['frac_seen']:.3f}")

    # --- how exposed is step 4c to tokens with no frequency estimate? ---
    qtok = AutoTokenizer.from_pretrained(TOKENIZERS["qwen"])
    qfull = arrays["qwen_full"]
    with open(READOUTS_PATH) as f:
        readouts = json.load(f)
    single_id = {}

    def unmeasured(token: str) -> bool:
        if token not in single_id:
            ids = qtok.encode(token, add_special_tokens=False)
            single_id[token] = len(ids) == 1 and qfull[ids[0]] == 0
        return single_id[token]

    exposure: dict[str, dict] = {}
    for kind in readouts["meta"]["kinds"]:
        cells = [c for c in readouts["cells"] if c["kind"] == kind]
        n = miss = 0
        bands = {"L0_5": range(0, 6), "L14_21": range(14, 22), "L26_30": range(26, 31)}
        band_counts = {b: [0, 0] for b in bands}
        for c in cells:
            for t in c["top"]:
                n += 1
                bad = unmeasured(t["t"])
                miss += bad
                for b, rng in bands.items():
                    if c["layer"] in rng:
                        band_counts[b][0] += 1
                        band_counts[b][1] += bad
        exposure[kind] = {
            "overall": round(miss / n, 4),
            **{b: round(v[1] / v[0], 4) for b, v in band_counts.items()},
        }
    print("\nreadout exposure (share of step-2 readout tokens with no frequency):")
    for kind, e in exposure.items():
        print(f"   {kind:>5}: overall {e['overall']:.3f} | L0-5 {e['L0_5']:.3f} | "
              f"L14-21 {e['L14_21']:.3f} | L26-30 {e['L26_30']:.3f}")
    summary["readout_exposure"] = {
        "source": READOUTS_PATH,
        "definition": "fraction of top-10 readout tokens whose Qwen id has zero pile-10k count",
        "by_kind": exposure,
    }

    # --- token id as a full-vocabulary frequency rank ---
    seen = qfull > 0
    ids = np.arange(qfull.size)
    rho = float(spearmanr(ids[seen], qfull[seen]).statistic)
    bands = [(0, 10_000), (10_000, 50_000), (50_000, 100_000),
             (100_000, 150_000), (150_000, 200_000), (200_000, int(qfull.size))]
    summary["id_rank_proxy"] = {
        "spearman_id_vs_pile_count_on_seen": round(rho, 4),
        "frac_seen_by_id_band": {
            f"{lo}-{hi}": round(float(seen[lo:hi].mean()), 4) for lo, hi in bands
        },
        "role": (
            "robustness check for step 4c only -- covers 100% of the vocabulary "
            "and derives from Qwen's own training corpus, but is a noisy rank "
            "(non-monotone across vocabulary blocks), not a count"
        ),
    }
    print(f"token-id frequency rank: Spearman vs pile counts = {rho:+.3f}")

    # --- step-3 tokenizer equality check: Qwen instruct vs base ---
    a = AutoTokenizer.from_pretrained(TOKENIZERS["qwen"])
    b = AutoTokenizer.from_pretrained(QWEN_BASE)
    identical = a.get_vocab() == b.get_vocab()
    probe = "The quick brown fox; 你好, café ((\n\nsupercalifragilistic 12345"
    same_encoding = a(probe)["input_ids"] == b(probe)["input_ids"]
    summary["qwen_tokenizer_equality"] = {
        "instruct": TOKENIZERS["qwen"],
        "base": QWEN_BASE,
        "vocab_identical": identical,
        "probe_encoding_identical": same_encoding,
        "consequence": (
            "identical -> Delta_t can be computed token-by-token across the two "
            "models; if it were False, Delta_t would need the curated token-list "
            "fallback (steps.md step 3)"
        ),
    }
    print(f"\nQwen instruct vs base tokenizer: vocab identical={identical}, "
          f"probe identical={same_encoding}")

    np.savez_compressed(COUNTS_PATH, **arrays)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"wrote {COUNTS_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
