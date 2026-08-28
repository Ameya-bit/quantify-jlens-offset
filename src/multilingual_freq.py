"""Step 3 asset: multilingual word frequency, to close the non-Latin gap.

THE GAP THIS EXISTS TO CLOSE. `src.frequencies` counts pile-10k, which is
English, so it sees only 9.3% of Qwen's non-Latin vocabulary. `src.zipf_
frequency` reads Qwen's own merge order, but that carries no information past
token id 100k (rho +0.04 / +0.01 / -0.04 by band) -- exactly where the
non-Latin vocabulary lives. Both fail on the same subpopulation, and that
subpopulation is what the junk in the readouts is made of. Without a
frequency number for those tokens, H1 (a whole-vocabulary claim) gets tested
only on Latin tokens, and the separate question of whether frequency
structures the junk cannot be asked at all (the D31 non-Latin cell).
(Reworded 24 Aug: the original phrasing called the junk "H1's actual
subject" -- a residue of the pre-correction framing; see devlog 0.1.0.)

`wordfreq` (pypi, Speer et al.) is a frequency table over 42 languages built
from Wikipedia, Google Books, Reddit, Twitter, OpenSubtitles, SUBTLEX and the
Leeds corpus, reported on a Zipf scale (log10 occurrences per billion words).
It covers Chinese, Japanese, Korean, Russian, Arabic and more, which is the
one thing our other two sources cannot do.

Per token we take the MAX Zipf score over LANGS. Rationale: Qwen is
multilingual and its training mixture is undisclosed, so "how common is this
string in whichever language uses it" is the closest available stand-in for
"how common is this string in Qwen's corpus". Taking the max rather than a
weighted sum avoids inventing mixture weights we do not know.

HOW THIS IS VALIDATED, AND HOW IT IS NOT. The obvious test -- "does wordfreq
agree with pile-10k counts on non-Latin tokens?" -- is close to circular and
was rejected after being run. It asks whether a multilingual source agrees
with an English-only source about non-English tokens; if it did, it would be
adding nothing. The measured disagreement is the asset working, not failing:
the token 'de' (Chinese) appears 86 times in pile-10k and scores Zipf 7.79 in
wordfreq, because it is the commonest character in Chinese and the Pile is
not Chinese. Same for ' privet' (pile 1, Zipf 5.13) and 'nin hao' (pile 0,
Zipf 3.93).

Three tests are used instead:
  1. Face validity on non-Latin, printed for human judgement. The top-scoring
     non-Latin tokens come out as the highest-frequency function words of
     their languages -- Chinese de, Japanese no/ni, Russian v, Hindi ke/hai,
     Arabic fi/min, Korean i.
  2. Agreement with pile counts on LATIN tokens with a non-noisy count, where
     the Pile *is* a valid yardstick. rho rises monotonically as the noisy
     tail is trimmed: +0.14 (count>=1), +0.28 (>=10), +0.39 (>=100), +0.52
     (>=1000). Moderate, as two genuinely different corpora should be.
  3. Coverage of the tokens that actually appear in step-2 readouts, which is
     the operational question.

KNOWN LIMITS, both structural:
  - wordfreq is WORD-level. Latin subword fragments ('zinho', 'correcti')
    are not words and score 0, so this source is weak exactly where pile-10k
    is strong. The two are complementary, not redundant, and neither alone
    covers the vocabulary.
  - It scores a SUBSTRING when the token is not a bare word. '.Scene' is
    scored as "scene", '.cpu' as "cpu", '_exchange' as "exchange" -- the
    punctuation is stripped and whatever word remains is looked up. That
    score describes the embedded word, not the token, and is an
    over-estimate of the token's own frequency. `is_bare_word` marks the
    tokens where the score can be trusted as the token's own; it is stored
    per token in the .npz so step 4c can exclude them, or report with and
    without, rather than inheriting the inflation silently.
  - Its corpora are not Qwen's, so this shares the wrong-corpus problem with
    pile-10k. It fixes coverage, not provenance.
  - The dataset was frozen around 2021 because generative-AI text polluted
    the web. For a frequency prior over human text that is a feature.

Run: .venv/bin/python -m src.multilingual_freq
  -> results/step3/step3_multilingual_freq.json  (coverage + validation report)
  -> results/step3/step3_wordfreq.npz            (Zipf score per token)
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr
from transformers import AutoTokenizer
from wordfreq import zipf_frequency

from src.flags import token_flags
from src.lens import MODEL_ID

# Major languages plausibly present in a multilingual pretraining mixture.
LANGS = ["en", "zh", "ja", "ko", "ru", "ar", "es", "fr", "de", "pt", "it", "hi"]
COUNTS_PATH = "results/step3/step3_token_counts.npz"
READOUTS_PATH = "results/step2/step2_readouts.json"
SUMMARY_PATH = "results/step3/step3_multilingual_freq.json"
VECTOR_PATH = "results/step3/step3_wordfreq.npz"


def score_vocabulary(tokenizer, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Max Zipf frequency over LANGS per token, plus `is_bare_word`.

    `is_bare_word` is True when the stripped token is purely alphabetic, i.e.
    wordfreq scored the token itself. When it is False the token carried
    punctuation or digits that wordfreq stripped before lookup, so the score
    belongs to an embedded substring and over-states the token's frequency.
    """
    scores = np.zeros(size, dtype=np.float32)
    bare = np.zeros(size, dtype=bool)
    for i in range(size):
        s = tokenizer.decode([i]).strip()
        if not s:
            continue
        bare[i] = s.isalpha()
        scores[i] = max(zipf_frequency(s, lang) for lang in LANGS)
    return scores, bare


def classify(tokenizer, size: int) -> dict[str, np.ndarray]:
    groups = {"latin_clean": [], "non_latin": [], "punctuation": []}
    for i in range(size):
        f = token_flags(tokenizer.decode([i]))
        if f["non_latin"] or f["byte_fragment"]:
            groups["non_latin"].append(i)
        elif f["punctuation"]:
            groups["punctuation"].append(i)
        else:
            groups["latin_clean"].append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    counts = np.load(COUNTS_PATH)["qwen_full"]
    size = counts.size
    wf, bare = score_vocabulary(tokenizer, size)
    groups = classify(tokenizer, size)
    pile_seen, wf_seen = counts > 0, wf > 0

    # readout tokens, needed by both the substring sizing and the coverage test
    with open(READOUTS_PATH) as f:
        cells = json.load(f)["cells"]
    cache: dict[str, int | None] = {}

    def token_id_of(t: str) -> int | None:
        if t not in cache:
            ids = tokenizer.encode(t, add_special_tokens=False)
            cache[t] = ids[0] if len(ids) == 1 else None
        return cache[t]

    report: dict = {
        "source": "wordfreq (pypi), max Zipf over " + ",".join(LANGS),
        "purpose": "close the non-Latin frequency gap left open by src.frequencies (D18) and src.zipf_frequency (D26)",
        "coverage": {},
        "validation_vs_pile": {},
        "readout_coverage": {},
    }

    # --- 1. coverage: who can see which part of the vocabulary? ---
    for name, ids in groups.items():
        both = pile_seen[ids] & wf_seen[ids]
        report["coverage"][name] = {
            "n_tokens": int(ids.size),
            "pile_only": round(float((pile_seen[ids] & ~wf_seen[ids]).mean()), 4),
            "wordfreq_only": round(float((~pile_seen[ids] & wf_seen[ids]).mean()), 4),
            "both": round(float(both.mean()), 4),
            "neither": round(float((~pile_seen[ids] & ~wf_seen[ids]).mean()), 4),
            "frac_seen_pile": round(float(pile_seen[ids].mean()), 4),
            "frac_seen_wordfreq": round(float(wf_seen[ids].mean()), 4),
            "frac_seen_either": round(float((pile_seen[ids] | wf_seen[ids]).mean()), 4),
        }

    # --- 2. validation: do the two agree where both can see? ---
    # If wordfreq tracks pile counts on tokens both measure, it is measuring
    # frequency and can be trusted where pile-10k is blind.
    for name, ids in groups.items():
        m = ids[pile_seen[ids] & wf_seen[ids]]
        report["validation_vs_pile"][name] = (
            {"n": int(m.size), "spearman": None, "note": "too few tokens"}
            if m.size < 50 else
            {"n": int(m.size),
             "spearman": round(float(spearmanr(wf[m], counts[m]).statistic), 4)}
        )

    for kind in ("J", "R", "logit"):
        n = no_pile = no_pile_no_wf = no_pile_but_wf = 0
        for c in cells:
            if c["kind"] != kind:
                continue
            for t in c["top"]:
                tid = token_id_of(t["t"])
                if tid is None:
                    continue
                n += 1
                if not pile_seen[tid]:
                    no_pile += 1
                    if wf_seen[tid]:
                        no_pile_but_wf += 1
                    else:
                        no_pile_no_wf += 1
        report["readout_coverage"][kind] = {
            "n_readout_tokens": n,
            "unmeasurable_by_pile": round(no_pile / n, 4),
            "of_those_rescued_by_wordfreq": round(no_pile_but_wf / max(no_pile, 1), 4),
            "still_unmeasurable_by_either": round(no_pile_no_wf / n, 4),
        }

    # --- validation on Latin tokens only, where the Pile is a valid yardstick,
    # trimming the noisy tail where the pile count is itself near-noise ---
    latin = groups["latin_clean"]
    trimmed = {}
    for floor in (1, 10, 100, 1000):
        m = latin[(counts[latin] >= floor) & wf_seen[latin]]
        trimmed[f"pile_count_ge_{floor}"] = {
            "n": int(m.size),
            "spearman": round(float(spearmanr(wf[m], counts[m]).statistic), 4),
        }
    report["validation_latin_trimmed"] = {
        "why": "the Pile is a valid frequency yardstick for English but not for "
               "Chinese/Russian/Arabic; and a pile count of 1-2 is noise, not a "
               "frequency. Restricting to Latin tokens with real counts is the "
               "only place the two sources SHOULD agree.",
        "by_count_floor": trimmed,
    }

    # face validity, printed and stored for a human to judge
    nl_ids = groups["non_latin"]
    top_nl = nl_ids[np.argsort(-wf[nl_ids])][:20]
    report["face_validity_top20_non_latin"] = [
        [tokenizer.decode([int(i)]), round(float(wf[i]), 2)] for i in top_nl
    ]

    nl = report["coverage"]["non_latin"]
    rescued = report["readout_coverage"]["J"]["of_those_rescued_by_wordfreq"]
    report["verdict"] = {
        "non_latin_coverage_pile": nl["frac_seen_pile"],
        "non_latin_coverage_wordfreq": nl["frac_seen_wordfreq"],
        "non_latin_coverage_combined": nl["frac_seen_either"],
        "share_of_pile_blind_J_readout_tokens_rescued": rescued,
        "J_readout_tokens_still_unmeasurable": report["readout_coverage"]["J"][
            "still_unmeasurable_by_either"
        ],
        "latin_agreement_on_real_counts": trimmed["pile_count_ge_100"]["spearman"],
        "rejected_test": (
            "agreement with pile counts on NON-LATIN tokens (rho 0.245). Close "
            "to circular -- it asks whether a multilingual source agrees with an "
            "English-only source about non-English tokens. Recorded in "
            "validation_vs_pile, not used as a gate. See DECISIONS.md D27."
        ),
        "closes_the_coverage_gap": bool(
            nl["frac_seen_wordfreq"] > 0.5 and rescued > 0.8
        ),
        "is_the_same_measurement_as_pile": False,
        "how_to_use": (
            "Report as a SECOND, independent frequency proxy alongside pile "
            "counts -- not merged with them, and not a replacement. Its value is "
            "that it makes the non-Latin subpopulation testable at all, which "
            "neither other source can do."
        ),
    }

    # --- the substring flaw, sized rather than described ---
    has = wf > 0
    substring_share = float((has & ~bare).sum() / has.sum())
    readout_substring = {}
    for kind in ("J", "R", "logit"):
        tot = sus = 0
        for c in cells:
            if c["kind"] != kind:
                continue
            for t in c["top"]:
                tid = token_id_of(t["t"])
                if tid is None:
                    continue
                tot += 1
                sus += bool(has[tid] and not bare[tid])
        readout_substring[kind] = round(sus / tot, 4)
    report["substring_flaw"] = {
        "definition": "wordfreq strips punctuation/digits and scores the remaining "
                      "word, so for non-bare tokens the score describes a substring "
                      "('.Scene' -> 'scene', '.cpu' -> 'cpu') and over-states the token",
        "share_of_scored_vocabulary": round(substring_share, 4),
        "n_scored": int(has.sum()),
        "n_bare_trustworthy": int((has & bare).sum()),
        "n_substring_inflated": int((has & ~bare).sum()),
        "share_of_readout_tokens_affected": readout_substring,
        "mitigation": "is_bare_word is stored per token in the .npz; step 4c reports "
                      "with and without, or excludes (D33)",
    }
    print(f"SUBSTRING FLAW: {substring_share:.3f} of scored tokens are non-bare "
          f"(score is for an embedded word, not the token)")
    print(f"   readout tokens affected: " +
          ", ".join(f"{k} {v:.3f}" for k, v in readout_substring.items()))

    print("\nCOVERAGE (share of each group with a frequency number):")
    print(f"{'group':>14} {'pile':>7} {'wordfreq':>9} {'either':>8} {'neither':>8}")
    for name, v in report["coverage"].items():
        print(f"{name:>14} {v['frac_seen_pile']:>7.3f} {v['frac_seen_wordfreq']:>9.3f} "
              f"{v['frac_seen_either']:>8.3f} {v['neither']:>8.3f}")
    print("\nVALIDATION -- Spearman(wordfreq, pile count) where both can see:")
    for name, v in report["validation_vs_pile"].items():
        print(f"   {name:>14}: rho={v['spearman']} (n={v['n']})")
    print("\nREADOUT COVERAGE -- of tokens pile cannot measure, how many does wordfreq rescue?")
    for kind, v in report["readout_coverage"].items():
        print(f"   {kind:>5}: {v['unmeasurable_by_pile']:.3f} pile-blind -> "
              f"{v['of_those_rescued_by_wordfreq']:.3f} rescued -> "
              f"{v['still_unmeasurable_by_either']:.3f} of all readout tokens still blind")
    print("\nFACE VALIDITY -- top non-Latin tokens by wordfreq (should be common function words):")
    print("  ", [t for t, _ in report["face_validity_top20_non_latin"][:14]])
    print("\nLATIN-ONLY AGREEMENT, noisy tail trimmed (the only valid comparison):")
    for k, v in trimmed.items():
        print(f"   {k:>18}: rho={v['spearman']:+.3f}  n={v['n']}")
    print(f"\nVERDICT closes_the_coverage_gap = "
          f"{report['verdict']['closes_the_coverage_gap']}  |  "
          f"same measurement as pile = {report['verdict']['is_the_same_measurement_as_pile']}")

    np.savez_compressed(VECTOR_PATH, wordfreq_zipf=wf, is_bare_word=bare)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {VECTOR_PATH} and {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
