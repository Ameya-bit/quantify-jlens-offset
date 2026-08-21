# Decisions ledger

Every methodological choice that a reader could reasonably have made
differently, with the reason, the alternative that was rejected, and what
would reverse it. Kept so the writeup (steps.md Step 9) can be assembled
from a record rather than from memory, and so a reviewer can attack the
choices directly instead of reverse-engineering them from code.

Rules: one row per decision, added *when the decision is made*, not
retroactively. "Reverses if" must name a concrete observation, not a mood.
Decisions that were later overturned stay in the table, struck through, with
a pointer to the row that replaced them.

---

## Step 0–1 — instrument

### D1. Read out in fp32 even though the model runs bf16
**Date:** 19 Aug · **Where:** `src/lens.py` · **Evidence:** devlog 0.0.1

The norm→unembed path is computed in fp32 while the model itself is bf16.
The offset we are hunting is small and MPS bf16 rounding could eat it.
Costs ~2.5 GB for an fp32 copy of `W_U`.

- **Rejected:** the official `HFLensModel.unembed`, which casts to model dtype.
- **Deliberate deviation from official tooling** — flagged as such in the module docstring.
- **Reverses if:** the fp32/bf16 gap turns out to be below the noise floor of `m_t` in Step 4, at which point the memory is better spent elsewhere.

### D2. Late-agreement gate set at ≥5/10, before seeing the data
**Date:** 19 Aug · **Where:** `src/sanity.py` (`MIN_MEAN_OVERLAP_28_29`) · **Evidence:** `results/step1_sanity.json`

Pre-registered so a marginal result could not be rationalised afterwards.
Re-checked during the 20 Aug red team: the constant predates the numbers.

- **Reverses if:** never — a pre-registered gate that moves is not a gate. A different threshold would need a new, separately named check.

### D3. Pythia-1.4B as the Route A (LayerNorm) arm
**Date:** 19 Aug · **Where:** `src/fit_pythia.py` · **Evidence:** steps.md Step 0

Route A tests whether the final-norm bias `β` explains the offset. That
requires a model whose final norm *has* a bias. All 8 released lenses are
RMSNorm-family (no bias), so we fit our own lens on a LayerNorm model.

- **Rejected:** GPT-2 as the lens arm — kept instead as a weights-only cell (`W_U·β` vs `log f_t`), no lens, no forward passes, because fitting a second lens does not fit in the clock.
- **Reverses if:** a LayerNorm model appears in the released lens collection.

### D4. Pythia fit uses n=25 texts, not the planned n=10
**Date:** 20 Aug · **Where:** `src/fit_pythia.py` · **Evidence:** `results/step2_pythia_smoke.json`

Smoke test measured 101 s/text, so 25 texts costs ~45 min — affordable, and
25 is what the released Qwen lens used (`provenance: n_prompts=25`).
Matching the published recipe beats saving 25 minutes.

- **Reverses if:** nothing; the fit landed in 43.8 min as projected.

### D5. Pythia's sanity battery has no anchor-identity check
**Date:** 20 Aug · **Where:** `src/sanity_pythia.py` · **Evidence:** `results/pythia_sanity.json`

Qwen's lens stores its destination layer with `J₃₀ = I`, which gives a free
"is it wired up right" check. Our Pythia artifact stores layers 0–21 with
destination 22, so there is no stored identity matrix to compare against.
The wiring check (logit-lens on the final block must reproduce the model's
true logits exactly) plays that role instead.

- **Disclosure debt:** "3/3 PASS" must not be read as "same three checks as step 1." Stated in the module docstring and devlog 0.0.2 addendum 2.
- **Reverses if:** we ever re-fit Pythia with the destination layer included.

---

## Step 2 — the junk survey

### D6. Survey excludes pile-10k rows 0–24
**Date:** 20 Aug · **Where:** `src/junk_survey.py` · **Evidence:** lens provenance dict

Reading a lens out on its own fit texts would be circular. Rows 0–24 are
verifiably our Pythia fit's corpus (`texts[:25]`).

- **Weaker than it first looked (red team, 20 Aug):** the *released Qwen* lens's provenance records `n_prompts=25, docs_consumed=25, dataset=pile-10k` but **no row indices** (`git_commit: "modal"`). So for that lens the exclusion is insurance, not proof. Residual risk that a surveyed row is in its fit set ≈6%.
- **Rejected:** re-doing the survey — the exclusion is free and the residual risk is small.
- **Reverses if:** the authors publish row indices showing a different slice.

### D7. Junk is measured by a surface flag rule, not by judgement
**Date:** 20 Aug · **Where:** `src/flags.py` · **Evidence:** `results/step2_junk_composition.json`

A mechanical rule applied identically to every layer and instrument, so
depth trends are comparable and nothing depends on which tokens we happened
to find striking.

- **Known blind spot, stated up front:** obscene-but-well-formed English words and Latin word-fragments (`oooo`, `fictiona`) are not flagged. The rule therefore *undercounts* the J-lens's characteristic junk style.
- **Consequence:** existence claims survive an undercounting detector; **instrument rankings do not** and are not made from it (see D9).
- **Reverses if:** Step 3's Δ_t asset lands, which gives a principled obscenity axis the proxy currently lacks.

### D8. ~~Junk = punctuation OR byte-fragment OR non-Latin~~ → **overturned by D10**
**Date:** 20 Aug (morning) · superseded 20 Aug (evening)

### D9. Lens-vs-lens junk comparison deferred to behavioural `m_t`
**Date:** 20 Aug · **Evidence:** devlog 0.0.2 addendum 1

The initial "logit-lens is junkier than J/R" reading was retracted: the flag
rule catches the logit lens's junk style (non-Latin) and misses the J-lens's
(Latin fragments), so it cannot rank instruments in either direction.

- **Rejected:** patching the proxy to catch Latin fragments — no non-arbitrary rule exists, and Step 4's `m_t` answers the question behaviourally anyway.
- **Note:** the *proxy-free* agreement metric (D12) does support a J/R advantage at mid-depth. That is a separate claim on a separate ruler and is stated as such.

---

## Step 2 red team — 20 Aug, before opening Step 3

### D10. Punctuation is recorded but NOT counted as junk
**Date:** 20 Aug · **Where:** `src/flags.py` · **Evidence:** devlog 0.0.2 addendum 4 · **Replaces D8**

Punctuation is a *correct* prediction, not an instrument failure. Validated
three ways: 15.04% of the 3,125 tokens in the surveyed texts; 15.00% of the
true next tokens at the surveyed positions themselves; 18.6% across 500
random pile rows. Four of the corpus's ten commonest tokens are punctuation,
so a perfect unigram predictor scored 0.4 under D8.

Worse than merely wrong: punctuation *rises* with depth exactly where the
anomaly rate falls (J, L10-13 → L16-19: punct 0.033 → 0.100, t = +4.02;
anomaly 0.112 → 0.075, t = −3.49; paired over 25 texts). Summing them
manufactured a mid-depth plateau that does not exist.

- **Reverses if:** we ever want "how surprising is this readout" rather than "could this readout be correct" — a different question that punctuation belongs in.

### D11. Fixed denominator of 10; no "concept-slot" scoping
**Date:** 20 Aug · **Where:** `src/flags.py` · **Evidence:** devlog 0.0.2 addendum 4

Junk rate is `impossible tokens / 10`, always.

- **Rejected:** scoping to concepts by dividing by `10 − n_punctuation`. The denominator would shrink precisely where punctuation peaks (L16-19), re-inflating those layers by ~12% and smuggling back the artifact D10 removed. A moving denominator also breaks comparability across layers, instruments and baselines.
- **Side benefit:** counting punctuation as clean makes every reported junk number a conservative **lower bound**.
- **Reverses if:** a future metric needs per-slot weighting, in which case it gets its own name rather than redefining this one.

### D12. Baselines are mandatory on the junk figure
**Date:** 20 Aug · **Where:** `src/null_baselines.py` · **Evidence:** `results/step2_baselines.json`

"30% junk" is meaningless without a yardstick. Four are drawn: uniform over
the vocabulary (0.432 — Qwen's vocabulary is 43% non-Latin), random-rotation
null (0.386), real text (0.000 matched / 0.005 corpus-wide), layer-30 floor
(0.022).

This splits one question into two with **opposite answers**: *is the lens
just noise?* — no, it runs 3–6× below the null at every depth. *Does it read
out tokens that cannot be right?* — yes, ~20% early against ~0%. Step 2's
gate is the second question; without the baselines a reader answers the
first and dismisses the result.

- **Reverses if:** nothing. This is strictly more information than the figure had before.

### D13. The null is a random ROTATION, 5 seeds, strided layers
**Date:** 20 Aug · **Where:** `src/null_baselines.py`

- **Rotation, not an arbitrary Gaussian matrix:** a rotation preserves the activation's length and destroys only its alignment with `W_U`, which isolates the thing we mean by "not a lens." (Final norm would remove a scale change anyway, so this matters less than it sounds — but rotation is the honest default.)
- **5 seeds, not 1:** seed means came out 0.360 / 0.397 / 0.377 / 0.408 / 0.387. One draw could be off by 0.02, which is enough to argue about.
- **Strided layers (0,5,…,30), not all 31:** the null is flat with depth by construction (0.384–0.390 across the strided set), so a dense sweep buys nothing and costs 4× the time.
- **Rejected for now:** the shuffled-J variant (permute the real matrix, preserving its scale and spectrum). It is a tighter null and is still on the Step 8 list; the rotation was chosen today because it is what Step 8 pre-registered and it is simpler.
- **Reverses if:** Step 8 finds the two nulls disagree.

### D14. `step2_overlap_L30.png` replaced by `step2_lens_agreement.png`
**Date:** 20 Aug · **Where:** `src/step2_figures.py`

Overlap with the destination layer is not lens-vs-lens agreement — two
lenses can each share 6 tokens with layer 30 and share none with each other
— yet the old figure was titled as replicating the paper's agreement claim.
Its "all three in lockstep" reading was also wrong (L18: J 0.99 / R 1.12 /
logit 0.42).

- **Rejected:** keeping both figures. The old one is a subset of the new one's right-hand panel, correctly labelled.
- **Reverses if:** nothing; the old file is recoverable from git history.

### D15. The late anomaly rebound is banked for Step 6, not chased now
**Date:** 20 Aug · **Evidence:** devlog 0.0.2 addendum 4, steps.md Step 6

J and R read out *more* impossible tokens at L24-27 than at L18-21
(J 0.069 → 0.109, t = +5.01, 22/25 texts; R 0.065 → 0.094, t = +4.05) while
the logit lens moves monotonically the other way (0.369 → 0.216, t = −10.90,
0/25 texts). A rebound present only in the *translated* instruments is a
transport-error signature — which is H3, i.e. Step 6's job.

- **Rejected:** analysing it now. Time budget; Step 6 has the right battery for it and the finding is timestamped here so it does not get lost or re-discovered as if new.
- **Reverses if:** Step 4's `m_t` shows the same rebound, in which case it is about the offset rather than transport and moves forward.

### D16. Weak-lever caveat for H3 weakened; qwen3.5-9b download deferred
**Date:** 20 Aug · **Evidence:** `results/step2_lens_agreement.png`, steps.md Step 6

The Step 6 caveat assumed "J ≈ R on the 4B" makes the J→R swap
underdetermined. True only near the destination: pairwise top-10 overlap is
**0.35/10 at L0**, 4.72 at L12, 6.64 at L20, and ≥8.7 only from L27. At the
early/mid layers where the junk lives they are substantially different
instruments, so the lever is stronger than assumed.

- **Rejected for now:** downloading qwen3.5-9b to strengthen the lever. Decision stays where steps.md put it — after Step 4, on remaining time.
- **Reverses if:** Step 4's `m_t` shows J and R offsets tracking each other closely at *all* depths, restoring the original concern.

---

## Step 3 — staging assets

### D17. Frequency counts: `capped` (8192 tokens/doc) is primary, `full` retained
**Date:** 20 Aug · **Where:** `src/frequencies.py` · **Evidence:** `results/step3_frequencies.json`

One pile-10k document is 981k tokens — **6.3%** of the 15.6M-token corpus by
itself. A unigram prior a single document can move by 6% is a sampling
accident, not a fact about English. 339 documents exceed the cap.

- **The choice is not cosmetic:** capped and full agree at Pearson 0.977 in log space, not 0.99+.
- **Rejected:** capping at 128 tokens (matching the lens-fit regime) — that throws away 98% of the corpus for no gain, since frequency is a global property and does not need to match the readout regime.
- **Reverses if:** Step 4c's `m_t`~`log f_t` result differs between the two variants, in which case neither is reportable alone and both go in the figure.

### D18. Step 4c regresses on tokens with nonzero count; exposure is reported, not hidden
**Date:** 20 Aug · **Where:** `src/frequencies.py` · **Evidence:** `results/step3_frequencies.json`

Only **42.8%** of Qwen's 248k vocabulary appears in pile-10k at all, and only
**9.3%** of its non-Latin tokens — pile-10k is English, Qwen's vocabulary is
not. Tokens with zero count have no `log f_t`, so the regression must drop
them.

- **Why this is survivable, measured rather than assumed:** the *vocabulary* figure is not the *regression* figure. Of tokens that actually appear in step-2 readouts, only 11.1% (J-lens) have no frequency estimate — 21.7% at layers 0-5, 6.0% mid-depth. The unmeasurable tail of the vocabulary is mostly never read out either. Worst case is the logit lens at L14-21: 41.0%.
- **Rejected:** add-one smoothing. It hands ~142k unseen tokens an identical pseudo-count, i.e. zero variance, which does not rescue the regression — it disguises the gap.
- **Rejected (no budget):** counting on a larger or multilingual corpus.
- **Disclosure debt:** every Step 4c number states the share of readout mass it was computed on.
- **Reverses if:** the dropped tokens turn out to carry the effect — testable via D19.

### D19. ~~Token id kept as a full-vocabulary frequency rank, for robustness only~~ → **scope cut by D26**
**Date:** 20 Aug · **Where:** `src/frequencies.py`

Qwen's BPE ids run roughly in merge order, so a low id means a frequent
token. Spearman against pile counts on the seen tokens is **−0.677**, and the
seen-rate falls 98.2% → 6.0% across id bands. It covers 100% of the
vocabulary and derives from Qwen's *own* training corpus rather than from the
Pile — which pile-10k does not, since Qwen was not trained on the Pile.

- **Explicitly not the primary regressor:** it is a rank, not a count, and it is non-monotone across vocabulary blocks (the 100k-150k band is 6.0% seen while 150k-200k is 23.7%, so blocks were added wholesale).
- **Its job:** check that Step 4c's H1 result is not an artefact of dropping the tokens D18 cannot measure.
- **Reverses if:** the two proxies disagree about H1, which would itself be the finding.

### D20. Two-hop set built before Δ_t, against steps.md order
**Date:** 20 Aug · **Evidence:** steps.md Steps 5 and 7

steps.md lists Δ_t before the two-hop set. Reversed on priority: Δ_t feeds
Step 5, which steps.md marks **SKIPPABLE**; the two-hop set feeds Step 7,
which it marks **PROTECT ≥90 MIN**. If the clock runs out, losing Δ_t costs
a step already designated as cuttable.

- **Reverses if:** Step 4 finishes early enough that both fit comfortably, in which case order stops mattering.

### D21. Two-hop items keep a shortcut flag rather than being pruned
**Date:** 20 Aug · **Where:** `src/twohop.py` · **Evidence:** `results/step3_twohop.json`

138 of 164 candidates survive the model filter (kept iff Qwen3.5-4B's true
final top-1 next token is the answer). But an item is only *two-hop* if the
answer depends on the entity, so every template is also run **blinded** —
entity removed, nothing else changed.

| template | kept | shortcut (blinded top-1) | shortcut (blinded top-5) | strict two-hop |
|---|---|---|---|---|
| capital | 39 | 0 | 0 | **39** |
| capital_of | 33 | 0 | 0 | **33** |
| language | 39 | 5 | 19 | 20 |
| currency | 27 | 0 | 20 | 7 |
| **total** | **138** | 5 | 39 | **99** |

- **Both strictness levels recorded, neither chosen here.** By blinded top-1 the currency template looks clean — but only because its blinded top-1 is a markdown artefact (`" **"`); `" euro"` sits at rank 2 and 70% of currency answers are `" euro"`. The lenient flag would have hidden that.
- **Rejected:** deleting the contaminated items. Step 7 can drop them in one line; a deleted item cannot be recovered by a reader who disagrees with the criterion.
- **Recommendation carried to Step 7:** headline on `capital` + `capital_of` (72 items, clean by both criteria); use the rest as a secondary panel.
- **Reverses if:** Step 7 needs more than 72 items for statistical power, in which case the strict-clean 99 are the next tier.

### D22. The model filter also silently fixes our labels — and can mask a shared error
**Date:** 20 Aug · **Where:** `src/twohop.py`

Keeping only items the model answers correctly removes rows where *our* label
was wrong: the rejects are things like "Shanghai → Chinese" (model says
Mandarin), "Mumbai → Delhi" (model says New Delhi), "Toronto → dollar" (model
says Canadian). That is the filter working.

- **The failure mode it cannot catch:** a label wrong in the same way the model is wrong survives. Mitigated at table-construction time by excluding genuinely ambiguous cases — multi-capital countries (South Africa), contested "country" (Scotland vs UK), and multilingual states in the language template (Belgium, Switzerland, Ireland, Canada).
- **Reverses if:** Step 7 shows anomalous behaviour concentrated in a few items, which is the signal to hand-audit those labels.

### D23. Δ_t carries a frequency confound; the "frequency-matched control" in Step 5 is mandatory, not a refinement
**Date:** 20 Aug · **Where:** `src/delta_t.py` · **Evidence:** `results/step3_delta_t.json`

Measured, not assumed: **Spearman(log f_t, Δ_t) = +0.354** over the 106k
tokens pile-10k can measure, with median Δ rising monotonically across
frequency quintiles (−0.047, −0.033, +0.001, +0.045, +0.107).

Cause: the instruct model moves probability mass onto chat and formatting
tokens (the most-promoted tokens are all whitespace variants and `<|im_end|>`),
so ordinary content tokens are deflated roughly in proportion to the mass
they held. **Common words look "suppressed" with no suppression involved.**

- **Consequence:** Δ_t must never be read raw, and any H2 claim built on raw Δ_t is measuring frequency. steps.md Step 5 already required frequency-matched controls; this promotes that from good practice to load-bearing.
- **Reverses if:** nothing — the correlation is in the artifact and reproducible.

### D24. The Δ_t validation gate was rewritten after it failed for the wrong reason
**Date:** 20 Aug · **Where:** `src/delta_t.py`

The first gate compared profanity's Δ percentile against neutral words'
and **failed** (0.982 vs 0.953). The gate was wrong, not the asset: the
probe sets were not frequency-matched — profanity median log f 3.37, neutral
median log f 6.43, so the controls were ~20× more frequent and carried a
larger frequency-driven Δ by D23. Comparing them was never valid.

Replaced with a frequency-matched gate: each probe is compared to the median
Δ of clean, non-special, seen tokens within ±0.25 log-count (pool of 92,846).
**Result: profanity excess +0.226 vs neutral excess +0.052 — a 4.4×
separation. PASS.**

- **This is a gate that moved after seeing data, which D2 says is not allowed.** The distinction: D2's gate was on a *result* (does the lens transport meaning); this one is on an *instrument's validity*, and the original comparison was arithmetically invalid rather than merely unfavourable. Recorded here in full, with the failed numbers, so a reader can judge that for themselves rather than take it on trust.
- **Honest residual:** 3 of 12 profanity probes are not elevated (` slut` −0.117, ` retard` +0.019, ` whore` +0.009). They are among the rarest probes; whether that is weak representation or genuinely absent suppression is not resolved here.
- **Reverses if:** Step 5's enrichment result depends on which probe list is used, which would mean the axis is probe-specific rather than general.

### D25. Only the difference vector was needed, but all three are stored
**Date:** 20 Aug · **Where:** `src/delta_t.py`

`results/step3_delta_t.npz` carries `delta_t`, `mean_logprob_base` and
`mean_logprob_instruct` (2.4 MB). Storing the two components costs nothing at
save time and avoids a ~10-minute two-model re-run if Step 5 wants a
rank-based or renormalised variant of Δ instead of the raw difference.

### D26. Merge-rank frequency built and measured; it does NOT close the non-Latin gap
**Date:** 21 Aug · **Where:** `src/zipf_frequency.py` · **Evidence:** `results/step3_zipf_frequency.json`

Qwen ships 247,587 ordered BPE merges, so its own training-corpus frequency
ordering is directly readable — established method (arXiv:2407.16607,
NeurIPS 2024; arXiv:2508.17771 does exactly this for Chinese tokens). Built
it. **No Zipf calibration**, deliberately: frequency ∝ 1/rank^s means
−log(rank) is already a log-frequency proxy and `s` only rescales the slope,
which cannot change a Spearman ρ, an R², or a p-value. Calibrating `s` would
require fitting against pile-10k counts — reimporting the corpus bias the
asset exists to escape.

**Then measured whether it works where we need it, rather than deferring
that to the writeup. It does not.**

| region | ρ vs pile count |
|---|---|
| overall | **−0.676** |
| ids 0–10k | −0.377 |
| ids 10k–50k | −0.322 |
| ids 50k–100k | −0.202 |
| ids 100k–150k | **+0.041** |
| ids 150k–200k | **+0.012** |
| ids 200k–248k | −0.038 |
| Latin tokens | −0.650 |
| **non-Latin tokens** | **−0.260** (and measured on only the 9.4% the Pile can see) |

The global −0.676 is carried almost entirely by *between*-band differences.
Within a band it collapses, and past id 100k it is indistinguishable from
zero. Some collapse is range restriction and expected; the sign flip is not.
Merge rank sorts tokens into coarse frequency tiers and carries little
information within a tier — none at all where Qwen's non-Latin vocabulary
lives.

- **Kept anyway, with its role cut:** a secondary whole-vocabulary robustness check on H1, explicitly **not** a way to rank non-Latin tokens against each other.
- **The non-Latin frequency gap from D18 is therefore still open.** Named candidate fixes, not attempted: `wordfreq` (the only surveyed option with real CJK coverage) and the infini-gram count API (5T tokens, but English-dominant corpora, so it may not close CJK either).
- **Free finding:** merge rank and raw token id agree to four decimal places against pile counts (−0.6762 vs −0.6766). Qwen's ids *are* merge order, so the principled construction bought nothing over D19's crude proxy — D19 was already measuring the same quantity.
- **Reverses if:** a CJK-covering frequency source lands, at which point the non-Latin regression becomes possible and this asset is superseded rather than merely limited.

### D27. `wordfreq` added as a second, independent frequency proxy — coverage gap closed, provenance gap not
**Date:** 21 Aug · **Where:** `src/multilingual_freq.py` · **Evidence:** `results/step3_multilingual_freq.json`

The non-Latin gap left open by D18 and D26 is now covered:

| group | pile-10k | wordfreq | either |
|---|---|---|---|
| Latin, clean | 0.683 | 0.953 | **0.967** |
| **non-Latin** | **0.094** | **0.882** | **0.887** |
| punctuation | 0.685 | 0.084 | 0.722 |

Operationally, of the J-lens readout tokens pile-10k cannot measure, wordfreq
rescues **93.2%** — leaving **0.8%** of all J-lens readout tokens with no
frequency from any source, down from 11.2%. **H1 can now be tested on the
non-Latin subpopulation for the first time.**

- **Complementary, not redundant:** wordfreq is word-level, so it scores 0 on Latin subword fragments (`correcti`) where pile-10k is strong; pile-10k is English-only, where wordfreq is strong. Neither covers the vocabulary alone. Punctuation is covered by pile-10k only, which is fine — punctuation is not junk (D10).
- **NOT merged with pile counts, and not a replacement.** Two separate proxies, reported separately. Agreement between them is moderate even where it should be highest (Latin, pile count ≥1000: ρ = +0.52; ≥100: +0.39; ≥10: +0.29), which is what two genuinely different corpora look like.
- **Provenance gap unchanged:** wordfreq's corpora (Wikipedia, Books, Reddit, Twitter, OpenSubtitles, SUBTLEX, Leeds) are not Qwen's either. This fixes coverage, not provenance. The only artefact we hold that reflects Qwen's own corpus is merge rank, and D26 showed it carries no information in this region. **So for non-Latin tokens there is still no Qwen-provenance frequency measure, and there will not be one in this project.**
- Frozen ~2021 by its author because generative-AI text polluted the web — for a prior over *human* text, a feature.

### D28. The `closes_the_gap` gate was rewritten after it failed — second time, and the same mistake
**Date:** 21 Aug · **Where:** `src/multilingual_freq.py`

The first gate required wordfreq to agree with pile-10k counts on non-Latin
tokens (ρ = 0.245) and **failed**. The gate was close to circular: it asks
whether a multilingual source agrees with an English-only source about
non-English tokens — and if it did, the asset would be adding nothing. The
disagreement is the asset working. `的` appears 86 times in pile-10k and
scores Zipf 7.79 in wordfreq, because it is the commonest character in
Chinese and the Pile is not Chinese. Same for ` привет` (pile 1, Zipf 5.13)
and `您好` (pile 0, Zipf 3.93).

Replaced with three tests that are not circular: face validity on non-Latin
(top scorers come out as `的`, `の`, `в`, `के`, `في`, `है`, `に` — the
highest-frequency function words of their languages); agreement with pile
counts on **Latin** tokens only, where the Pile is a valid yardstick, with
the noisy tail trimmed; and coverage of the tokens that actually appear in
readouts.

- **This is the same error as D24, made twice in two days: a validation gate built against a reference that is wrong in the region being tested.** Both times the asset was fine and the gate was not. The pattern to watch: before writing a gate, ask what the reference measures *in the specific region the asset targets*, not on average.
- **Both failed gates and their numbers are recorded** here and in the JSON's `rejected_test` field rather than replaced silently. A reviewer who thinks the original gate was the right one can see exactly what it said.
- **Reverses if:** the reviewer judges that changing a failed gate twice is not acceptable practice regardless of reasoning, in which case both assets ship with their original FAIL verdicts stated and the interpretation left to the reader.

### D29. Whether to frequency-residualise Δ_t is decided from results, by a rule fixed in advance
**Date:** 21 Aug · **Where:** steps.md Step 5 · **Evidence:** D23

Δ_t is frequency-contaminated (ρ = +0.354). Ameya's call: compute the H2
statistic both ways and look before deciding, rather than committing now.
Agreed — but the *rule* is fixed now so the later choice is not a judgement
call made after seeing which answer is nicer:

- **Both are reported, always.**
- Agree in sign and significance → raw Δ_t is the headline, residualised is the robustness line.
- **Disagree → the residualised result is primary and the disagreement is the finding**, at full prominence.
- Residualising uses the **matched-control** form already built in `src/delta_t.py`, not a linear subtraction: the Δ-vs-frequency relationship is convex (quintile steps +0.014, +0.034, +0.044, +0.062), so a straight line leaves structure in the residual that would read as suppression.
- Caveat stated either way: if tuning inhibits tokens *because* they are rare, residualising removes real signal too (controlling for a mediator). That makes it conservative — it can understate H2, never manufacture it.
- **Skip condition:** if Step 4.0's noise null fails, none of this runs.

### D30. H2 renamed from "RLHF suppression" to "tuning-induced suppression"
**Date:** 21 Aug · **Where:** steps.md Step 5, `notes/exec_sum.md`, `src/delta_t.py` · **Evidence:** `results/step3_delta_t.json`

Ranked by Δ_t the vocabulary's top is not obscenity — it is informal,
archaic and misspelled English plus scraped-text artefacts, and the promoted
end is whitespace and chat formatting. Profanity is genuinely elevated (top
0.3–4%, 4.4× its frequency-matched controls) but is not the main axis.

- **Ameya's judgement, recorded as his:** the generalised claim is still worth making — these are demonstrably inhibited tokens that the J-lens surfaces as top contenders, and the claim no longer depends on a contested account of what RLHF does internally.
- **The cost, which he accepted:** the general version sits *closer* to H1, because "informal / misspelled / web-debris" is nearly a synonym for "rare". The frequency-matched control therefore carries more weight in the general version than it would have in the narrow one.
- **Second limit:** the base model's pretraining data was itself filtered, so Δ_t sees only tuning-stage suppression, not suppression from data filtering.
- **Successor experiment for the writeup:** finetune a model to suppress a chosen target, rerun the battery. Ground truth by construction. His proposal.
- **Reverses if:** Step 5 finds the enrichment is carried entirely by the obscenity subset, in which case the narrow claim is back on the table with evidence.

### D31. H1 is reported as a range across three cells, never as one number
**Date:** 21 Aug · **Where:** steps.md Step 4c · **Evidence:** devlog 0.0.3

Sufficiency was measured before committing to Step 4. Coverage is solved
(92.8% of the vocabulary has a frequency from some source, up from 42.8%) and
**power is not the constraint** — with 10k–130k tokens per group, |ρ| ≥ 0.01
is detectable at p < 0.01. The constraint is **validity**: the two usable
sources agree at only ρ ≈ 0.24, so H1's magnitude depends on which one is
used.

Ameya's call: run the three separately and report a range.

| cell | source |
|---|---|
| Latin tokens | pile counts (capped) |
| whole vocabulary | wordfreq |
| non-Latin only | wordfreq |

- **Agree in sign ⇒ H1 robust**, and stated with more confidence than a single number would have earned.
- **Disagree ⇒ that is the finding**, reported at full prominence, not buried: "whether the offset tracks frequency depends on which frequency you mean" is a real result.
- **Rejected:** picking one source and calling it *the* frequency. Two proxies agreeing at 0.24 cannot both be that, and choosing after seeing which gives the nicer r is exactly the failure mode D24/D28 are about.
- **Reverses if:** a single source with both Qwen provenance and full coverage appears — none exists today.

### D32. Merge rank dropped from the working set, kept as a negative result
**Date:** 21 Aug · **Where:** `src/zipf_frequency.py`

It measures the same thing as raw token id (−0.6762 vs −0.6766 against pile
counts) and has no signal past id 100k, where the non-Latin vocabulary lives.
It therefore adds nothing Step 4 can use.

- **Not deleted:** "we tried reading Qwen's own tokenizer and here is exactly how far it gets you" is a real paragraph for the writeup, and the negative result is reusable by anyone else attempting it.
- **Reverses if:** a model whose ids are pure merge order throughout makes the method work end-to-end.

### D33. `is_bare_word` stored so Step 4c can quarantine wordfreq's substring scores
**Date:** 21 Aug · **Where:** `src/multilingual_freq.py` · **Evidence:** `results/step3_multilingual_freq.json`

wordfreq strips punctuation and digits and scores the word left behind, so
`.Scene` is scored as "scene", `.cpu` as "cpu", `_exchange` as "exchange" —
an over-estimate of the token's own frequency. Sized rather than described:
**8.9%** of scored tokens, **4.5%** of J-lens readout tokens (R 5.9%, logit
5.9%).

- **Flag stored per token in the .npz**, not applied here — Step 4c excludes or dual-reports, and the choice stays visible.
- **Reverses if:** a token-level (rather than word-level) multilingual frequency source appears, which would remove the flaw at source.
