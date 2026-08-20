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
