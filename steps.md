# Step-by-Step Execution Plan — J-Lens Offset Study

*Written 19 Aug 2026. Companion to `mats-application-plan-v2.md` (strategy/rationale lives there; this file is the checklist). Deadline: Fri 4 Sep, 11:59pm PT.*

**Models:** Qwen3.5-4B (+Base) = headline. **Pythia-1.4B** = Route A arm (LayerNorm + β; lens fitted in-clock). GPT-2 = one weights-only cell (phoenix r≈0.67 replication), no lens, no forward passes.

**Bright lines (from v2):** no battery code pre-clock; log hours honestly; every agent-produced number gets an independent sanity check; when a result surprises, next hour goes to breaking it; gates are hard.

---

## Step 0 — Pre-clock (unclocked; generic setup only) — DONE 19 Aug

- [x] Python env: `.venv` (Python 3.13, torch 2.13 + MPS, transformers 5.15.1, datasets, numpy, matplotlib, scipy). See `requirements.txt`. **HF login still pending (not needed — all assets public; only needed if Gemma stretch runs).**
- [x] Download models: `Qwen/Qwen3.5-4B`, `Qwen/Qwen3.5-4B-Base`, `EleutherAI/pythia-1.4b`, `openai-community/gpt2` (weights only). All verified public/ungated.
- [x] Download lenses: `qwen3.5-4b/{j,r}-lens/lens.pt` (775 MB). Integrity-checked: both load; `J` is a per-layer dict, layers 0–30, per-layer 2560×2560 **fp16** (→ fp32 readout rule matters); provenance confirms `target_layer=30`, `skip_first=4`, `n_prompts=25`.
- [x] **Released 8-model list checked: all RMSNorm-family (5×Qwen3.5/3.6, gemma-3-27b-it, deepseek-v4-flash). No LayerNorm model → in-clock Pythia-1.4B fit stands.**
- [x] Download `NeelNanda/pile-10k` (10,000 rows).
- [x] Official tooling installed: `anthropics/jacobian-lens` cloned to `~/mech_interp/jacobian-lens`, `pip install -e`, `jlens.fit` and `JacobianLens` importable.
- [x] Residual scoop check: **CLEAR.** Paper (transformer-circuits.pub/2026/workspace) has no frequency regression, no unigram prior, no offset characterization. No pivot needed.
- [x] Route A assumption verified from weights: Pythia-1.4B `final_layer_norm.bias` exists (‖β‖=2.64; 24 layers, d_model=2048); GPT-2 `ln_f.bias` exists (‖β‖=11.62). Both LayerNorm.
- [x] Repo-zero state: git history = LICENSE only; strategy files gitignored; tracked = LICENSE, `.gitignore`, `steps.md`, `requirements.txt`.

## Step 1 — Load + sanity-check the instrument (hour 0–1) — DONE 19 Aug, verified 20 Aug

*The lens turns a layer's activation into a score for every vocab token; if it loads wrong, everything downstream is garbage.*

- [x] `src/lens.py`: load model + `lens.pt`; `score(h, layer, kind)` for kind ∈ {logit, J, R} sharing all code. Readout `softmax(W_U · finalnorm(J_ℓ · h))`; **fp32 for norm→unembed on MPS** even if model is bf16.
- [x] `src/sanity.py`:
  - [x] Assert target-layer anchor row of `J` ≈ identity (artifact guarantees this). (max|J₃₀−I| = 0.0, same for R.)
  - [x] Late layers: J-lens ≈ logit-lens agreement (top-k overlap). (6–10/10 at layers 26–29; exactly 10/10 at layer 30.)
  - [x] Known-fact prompts: sensible mid-layer readouts (e.g. Eiffel→Paris by mid-depth). (Paris & Tokyo at layer 24; Au late at 28 → 2/3, gate met.)
- **Gate:** passes by 1h. **Fallback:** switch headline to qwen3.5-9b (next lensed dense model in repo). → **Gate PASSED**; no fallback needed. Evidence: `results/step1_sanity.json`, devlog `0.0.1`.
- Verified 20 Aug (independent re-check): sanity battery re-run bit-identical; conventions match official `jlens` (block-output residuals, `h·Jᵀ` transport, no softcap on Qwen3.5-4B); logit-lens on final block reproduces the model's true logits (top-10 set match; ≤0.06 bf16 rounding); raw ckpt fp16, layers 0–30, J₃₀=R₃₀=I exactly.

## Step 2 — Reproduce the public junk (hour 1–2.5) — Part A DONE 20 Aug; Pythia fit in progress

*Confirm the phenomena the hypotheses are supposed to explain actually appear on our setup.*

- [x] ~25 pile-10k prompts; per layer, top-10 readout tokens at several positions (skip first 4 — matches lens fit). (Rows 25–9999 only — rows 0–24 are the fit corpus of our Pythia fit (verifiably `texts[:25]`) and, *by inference*, of the released lens: its provenance gives `n_prompts=25, docs_consumed=25, dataset=pile-10k` but **no row indices**, so the exclusion is free insurance rather than a proof of non-circularity (residual risk of any overlap ≈6%; red team 20 Aug). 11,625 cells → `results/step2_readouts.json`; layer-30 J=R=logit identity holds per-cell; one cell re-checked by eye via `src.peek`, exact match.)
- [x] Setup figure: junk (obscene/CJK/punctuation/rare fragments) at early layers, fading with depth — J-lens vs R-lens vs logit-lens side by side. (`results/step2_junk_fraction.png` + grid figures; junk present in all three instruments. Initial "logit junkier than J/R" reading **retracted** — flag-proxy recorder bias (counts CJK, blind to Latin-fragment junk; see devlog 0.0.2 addendum + `results/step2_junk_composition.json`). Lens-vs-lens junk comparison deferred to behavioral m_t (Step 4/6).)
- [x] **Red-teamed 20 Aug before opening Step 3 (devlog 0.0.2 addendum 4).** Machinery survived (survey re-run bit-identical: 0/11,625 readout diffs; 300 cells re-derived from fresh forward passes; L30 identity 0 mismatches). Three claims did not. (1) **Punctuation removed from `is_junk`** — real pile text is 15.0% punctuation and a perfect unigram predictor scored 0.4 under the old rule; the two flags move in *opposite* directions mid-depth (J anomaly 0.112→0.075 t=−3.49 vs punct 0.033→0.100 t=+4.02), so the old "mid-depth plateau" was an artifact of summing them. Junk is now non-Latin|byte only (base rate 0/3125 on the surveyed texts, 0.005 across 500 random rows; punctuation 0.150 validated against the true next tokens at the surveyed positions, 0.1500). Punctuation plotted separately vs that 0.150 line. (2) **Baselines added** (`results/step2_baselines.json`, `src/null_baselines.py`): uniform-vocab 0.432, random-rotation null 0.386 (5 seeds, 0.318–0.470), text 0.000, L30 floor 0.022 — J/R run 3–6× below the null at every depth, so "the lens is just noise" is answered, while 20%-early-vs-0%-in-corpus keeps the gate. (3) `step2_overlap_L30.png` → **`step2_lens_agreement.png`**: the old figure measured distance-to-destination, not lens-vs-lens agreement, and its "lockstep" reading was wrong (L18: J 0.99 / R 1.12 / logit 0.42 — J/R ~2× the logit lens L12–L25, junk-proxy-free).
- [x] Background: Pythia-1.4B J-lens fit via official `jlens.fit` — **n raised 10→25** (smoke test: 101 s/text, `results/step2_pythia_smoke.json`; 25 matches released recipe). Launched, checkpointed. Precondition before Step 4 uses it: pass the Step-1-style sanity battery. → **PASSED 3/3** 20 Aug, re-verified at n=125 samples (`results/pythia_sanity.json`; facts: J-lens never later than logit baseline, once earlier. Qwen agreement retro-upgraded to n=125 from survey JSON, holds: `results/qwen_agreement.json`. Caveat: Pythia L21 agreement 5.0 vs Qwen L28/29 6.5–7.4 — logit-lens baseline itself unreliable on Pythia-family (tuned-lens lit.); disclosed in devlog 0.0.2).
- **Gate:** setup figures show the junk. → **Gate PASSED.** Evidence: devlog `0.0.2`.

## Step 3 — Stage assets (overlaps hour 1–2.5) — DONE 20 Aug

- [x] `log f_t`: token counts on pile-10k under Qwen tokenizer; separately under Pythia tokenizer (bonus: Pythia trained on the Pile — counts match its training distribution); separately under GPT-2 tokenizer (for the phoenix cell). (`src/frequencies.py` → `results/step3_frequencies.json` + `results/step3_token_counts.npz`. 15.6M/15.4M/17.4M tokens. **Two counts each: `capped` at 8192 tok/doc is primary, `full` retained** — one document is 6.3% of the corpus and the two agree at only r=0.977 in log space (D17). **Coverage warning: only 42.8% of Qwen's 248k vocab is ever seen, 9.3% of non-Latin** — but readout exposure is far smaller: 11.1% of J-lens readout tokens lack a frequency estimate (21.7% at L0-5, 6.0% mid; logit worst at 41.0% for L14-21). Step 4c reports this per number (D18). Full-vocab fallback rank: built properly from Qwen's 247,587 ordered BPE merges (`src/zipf_frequency.py` → `results/step3_zipf_frequency.json` + `step3_merge_rank.npz`), no Zipf calibration needed since −log(rank) is already a log-frequency proxy. **Measured, and it does not close the gap:** global ρ=−0.676 but that is between-band only — within-band ρ collapses to −0.38/−0.32/−0.20 and past id 100k is zero (+0.04/+0.01/−0.04); non-Latin ρ=−0.260. Usable as a whole-vocabulary robustness check on H1, **not** to rank non-Latin tokens against each other. The non-Latin *coverage* gap is now CLOSED by `src/multilingual_freq.py` (`wordfreq`, max Zipf over 12 languages → `results/step3_multilingual_freq.json` + `step3_wordfreq.npz`): non-Latin coverage 9.4% → **88.2%**, and of the J-lens readout tokens pile-10k cannot measure **93.2% are rescued**, leaving **0.8%** of all J-lens readout tokens unmeasurable. H1 is testable on the non-Latin subpopulation for the first time. Reported as a SECOND proxy alongside pile counts, never merged (Latin agreement ρ=+0.52 at count≥1000, +0.39 at ≥100 — two different corpora). **The provenance gap is NOT closed: wordfreq's corpora are not Qwen's either, and merge rank, the only Qwen-provenance artefact, is uninformative here (D26). For non-Latin tokens there will be no Qwen-provenance frequency measure in this project** — D26, D27, D28.)
- [x] Tokenizer equality check: Qwen base vs instruct identical (2 lines). If not → Δ_t token-list fallback. → **IDENTICAL** (vocab dicts equal; probe string with CJK/accents/punctuation encodes identically). Δ_t can be computed token-by-token; no fallback needed.
- [x] Two-hop prompt set (100–200) with known intermediate, filtered to prompts Qwen3.5-4B answers correctly; difficulty tuned to a 4B model. *(Moved ahead of Δ_t: feeds Step 7 "PROTECT ≥90 MIN" vs Δ_t feeding Step 5 "SKIPPABLE" — D20.)* → `src/twohop.py` → `results/step3_twohop.json`. **138 kept of 164** candidates (45 entities × 4 templates; filter = model's true final top-1 == answer). Intermediate and answer are both single Qwen tokens by construction; 25 distinct intermediates. **Shortcut control** (each template re-run with the entity blinded): 5 items solvable by blinded top-1, 39 by blinded top-5 → **99 strictly two-hop**. `capital` + `capital_of` (72 items) are clean by both criteria — Step 7 headline should use those; `currency` is 70% " euro" and mostly fails the strict test (D21). Rejects show the filter working on our own labels: Shanghai→"Mandarin", Mumbai→"New Delhi", Toronto→"Canadian" (D22).
- [x] Δ_t: `mean logprob_base(t) − mean logprob_instruct(t)` over a few hundred neutral contexts. Fallback: curated NSFW/profanity list. *(Tokenizer check passed, so the fallback is not needed.)* → `src/delta_t.py` → `results/step3_delta_t.json` + `.npz`. 200 pile texts (seed 1, disjoint from the step-2 survey's seed 0), 23,912 (doc, position≥4) samples per model, both models run and freed sequentially. **Δ_t is frequency-confounded: Spearman(log f_t, Δ_t) = +0.354**, median Δ by frequency quintile −0.047 → +0.107, because the instruct model shifts mass onto chat/formatting tokens and deflates content tokens in proportion to the mass they held (D23). **Validated frequency-matched: profanity excess +0.226 vs neutral +0.052 over controls within ±0.25 log-count (pool 92,846) — PASS** (D24; the first, un-matched gate failed for an invalid reason and is recorded). Components `mean_logprob_base`/`_instruct` stored alongside (D25).
- [x] Token flags: leading-space, punctuation, byte-fragment, non-English script. → done in Step 2 (`src/flags.py`); punctuation recorded but not junk (D10).

## Step 4 — The offset battery (hour 2.5–7, CORE) — DONE 21 Aug

*All averaging in score space; nothing is ever subtracted from a lens vector.*

- [x] **4.0 Noise null FIRST:** N ≈ 500–2000 (prompt, position) pairs, positions ≥ 4, diverse pile prompts. Split into two disjoint halves; compute `m_t` (mean score per token per layer) on each; correlate. **No replication ⇒ no stable offset ⇒ all three public claims fail at once — that is the writeup.** → **GATE PASSED, pre-registered criterion (D34): min J-lens split-half Pearson 0.9589 over layers (R 0.9625, logit 0.9570; Spearman ≥0.95 all three); 100 seed-2 docs (row-disjoint from seed-0/seed-1 samples, asserted), 1,987 samples, document-level halves. Pythia same design: 0.9902.** Caveat: top-100 *identity* overlap between halves only 0.56–0.84 — tail-membership claims are noisy, vector-level claims are not. (`src/offset_battery.py` → `results/step4_mt.npz`, `step4_noise_null.json`; Pythia: `src/offset_battery_pythia.py` → `step4_mt_pythia.npz`.)
- [x] `m_t`, `σ_t` per layer on Qwen (and Pythia when fit lands). High-m/low-σ = offset; high-σ = content. (Both elevated = multiplicative/unembedding-norm signature — Neel's own aside; flag if seen.) → **SEEN and flagged: top-100 m_t tokens sit at the 91st–99th σ_t percentile at every layer/instrument.** Composition: J L0 = `\xa0`/soft-hyphen/`�`/code fragments (junk share 0.22), L12 = ` thru`/`...`/whitespace runs (the Δ_t web-debris axis), L24 = function words. (`src/mt_diagnostics.py` → `results/step4_mt_diagnostics.json` + cached per-token flags `step4_token_flags.npz`.)
- [x] **4a Route A (Pythia):** analytic `W_U·β` from weights; compare to empirical `m_t`; correlate with `log f_t`. **GPT-2 cell (weights-only):** `W_U·β` vs `log f_t` — replicate/refute phoenix's r≈0.67. → **Phoenix REPLICATED: GPT-2 r=0.70 (claim 0.67), Pythia r=0.73; random-direction null |r|≤0.29 (5 seeds); row-norm confound reported (−0.21/−0.54, rare tokens have larger unembedding rows). Analytic-vs-empirical DISSOCIATES: W_U·β explains Pythia's logit-lens m_t at r=0.93–0.95 from L12 on, but the J-lens m_t only at 0.21–0.72 — the transported lens's offset is NOT mostly the bias.** (`src/route_a.py` → `results/step4_route_a.json` + `step4_wu_beta.npz`.)
- [x] **4b Route B (both):** `μ̄` = mean activation; readout `s(μ̄)` (does it look like the junk?); mean-centered `s(h − μ̄)` (does the junk disappear?). RMSNorm removes nothing → more Route B room on Qwen. → **s(μ̄) ≈ m_t at ρ=0.98–0.997 every layer/instrument (the offset lives in the mean activation; available on Qwen where Route A is impossible). But centering does NOT remove the junk — it increases it (J L0–10: 0.139→0.196; out-of-sample by construction, D35). Activation-space correction is dead; step 7's score-space correction is the one left standing.** (`src/route_b.py` → `results/step4_route_b.json` + `step4_mu.npz`.)
- [x] **4c H1 regression:** Spearman/OLS of `m_t` on `log f_t` per layer; r vs depth. Headline figure's main line. → **Mid-depth (L5–28): all three cells agree, POSITIVE (frequency-prior-like), peak L18–21 at ρ = 0.48 / 0.25 / 0.14 — the D31 range in action. Early layers (L0–4), where the junk lives: cells DISAGREE in sign and every |ρ|<0.12 — frequency does not explain the junk, and H1-as-junk-explanation fails while H1-as-frequency-prior holds. Bare-word restriction (D33) shifts ρ by ≤0.05, sign flips only where |ρ|<0.02. One headline number re-derived independently without scipy: exact match.** Interpretive limit stated: positive m_t–frequency correlation cannot separate instrument bias from genuine content. (`src/h1_regression.py` → `results/step4_h1_regression.json` + `step4_h1_r_by_depth.png`; refuses to run on a failed 4.0 gate.)
  - **Report a RANGE across three cells, never a single number (D31).** The two usable frequency sources agree at only ρ≈0.24, so H1's magnitude is source-dependent; power is not the constraint (N=10k–130k detects |ρ|≥0.01 at p<0.01), validity is.

    | cell | source | why this cell |
    |---|---|---|
    | Latin tokens | pile counts (`capped`) | best-measured; sources agree best here (ρ 0.39–0.52 with the noisy tail trimmed) |
    | whole vocabulary | wordfreq | widest coverage (90.5%) on one consistent scale |
    | **non-Latin only** | wordfreq | the junk itself — testable for the first time |

  - **Agree in sign across all three ⇒ H1 robust**, stated with more confidence than one number earns. **Disagree ⇒ that is the finding** ("whether the offset tracks frequency depends on which frequency you mean"), reported at full prominence.
  - **Exclude or dual-report `is_bare_word == False` tokens** (`results/step3_wordfreq.npz`): wordfreq strips punctuation and scores the embedded word, so `.Scene`→"scene" over-states the token. 8.9% of scored tokens; 4.5% of J-lens readout tokens (D33).
  - **Do NOT use merge rank** (`src/zipf_frequency.py`): same measurement as raw token id, and no signal past id 100k where the non-Latin vocabulary lives. Kept as a documented negative result for the writeup only (D32).
  - **Inherited hole, state it in every 4c number:** for non-Latin tokens there is no frequency measure from Qwen's own training data, and there will not be one in this project. wordfreq fixes coverage, not provenance (D26–D28).
- **Gate:** mechanism table has entries by 7h. If behind, **cut H2 (Step 5) before the dissociation.** → **Gate met.** Evidence: devlog `0.1.0`. Banked for later steps: R-lens m_t/σ_t (step 6 swap battery is now pure analysis), Pythia m_t (Route A arm), per-token flags npz.

## Step 5 — H2: tuning-induced suppression (hour 7–10, SKIPPABLE)

**Renamed 21 Aug, and the claim is narrower than "RLHF".** Δ_t measures the base→instruct distribution shift, and ranked by Δ_t the top of the vocabulary is not obscenity — it is informal, archaic and misspelled English (` ordinarily`, ` thankfully`, ` anyways`, ` hubby`, ` thru`, ` seper`) plus scraped-text artefacts (` ;-)`, `…the`), while the most-promoted end is entirely whitespace and chat formatting. The dominant axis is **"raw web text style" vs "clean assistant style."** Obscenity is real but secondary within it: profanity lands in the top 0.3–4% of the vocabulary, which is why the frequency-matched gate passes. So every H2 claim must read **"tuning-inhibited tokens, of which obscenity is one component"** — never "RLHF suppression" unqualified.

Two further honest limits: (a) the *base* model's pretraining data was itself filtered, so Δ_t captures only tuning-stage suppression, not suppression baked in at the data-filtering stage; (b) the generalised claim sits *closer* to H1 than the narrow one did — "informal, misspelled, web-debris" is very nearly a synonym for "rare" — so the frequency-matched control carries more weight here, not less. The honest headline is "tuning-inhibited tokens are over-represented early **over and above what their rarity predicts**", with the last clause doing the real work.

Natural successor experiment (writeup, not this project): take a model, finetune it to suppress a target *you chose*, then rerun this battery. Ground truth by construction, no confound to argue about. That is the experiment that would settle H2.

- [ ] Early-layer high-`m_t` tokens: enriched for high-Δ_t vs **frequency-matched controls**? (Obscene tokens are also rare — without the control, H1 and H2 are confounded.) **Confound now quantified, not hypothetical: Spearman(log f_t, Δ_t) = +0.354 (D23). The matched-control machinery is built and validated in `src/delta_t.py` (±0.25 log-count window, 92,846-token eligible pool) — reuse it, do not re-derive.**
- [ ] Partial correlation `m_t` ~ Δ_t controlling `log f_t`; enrichment gradient by depth; base-model contrast.
- [ ] **Conditional subtraction (added 21 Aug — decide from results, not in advance).** Δ_t is frequency-contaminated (ρ = +0.354, D23), so the H2 statistic must be computed BOTH ways: raw Δ_t, and frequency-residualised. Residualising means "the part of Δ_t that frequency does not explain" — use the **matched-control** form already built and validated in `src/delta_t.py` (median Δ of clean tokens within ±0.25 log-count), NOT a linear subtraction: the Δ-vs-frequency relationship is convex (quintile medians −0.047, −0.033, +0.001, +0.045, +0.107; steps of +0.014, +0.034, +0.044, +0.062), so a straight-line fit leaves structure in the residual that would masquerade as suppression.
  - **Decision rule, fixed now so it is not a judgement call later:** report both. If they agree in sign and significance → raw is the headline, residualised is the robustness line. If they **disagree** → the residualised result is primary and the disagreement is itself the finding, reported at full prominence.
  - **Caveat to state either way:** if tuning inhibits tokens *because* they are rare, residualising removes real causal signal along with the confound (controlling for a mediator). That makes the residualised result conservative — it can understate H2, not manufacture it.
  - **Skip condition:** if Step 4.0's noise null fails, none of this runs.
- **If cut:** "designed, not run" — say so at full prominence.

## Step 6 — H3: transport error (hour 10–12, cheap by construction)

- [ ] Identical Step 4 battery through the released R-lens at matched layers (same forward pass; only gradients differ).
- [ ] Offset ratio R/J by depth. **Shrinks under swap ⇒ transport error; survives ⇒ not transport** (bias route invariant by construction — final norm/unembed shared).
- [ ] Which junk *composition* survives (freq-correlated part? Δ_t part?).
- Note (20 Aug): R≈J on Qwen3.5-4B is *expected* — R-lens post: "no R-lens advantage for both the smallest dense and MoE models we tested"; advantage grows with scale (largest on DeepSeek-V4-Flash). Cite this in the writeup; our result is consistent, not contrary.
- **Weak-lever caveat (disclose in H3 writeup):** because J≈R on the 4B, "offset survives the swap" underdetermines transport error (not-transport vs. both-recipes-share-the-error); "shrinks under swap" remains decisive. The v2 "separate by construction" framing needs this qualifier.
  - **Qualifier weakened 20 Aug (red team, devlog 0.0.2 addendum 4).** "J≈R on the 4B" only holds near the destination. Pairwise top-10 overlap: **0.35/10 at L0**, 4.72 at L12, 6.64 at L20, ≥8.7 only from L27 (`results/step2_lens_agreement.png`). At the early/mid layers where the junk lives they are substantially different instruments, so the swap lever is stronger than assumed — reduces the case for the qwen3.5-9b download below.
- [ ] **Free H3 evidence already banked (Step 2 red team):** on the corrected junk rule, J and R read out *more* impossible tokens at L24-27 than L18-21 (J 0.069→0.109, t=+5.01, 22/25 texts; R 0.065→0.094, t=+4.05) while the logit lens moves monotonically the other way (0.369→0.216, t=−10.90, 0/25). A late rebound present only in the *translated* instruments is a transport-error signature. Pick this up here.
- [ ] **Decision after Step 4:** optional strengthener — run the Step 6 swap battery *only* on qwen3.5-9b (released matched J/R pair; first size where the R advantage plausibly appears) to turn the weak lever into a real one. Costs one model download + one battery run; decide on remaining time.

## Step 7 — Payoff: the calibration (hour 12–14.5, PROTECT ≥90 MIN)

- [ ] Corrected readout: re-rank by `s_t(h) − m_t`.
- [ ] Two-hop set: rank of known intermediate, by depth, pre/post correction, per instrument (logit/J/R). Second figure.

## Step 8 — Red team (hour 14.5–16)

- [ ] Shuffled-J / random-rotation null: does a "stable offset" appear anyway? (Then it's a `W_U`/norm property, not the lens.) — **random-rotation half done early** (Step 2 red team, `src/null_baselines.py`, 5 seeds): on the *junk* metric the null sits at 0.386 vs J/R 0.02–0.20. Still to do here: the same null on `m_t` itself, plus the shuffled-J variant.
- [ ] Out-of-sample `m_t` on a third disjoint prompt set.
- [ ] One deliberate attempt to break the headline result.
- [ ] Logit-lens baseline present in every table.

## Step 9 — Writeup (+2h; exec summary drafted from hour 1)

- [ ] Exec summary: question in 2 sentences; headline figure (`m_t`-vs-log-freq r by depth, J and R, Qwen, Pythia `W_U·β` inset); 3–5 takeaways with epistemic-status tags; honest hours; "what would change my mind."
- [ ] Takeaway map: (1) offset null? →4.0; (2) freq confound? →4c; (3) real without bias? →4b/Qwen; (4) carries to R-lens? →6; (5) correctable? →7.
- [ ] Failures/nulls at equal prominence. Figures titled as claims. Route A framing sentence: architecture-specific mechanism test, not general explanation.
- [ ] External review before 2 Sep (non-negotiable).

---

## Per-step model matrix

| Step | Qwen3.5-4B | Pythia-1.4B | GPT-2 |
|---|---|---|---|
| 1 Sanity | released J/R pair | — | — |
| 2 Junk figures | ✅ | fit in background | — |
| 3 Assets | freq, Δ_t, two-hop | freq | freq |
| 4.0 m/σ + null | ✅ primary | ✅ when fit lands | — |
| 4a Route A | impossible (that's the point) | ✅ analytic + empirical | weights-only phoenix cell |
| 4b Route B | ✅ full room | ✅ reduced room | — |
| 4c H1 regression | ✅ headline | ✅ | via 4a |
| 5 H2 | ✅ (+Base) | — | — |
| 6 H3 (R-lens) | ✅ | — | — |
| 7 Payoff | ✅ | optional | — |
