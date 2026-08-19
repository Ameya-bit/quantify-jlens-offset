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

## Step 1 — Load + sanity-check the instrument (hour 0–1)

*The lens turns a layer's activation into a score for every vocab token; if it loads wrong, everything downstream is garbage.*

- [ ] `src/lens.py`: load model + `lens.pt`; `score(h, layer, kind)` for kind ∈ {logit, J, R} sharing all code. Readout `softmax(W_U · finalnorm(J_ℓ · h))`; **fp32 for norm→unembed on MPS** even if model is bf16.
- [ ] `src/sanity.py`:
  - [ ] Assert target-layer anchor row of `J` ≈ identity (artifact guarantees this).
  - [ ] Late layers: J-lens ≈ logit-lens agreement (top-k overlap).
  - [ ] Known-fact prompts: sensible mid-layer readouts (e.g. Eiffel→Paris by mid-depth).
- **Gate:** passes by 1h. **Fallback:** switch headline to qwen3.5-9b (next lensed dense model in repo).

## Step 2 — Reproduce the public junk (hour 1–2.5)

*Confirm the phenomena the hypotheses are supposed to explain actually appear on our setup.*

- [ ] ~25 pile-10k prompts; per layer, top-10 readout tokens at several positions (skip first 4 — matches lens fit).
- [ ] Setup figure: junk (obscene/CJK/punctuation/rare fragments) at early layers, fading with depth — J-lens vs R-lens vs logit-lens side by side.
- [ ] Background: kick off Pythia-1.4B J-lens fit via official `jlens.fit` (n=10). If not converged by hour 4, continue without; Route A degrades to analytic-`W_U·β`-only (disclose).
- **Gate:** setup figures show the junk.

## Step 3 — Stage assets (overlaps hour 1–2.5)

- [ ] `log f_t`: token counts on pile-10k under Qwen tokenizer; separately under Pythia tokenizer (bonus: Pythia trained on the Pile — counts match its training distribution); separately under GPT-2 tokenizer (for the phoenix cell).
- [ ] Tokenizer equality check: Qwen base vs instruct identical (2 lines). If not → Δ_t token-list fallback.
- [ ] Δ_t: `mean logprob_base(t) − mean logprob_instruct(t)` over a few hundred neutral contexts. Fallback: curated NSFW/profanity list.
- [ ] Two-hop prompt set (100–200) with known intermediate, filtered to prompts Qwen3.5-4B answers correctly; difficulty tuned to a 4B model.
- [ ] Token flags: leading-space, punctuation, byte-fragment, non-English script.

## Step 4 — The offset battery (hour 2.5–7, CORE)

*All averaging in score space; nothing is ever subtracted from a lens vector.*

- [ ] **4.0 Noise null FIRST:** N ≈ 500–2000 (prompt, position) pairs, positions ≥ 4, diverse pile prompts. Split into two disjoint halves; compute `m_t` (mean score per token per layer) on each; correlate. **No replication ⇒ no stable offset ⇒ all three public claims fail at once — that is the writeup.**
- [ ] `m_t`, `σ_t` per layer on Qwen (and Pythia when fit lands). High-m/low-σ = offset; high-σ = content. (Both elevated = multiplicative/unembedding-norm signature — Neel's own aside; flag if seen.)
- [ ] **4a Route A (Pythia):** analytic `W_U·β` from weights; compare to empirical `m_t`; correlate with `log f_t`. **GPT-2 cell (weights-only):** `W_U·β` vs `log f_t` — replicate/refute phoenix's r≈0.67.
- [ ] **4b Route B (both):** `μ̄` = mean activation; readout `s(μ̄)` (does it look like the junk?); mean-centered `s(h − μ̄)` (does the junk disappear?). RMSNorm removes nothing → more Route B room on Qwen.
- [ ] **4c H1 regression:** Spearman/OLS of `m_t` on `log f_t` per layer; r vs depth. Headline figure's main line.
- **Gate:** mechanism table has entries by 7h. If behind, **cut H2 (Step 5) before the dissociation.**

## Step 5 — H2: RLHF suppression (hour 7–10, SKIPPABLE)

- [ ] Early-layer high-`m_t` tokens: enriched for high-Δ_t vs **frequency-matched controls**? (Obscene tokens are also rare — without the control, H1 and H2 are confounded.)
- [ ] Partial correlation `m_t` ~ Δ_t controlling `log f_t`; enrichment gradient by depth; base-model contrast.
- **If cut:** "designed, not run" — say so at full prominence.

## Step 6 — H3: transport error (hour 10–12, cheap by construction)

- [ ] Identical Step 4 battery through the released R-lens at matched layers (same forward pass; only gradients differ).
- [ ] Offset ratio R/J by depth. **Shrinks under swap ⇒ transport error; survives ⇒ not transport** (bias route invariant by construction — final norm/unembed shared).
- [ ] Which junk *composition* survives (freq-correlated part? Δ_t part?).

## Step 7 — Payoff: the calibration (hour 12–14.5, PROTECT ≥90 MIN)

- [ ] Corrected readout: re-rank by `s_t(h) − m_t`.
- [ ] Two-hop set: rank of known intermediate, by depth, pre/post correction, per instrument (logit/J/R). Second figure.

## Step 8 — Red team (hour 14.5–16)

- [ ] Shuffled-J / random-rotation null: does a "stable offset" appear anyway? (Then it's a `W_U`/norm property, not the lens.)
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
