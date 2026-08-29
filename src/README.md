# src/ — module map

Every module is a step-scoped, runnable entry point (`python -m src.<name>`);
shared machinery lives in the instrument/asset modules and is imported, never
duplicated. Modules refuse to run if their upstream gate failed. Numbers in
`results/` trace to exactly one module here.

## Instrument (shared)

| Module | What it is |
|---|---|
| `lens.py` | The instrument: loads Qwen3.5-4B + released J/R lenses; `Instrument.score(h, layer, kind)` for kind ∈ {logit, J, R} — one readout path (`W_U · finalnorm(J_ℓ·h)`, fp32) shared by every battery |
| `sanity.py` | Step-1 hard gate: anchor-row identity, late-layer J≈logit agreement, known-fact probes |
| `peek.py` | Manual single-cell inspection (used for by-eye re-checks) |
| `flags.py` | Token flags: junk (non-Latin/byte), punctuation, leading-space, bare-word |

## Step 2 — reproduce the public junk

`junk_survey.py` (25-text survey, top-10 readouts per layer/position) ·
`step2_figures.py` (junk-fraction + grid figures) · `null_baselines.py`
(uniform / random-rotation / L30 floors) · `qwen_agreement.py` (lens-vs-lens
agreement, n=125) · `fit_pythia.py` + `sanity_pythia.py` (Route-A arm: official
`jlens.fit` on Pythia-1.4B + its gate)

## Step 3 — assets

`frequencies.py` (pile-10k token counts, 3 tokenizers) · `zipf_frequency.py`
(merge-rank fallback — documented negative, D32) · `multilingual_freq.py`
(wordfreq 12-language rescue of non-Latin coverage) · `delta_t.py`
(base-vs-instruct suppression axis + frequency-matched controls) · `twohop.py`
(138-item two-hop bench with shortcut controls)

## Steps 4–8 — the batteries

| Module | Step | Question |
|---|---|---|
| `offset_battery.py` (+ `_pythia`) | 4.0 | Is the offset m_t stable? (split-half noise null) |
| `mt_diagnostics.py` | 4 | m_t/σ_t composition by depth |
| `route_a.py` | 4a | Is it the LayerNorm bias? (analytic W_U·β; phoenix r replication) |
| `route_b.py` | 4b | Is it the mean activation? Does centering remove it? |
| `h1_regression.py` | 4c | Does m_t track log-frequency? (three-cell range, D31) |
| `h2_battery.py` | 5 | Is it tuning-suppressed content? (frequency-matched controls) |
| `h3_swap.py` | 6 | Transport error? (J→R instrument swap) |
| `calibration.py` | 7 | Does subtracting m_t help? (registered gate — it fails; z-score variant) |
| `domain_mt.py` | 8a–c | Domain-stratified m_t, out-of-sample replication, rotation nulls |
| `freq_slice.py` | 8d | Break attempt: does correction damage scale with frequency? |

## Step 10 — taboo elicitation (ELK bridge)

Two arms, same design (registered as D43/D44 in `devlog/DECISIONS.md`):
lens fitted on the **clean base**, read out on Cywiński et al.'s taboo
organisms; their protocol reimplemented verbatim; variants raw vs z-scored.

| Qwen3-1.7B arm | Gemma-2-9B arm (20-word headline set) |
|---|---|
| `fit_qwen3.py` | `fit_gemma2.py` |
| `sanity_qwen3.py` | `sanity_gemma2.py` |
| `taboo_mt.py` | `taboo_mt_gemma.py` |
| `taboo_eval.py` (protocol + shared eval machinery) | `taboo_eval_gemma.py` |
| `taboo_analysis.py` | `taboo_analysis_gemma.py` (adds P3 + reproduction check) |
