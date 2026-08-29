# The J-lens non-context offset: mostly the model's own frequency prior — and knowing that wins on a secret-elicitation benchmark

A ~16-hour clocked study for the MATS 12.0 application, plus a registered
extension. **Question:** J-lens readouts contain score that tokens collect
regardless of what the activation means — the "junk" three public threads
noticed and left hanging. What is it made of, and can you correct for it?

**Answer:** the offset (m_t: each token's mean score over ~2,000 unrelated
contexts) decomposes into three parts —

1. **Early layers: recipe-specific transport artifact.** J and R lenses
   manufacture *different* junk (top-100 overlap 0.03–0.14, R ≈ 2× J's size).
   Removable, and removing it helps (~2× rank improvement on R at L0–4).
2. **Mid depth: the model's own frequency prior — signal, not bias.** Shared
   across J/R, Spearman(m_t, log f) ≈ 0.48 at L18. The registered subtraction
   gate **fails**: removing m_t destroys real signal, with damage scaling in
   token frequency (ρ ≈ 0.69 dose-response). Domain-stratified m_t also
   fails its gate (0/6 pairs) — the offset is within-distribution only.
3. **Late layers: junk rebound in the transported lenses**, where only
   variance-scaling (z-scoring) helps.

**Payoff (registered, two model families):** on Cywiński/Ryd/Rajamanoharan/
Nanda's taboo secret-word organisms, z-score calibration built *only from the
clean base model* takes lens-based secret elicitation from **zero at every
layer to the only working method** on their Qwen3-1.7B organisms, and from
mid-pack to **best method on the 20-word Gemma-2-9B set (0.805 accuracy vs
0.665 for their protocol, leave-one-word-out selection)**. The raw J-lens
*loses* to the raw logit lens at 9B — the transported instrument carries the
bigger offset and only wins once calibrated. Registered failures reported at
equal prominence: "J peaks earlier" — no; frequency predicting *which*
secrets surface (P3) — null.

## Read this in order

1. **`steps.md`** — the full execution checklist with every gate, verdict,
   and evidence pointer inline.
2. **`devlog/`** — one evidence-grounded entry per step (below), plus
   **`devlog/DECISIONS.md`**: 44 numbered decisions/registrations (D1–D44),
   each with date, evidence, and a "reverses if" clause. Designs were
   registered before code; failed gates are reported as registered, never
   renegotiated.
3. **`src/`** — one runnable module per experiment
   ([module map](src/README.md)); **`results/`** — outputs foldered by step
   ([layout note](results/README.md)).

| Step | Question | Verdict | Devlog |
|---|---|---|---|
| 1 | Does the instrument work? | Sanity 3/3; L30 J=R=logit exact | [0.0.1](devlog/0.0.1-verifying-the-machinery.md) |
| 2 | Does the public junk reproduce? | Yes, all 3 instruments; nulls bracketed | [0.0.2](devlog/0.0.2-reproducing-the-junk.md) |
| 3 | Assets (frequencies, Δ_t, two-hop bench) | Δ_t is frequency-confounded (+0.354) — controls built | [0.0.3](devlog/0.0.3-staging-the-assets.md) |
| 4 | Is the offset stable? What explains it? (H1) | Stable (split-half ≥0.96); frequency-shaped at mid-depth; not the LayerNorm bias; centering fails | [0.1.0](devlog/0.1.0-the-offset-battery.md) |
| 5 | Tuning-induced suppression? (H2) | Not on the J-lens (null both ways); narrow R residue, later reattributed to R's recipe | [0.2.0](devlog/0.2.0-the-suppression-battery.md) |
| 6 | Transport error? (H3, J→R swap) | Early offset is recipe-specific (grows under swap, disjoint tokens); mid-depth component survives = model-side | [0.3.0](devlog/0.3.0-the-swap-battery.md) |
| 7 | Does subtracting m_t help? | **Registered gate FAILS** — hurts 5–8× at working depths; z-score rescues the late band | [0.4.0](devlog/0.4.0-the-calibration.md) |
| 8 | Red team | Headline survives its break attempt (dose-response); domain gate FAILS → claim narrowed; stability is readout-path generic, only *content* is lens-specific | [0.5.0](devlog/0.5.0-the-red-team.md) |
| 10 | Does any of this matter? (taboo/ELK bridge) | Calibration is the difference between eliciting nothing and winning the benchmark; P3 null | [0.6.0](devlog/0.6.0-the-taboo-arm.md) · [0.7.0](devlog/0.7.0-the-gemma-arm.md) |

## Setup

Models: Qwen3.5-4B (+Base) with the released `camilablank/workspace-lenses`
J/R pair (headline); Pythia-1.4B (LayerNorm arm, lens self-fitted with the
official `anthropics/jacobian-lens` tooling); GPT-2 (weights-only phoenix
cell); Qwen3-1.7B and Gemma-2-9B-it (taboo arms, lenses self-fitted on the
clean bases — the Gemma fit ran on a rented H100, total cloud cost $5.98).

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# plus: pip install -e <path-to-jacobian-lens>   (official tooling)
.venv/bin/python -m src.sanity        # step-1 gate; everything else follows steps.md
```

Large artifacts (fitted lenses `.pt`, m_t `.npz` over GitHub's limit) are
local-only and regenerable from the scripts; every JSON a number is quoted
from is tracked.

## Honest accounting

Clocked study on released artifacts, ~16 h active; lens fits and GPU runs
ran unattended off-clock (Pythia ~40 min MPS; Qwen3-1.7B ~84 min MPS;
Gemma-2-9B 79 min on an H100). Every agent-produced number got an
independent re-check before entering a figure; corrections and retracted
first readings are kept in the devlogs, not overwritten.
