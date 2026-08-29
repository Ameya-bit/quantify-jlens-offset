# Results layout

Reorganized 28 Aug 2026 (was flat). Files keep their original `stepN_` names,
foldered by step: `step1/` … `step8/`. Devlog and steps.md entries written
before 28 Aug cite the old flat paths (`results/stepN_*`); map them as
`results/stepN_X` → `results/stepN/stepN_X`. Non-prefixed step-2 artifacts
(`pythia_jlens.pt`, `pythia_fit_ckpt.pt`, `pythia_sanity.json`,
`qwen_agreement.json`) live in `step2/`; the two `.pt` lens artifacts are
gitignored.

Large `.npz` arrays (step-8 domain arms, taboo m_t) exceed GitHub's 100 MB
file limit and are gitignored: they live only on the machine that produced
them and are regenerable from the committed scripts (`src/domain_mt.py`,
`src/taboo_mt.py`); the JSON summaries derived from them are tracked.
