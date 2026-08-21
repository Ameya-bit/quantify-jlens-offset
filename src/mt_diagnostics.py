"""Step 4 diagnostics: what kind of tokens carry the offset, and is it an
offset at all (high m, low sigma) or content (high sigma) or the
multiplicative/unembedding-norm signature (both elevated -- flag if seen)?

Per (instrument, layer) on the Qwen battery (results/step4_mt.npz):
  - corr(m_t, sigma_t) across the vocabulary
  - for the top-100 tokens by m_t: median sigma_t percentile, junk share
    (non_latin | byte_fragment, the step-2 corrected rule), share never seen
    in pile-10k, and example tokens at a few layers.

Also caches per-token-id flags to results/step4_token_flags.npz so later
steps stop re-deriving them (decode + flag over 248k ids, ~1 min).

Run: .venv/bin/python -m src.mt_diagnostics   (writes
results/step4_mt_diagnostics.json + results/step4_token_flags.npz)
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoTokenizer

from src.flags import token_flags
from src.lens import MODEL_ID

MT_NPZ = "results/step4_mt.npz"
COUNTS_NPZ = "results/step3_token_counts.npz"
FLAGS_NPZ = "results/step4_token_flags.npz"
OUT_JSON = "results/step4_mt_diagnostics.json"
TOP_K = 100
EXAMPLE_LAYERS = (0, 6, 12, 18, 24, 30)


def load_or_build_flags(n_vocab: int) -> dict[str, np.ndarray]:
    if os.path.exists(FLAGS_NPZ):
        z = np.load(FLAGS_NPZ)
        return {k: z[k] for k in z.files}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    keys = ("non_latin", "byte_fragment", "punctuation", "leading_space")
    cols = {k: np.zeros(n_vocab, dtype=bool) for k in keys}
    for i in range(n_vocab):
        f = token_flags(tokenizer.decode([i]))
        for k in keys:
            cols[k][i] = f[k]
    np.savez(FLAGS_NPZ, **cols)
    return cols


def main() -> None:
    mt = np.load(MT_NPZ)
    kinds = [str(k) for k in mt["kinds"]]
    layers = [int(l) for l in mt["layers"]]
    counts = np.load(COUNTS_NPZ)["qwen_capped"]
    n_vocab = len(counts)
    flags = load_or_build_flags(n_vocab)
    is_junk = flags["non_latin"] | flags["byte_fragment"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = []
    for kind_idx, kind in enumerate(kinds):
        for layer_idx, layer in enumerate(layers):
            m = mt["m_t"][kind_idx, layer_idx][:n_vocab].astype(np.float64)
            s = mt["sigma_t"][kind_idx, layer_idx][:n_vocab].astype(np.float64)
            top = np.argsort(m)[-TOP_K:]
            sigma_pctl = (s[:, None] <= s[top]).mean(axis=0)  # percentile of each top token's sigma
            row = {
                "kind": kind,
                "layer": layer,
                "corr_m_sigma": {
                    "pearson": round(float(pearsonr(m, s)[0]), 4),
                    "spearman": round(float(spearmanr(m, s)[0]), 4),
                },
                "top100_by_m": {
                    "median_sigma_percentile": round(float(np.median(sigma_pctl)), 4),
                    "junk_share": round(float(is_junk[top].mean()), 4),
                    "never_seen_in_pile_share": round(float((counts[top] == 0).mean()), 4),
                },
            }
            if layer in EXAMPLE_LAYERS:
                best = top[np.argsort(m[top])[::-1][:10]]
                row["top10_tokens"] = [tokenizer.decode([int(t)]) for t in best]
            rows.append(row)

    with open(OUT_JSON, "w") as f:
        json.dump({"meta": {"mt_npz": MT_NPZ, "top_k": TOP_K}, "rows": rows}, f,
                  ensure_ascii=False, indent=1)
    for kind in kinds:
        picks = [r for r in rows if r["kind"] == kind and r["layer"] in (0, 12, 24)]
        print(kind, [(r["layer"],
                      r["corr_m_sigma"]["spearman"],
                      r["top100_by_m"]["median_sigma_percentile"],
                      r["top100_by_m"]["junk_share"]) for r in picks])
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
