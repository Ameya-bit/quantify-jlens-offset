"""Step 4a Route A: the analytic bias W_U @ beta (weights only, no text).

LayerNorm models end with LN(h) = gamma * norm(h) + beta before the output
head, so logits(h) = W_U @ (gamma * norm(h)) + W_U @ beta -- an exact split
into a context-dependent term and a per-token constant. This script computes
that constant for the two LayerNorm models and asks whether it explains the
offset:

  Pythia-1.4B : W_U @ beta  vs  log f_t (Pythia tokenizer, pile-10k counts)
                and, when results/step4/step4_mt_pythia.npz exists, vs the
                EMPIRICAL m_t per layer (the Route A hypothesis test proper).
  GPT-2       : W_U @ beta  vs  log f_t -- the phoenix cell, replicating or
                refuting the public claim of r ~ 0.67. Weights only.

Controls (every number gets one):
  - random-direction null: replace beta with 5 random gaussian vectors of the
    same norm; r(W_U @ g, log f) should sit near 0 if the beta result is
    about beta and not about W_U row geometry.
  - row-norm diagnostic: r(||W_U row||, log f), the confound the null guards
    against, reported openly.

Counts: `capped` is primary, `full` reported alongside (D17). Tokens with
count 0 are excluded from log-frequency correlations (no smoothing), and the
excluded share is reported per number (D18). Qwen appears here only as the
contrast class: RMSNorm has no beta, so the analytic constant does not exist.

Run: .venv/bin/python -m src.route_a   (~1 min CPU; writes
results/step4/step4_route_a.json + results/step4/step4_wu_beta.npz)
"""

from __future__ import annotations

import json

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM

N_NULL_SEEDS = 5
COUNTS_NPZ = "results/step3/step3_token_counts.npz"
MT_PYTHIA_NPZ = "results/step4/step4_mt_pythia.npz"
OUT_JSON = "results/step4/step4_route_a.json"
OUT_NPZ = "results/step4/step4_wu_beta.npz"

MODELS = {
    # model_id, path to final LayerNorm, path to output head, counts key
    "pythia-1.4b": ("EleutherAI/pythia-1.4b", "gpt_neox.final_layer_norm", "lm_head", "pythia"),
    "gpt2": ("openai-community/gpt2", "transformer.ln_f", "lm_head", "gpt2"),
}


def get_module(model, dotted: str):
    for attr in dotted.split("."):
        model = getattr(model, attr)
    return model


def analytic_bias(model_id: str, norm_path: str, head_path: str):
    """(W_U @ beta, ||W_U row|| per token, ||beta||, W_U) from weights alone, fp32 CPU."""
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
    beta = get_module(model, norm_path).bias.detach()
    w_u = get_module(model, head_path).weight.detach().clone()
    c = (w_u @ beta).numpy()
    row_norms = w_u.norm(dim=1).numpy()
    del model
    return c, row_norms, float(beta.norm()), w_u


def corr_vs_logfreq(vec: np.ndarray, counts: np.ndarray) -> dict:
    """Pearson/Spearman of vec vs log(count) on tokens with count >= 1."""
    seen = counts >= 1
    logf = np.log(counts[seen].astype(np.float64))
    return {
        "pearson": round(float(pearsonr(vec[seen], logf)[0]), 4),
        "spearman": round(float(spearmanr(vec[seen], logf)[0]), 4),
        "n_tokens": int(seen.sum()),
        "excluded_zero_count_share": round(float(1 - seen.mean()), 4),
    }


def main() -> None:
    counts_npz = np.load(COUNTS_NPZ)
    report: dict = {}
    vectors: dict[str, np.ndarray] = {}

    for name, (model_id, norm_path, head_path, counts_key) in MODELS.items():
        c, row_norms, beta_norm, w_u = analytic_bias(model_id, norm_path, head_path)
        n_vocab = len(counts_npz[f"{counts_key}_capped"])
        assert len(c) >= n_vocab, f"{name}: head rows {len(c)} < counts {n_vocab}"
        padded_rows = len(c) - n_vocab
        c, row_norms, w_u = c[:n_vocab], row_norms[:n_vocab], w_u[:n_vocab]  # drop padded embed rows
        vectors[f"{name}_wu_beta"] = c.astype(np.float32)

        entry: dict = {
            "model_id": model_id,
            "beta_norm": round(beta_norm, 4),
            "vocab_used": n_vocab,
            "padded_rows_dropped": padded_rows,
            "vs_log_freq": {},
            "row_norm_diagnostic": {},
            "random_direction_null": {},
        }
        rng = np.random.default_rng(0)
        for variant in ("capped", "full"):
            counts = counts_npz[f"{counts_key}_{variant}"]
            entry["vs_log_freq"][variant] = corr_vs_logfreq(c, counts)
            entry["row_norm_diagnostic"][variant] = corr_vs_logfreq(row_norms, counts)
        # null uses the primary (capped) counts; 5 same-norm random directions
        counts = counts_npz[f"{counts_key}_capped"]
        nulls = []
        for _ in range(N_NULL_SEEDS):
            g = torch.from_numpy(rng.standard_normal(w_u.shape[1])).float()
            g = g / g.norm() * beta_norm
            nulls.append(corr_vs_logfreq((w_u @ g).numpy(), counts)["pearson"])
        del w_u
        entry["random_direction_null"] = {
            "pearsons": nulls,
            "abs_max": max(abs(x) for x in nulls),
        }
        report[name] = entry
        print(f"{name}: r(W_U b, log f) capped = "
              f"{entry['vs_log_freq']['capped']['pearson']}, "
              f"null |r|max = {entry['random_direction_null']['abs_max']}", flush=True)

    # Route A hypothesis test proper: analytic constant vs empirical Pythia m_t
    try:
        mt = np.load(MT_PYTHIA_NPZ)  # our own artifact; plain arrays, no pickle needed
        kinds, layers = list(mt["kinds"]), list(mt["layers"])
        c = vectors["pythia-1.4b_wu_beta"].astype(np.float64)
        per_layer = []
        for kind_idx, kind in enumerate(kinds):
            for layer_idx, layer in enumerate(layers):
                m = mt["m_t"][kind_idx, layer_idx][: len(c)].astype(np.float64)
                per_layer.append({
                    "kind": str(kind),
                    "layer": int(layer),
                    "pearson": round(float(pearsonr(c, m)[0]), 4),
                    "spearman": round(float(spearmanr(c, m)[0]), 4),
                })
        report["pythia_analytic_vs_empirical"] = per_layer
    except FileNotFoundError:
        report["pythia_analytic_vs_empirical"] = "pending: run src.offset_battery_pythia first"

    np.savez(OUT_NPZ, **vectors)
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(f"wrote {OUT_JSON} and {OUT_NPZ}")


if __name__ == "__main__":
    main()
