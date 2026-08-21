"""Step 4b Route B: the mean activation mu_bar -- the context-average term.

The Route A split (logits = context term + W_U beta) has no constant term on
Qwen (RMSNorm, no beta), so any stable offset must live in the AVERAGE of the
context-dependent term. This script measures that directly:

  1. mu_bar per layer = mean residual over the HALF-0 documents of the step-4
     sample (same seed-2 rows, same positions; deterministic re-derivation).
  2. Readout s(mu_bar): does the average activation alone read out the junk?
     (top-10 + junk flags per layer per instrument; correlation with m_t.)
  3. Centering, evaluated OUT-OF-SAMPLE on HALF-1 documents: junk fraction of
     top-10 readouts for h vs h - mu_bar. Junk that disappears under
     centering lived in the mean activation. The subtraction happens to the
     ACTIVATION before the lens -- never to scores ("nothing is ever
     subtracted from a lens vector").

Run: .venv/bin/python -m src.route_b   (~8 min; writes
results/step4_route_b.json + results/step4_mu.npz)
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr

from src.flags import junk_fraction, token_flags
from src.junk_survey import DATASET_ID, MAX_SEQ_LEN
from src.lens import KINDS, Instrument
from src.offset_battery import SEED, pick_fresh_rows, pick_random_positions

MT_NPZ = "results/step4_mt.npz"
OUT_JSON = "results/step4_route_b.json"
OUT_NPZ = "results/step4_mu.npz"
N_EVAL_DOCS = 25  # half-1 docs used for the centering evaluation
N_EVAL_POSITIONS = 5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-texts", type=int, default=100)
    parser.add_argument("--n-positions", type=int, default=20)
    args = parser.parse_args()

    all_texts = load_dataset(DATASET_ID, split="train")["text"]
    rows = pick_fresh_rows(args.n_texts, args.seed, len(all_texts))
    shuffled = rows.copy()
    random.Random(args.seed).shuffle(shuffled)
    half0 = shuffled[: len(shuffled) // 2]
    half1 = shuffled[len(shuffled) // 2:]

    inst = Instrument()
    layers = inst.source_layers
    d_model = inst.W_U.shape[1]

    # --- 1. mu_bar from half-0 docs, same positions as the battery would use
    # for those docs is NOT reproducible per-doc (the battery's position rng
    # is shared across all 100 docs in row order), so mu_bar uses its own
    # seeded positions; mu_bar is an estimate of E[h], any position sample of
    # the same distribution serves.
    pos_rng = random.Random(args.seed + 2000)
    mu_sum = {l: torch.zeros(d_model) for l in layers}
    n_mu = 0
    for i, row in enumerate(sorted(half0)):
        acts, input_ids = inst.residuals(all_texts[row], layers, max_seq_len=MAX_SEQ_LEN)
        positions = pick_random_positions(input_ids.shape[1], args.n_positions, pos_rng)
        for l in layers:
            mu_sum[l] += acts[l][positions].float().sum(dim=0).cpu()
        n_mu += len(positions)
        if (i + 1) % 20 == 0:
            print(f"mu_bar: {i + 1}/{len(half0)} docs", flush=True)
    mu = {l: (mu_sum[l] / n_mu).to(inst.device) for l in layers}

    # --- 2. readout of mu_bar itself
    mt = np.load(MT_NPZ)
    mt_kinds = [str(k) for k in mt["kinds"]]
    mu_readout = []
    for layer_idx, layer in enumerate(layers):
        for kind in KINDS:
            scores = inst.score(mu[layer], layer, kind)
            top = inst.top_tokens(scores, 10)
            top_strings = [t for t, _ in top]
            s_np = scores.cpu().numpy().astype(np.float64)
            m_vec = mt["m_t"][mt_kinds.index(kind), layer_idx].astype(np.float64)
            mu_readout.append({
                "layer": layer,
                "kind": kind,
                "top10": top_strings,
                "junk_fraction": junk_fraction(top_strings),
                "corr_with_m_t": {
                    "pearson": round(float(pearsonr(s_np, m_vec)[0]), 4),
                    "spearman": round(float(spearmanr(s_np, m_vec)[0]), 4),
                },
            })

    # --- 3. out-of-sample centering on half-1 docs
    eval_docs = sorted(half1)[:N_EVAL_DOCS]
    eval_rng = random.Random(args.seed + 3000)
    per_layer_junk = {l: {k: {"raw": [], "centered": []} for k in KINDS} for l in layers}
    for i, row in enumerate(eval_docs):
        acts, input_ids = inst.residuals(all_texts[row], layers, max_seq_len=MAX_SEQ_LEN)
        positions = pick_random_positions(input_ids.shape[1], N_EVAL_POSITIONS, eval_rng)
        for layer in layers:
            h = acts[layer][positions].float()
            for kind in KINDS:
                for variant, hv in (("raw", h), ("centered", h - mu[layer])):
                    scores = inst.score(hv, layer, kind)
                    for p in range(scores.shape[0]):
                        top_ids = scores[p].topk(10).indices.tolist()
                        tokens = [inst.tokenizer.decode([t]) for t in top_ids]
                        per_layer_junk[layer][kind][variant].append(junk_fraction(tokens))
        print(f"centering eval: {i + 1}/{len(eval_docs)} docs", flush=True)

    centering = []
    for layer in layers:
        for kind in KINDS:
            raw = per_layer_junk[layer][kind]["raw"]
            cen = per_layer_junk[layer][kind]["centered"]
            centering.append({
                "layer": layer,
                "kind": kind,
                "junk_raw": round(float(np.mean(raw)), 4),
                "junk_centered": round(float(np.mean(cen)), 4),
                "n_cells": len(raw),
            })

    np.savez(OUT_NPZ,
             mu=np.stack([mu[l].cpu().numpy() for l in layers]).astype(np.float32),
             layers=np.array(layers), n_mu_samples=np.array([n_mu]))
    out = {
        "meta": {
            "seed": args.seed,
            "mu_docs_half0": sorted(half0),
            "eval_docs_half1": eval_docs,
            "n_mu_samples": n_mu,
            "n_eval_positions": N_EVAL_POSITIONS,
            "note": "mu_bar from half-0 docs only; centering evaluated on half-1 docs "
                    "(out-of-sample). Subtraction applied to activations, never scores.",
        },
        "mu_readout": mu_readout,
        "centering": centering,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    j20 = [c for c in centering if c["kind"] == "J" and c["layer"] <= 10]
    print("J-lens junk raw->centered, layers 0-10:",
          [(c['layer'], c['junk_raw'], c['junk_centered']) for c in j20])
    print(f"wrote {OUT_JSON} and {OUT_NPZ}")


if __name__ == "__main__":
    main()
