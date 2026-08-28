"""m_t/sigma_t for the gemma-2-9b-it lens (Gemma taboo arm's calibration).

Mirror of src.taboo_mt on the src.fit_gemma2 lens: same fresh seed-2
document selection, same document-level split-half check, same score space
(pre-softmax logits after final norm + head, INCLUDING Gemma's final logit
softcap — the lens wrapper's unembed applies it, so m_t lives in the same
space the eval reads). Estimated on the CLEAN base model. The pile->chat
domain-shift caveat from taboo_mt applies unchanged.

Run: python -m src.taboo_mt_gemma   (writes
results/taboo_gemma/gemma2_mt.npz + results/taboo_gemma/gemma2_noise_null.json)
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from jlens.hooks import ActivationRecorder

from src.fit_gemma2 import MODEL_ID, load_lens_model, pick_device
from src.junk_survey import DATASET_ID, MAX_SEQ_LEN
from src.offset_battery import (
    N_POSITIONS,
    N_TEXTS,
    SEED,
    Accumulator,
    pick_fresh_rows,
    pick_random_positions,
    prior_rows,
    split_half_stats,
)

LENS_PATH = "results/taboo_gemma/gemma2_9b_jlens.pt"
KINDS = ("logit", "J")
NPZ_PATH = "results/taboo_gemma/gemma2_mt.npz"
JSON_PATH = "results/taboo_gemma/gemma2_noise_null.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-texts", type=int, default=N_TEXTS)
    parser.add_argument("--n-positions", type=int, default=N_POSITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    all_texts = load_dataset(DATASET_ID, split="train")["text"]
    rows = pick_fresh_rows(args.n_texts, args.seed, len(all_texts))
    assert not set(rows) & prior_rows(), "row leakage into prior samples"
    shuffled = rows.copy()
    random.Random(args.seed).shuffle(shuffled)
    half_of = {row: (0 if i < len(shuffled) // 2 else 1) for i, row in enumerate(shuffled)}

    device = pick_device()
    lm = load_lens_model(device)
    ckpt = torch.load(LENS_PATH, map_location="cpu", weights_only=True)
    lens = {layer: m.float().to(device) for layer, m in ckpt["J"].items()}
    layers = sorted(lens)

    halves: list[Accumulator] | None = None
    vocab = None

    pos_rng = random.Random(args.seed + 1000)
    for i, row in enumerate(rows):
        input_ids = lm.encode(all_texts[row], max_length=MAX_SEQ_LEN)
        with torch.no_grad(), ActivationRecorder(lm.layers, at=layers) as rec:
            lm.forward(input_ids)
            acts = {l: rec.activations[l][0].detach() for l in layers}
        positions = pick_random_positions(input_ids.shape[1], args.n_positions, pos_rng)
        with torch.no_grad():
            for layer_idx, layer in enumerate(layers):
                h = acts[layer][positions].float()
                for kind_idx, kind in enumerate(KINDS):
                    ht = h if kind == "logit" else h @ lens[layer].T
                    scores = lm.unembed(ht)
                    if halves is None:
                        vocab = scores.shape[-1]
                        halves = [Accumulator(len(KINDS), len(layers), vocab)
                                  for _ in range(2)]
                    halves[half_of[row]].add(kind_idx, layer_idx, scores.cpu().numpy())
        halves[half_of[row]].n += len(positions)
        print(f"[{i + 1}/{len(rows)}] row {row} half {half_of[row]}", flush=True)

    m_half = [acc.mean() for acc in halves]
    combined = Accumulator(len(KINDS), len(layers), vocab)
    combined.total = halves[0].total + halves[1].total
    combined.total_sq = halves[0].total_sq + halves[1].total_sq
    combined.n = halves[0].n + halves[1].n

    np.savez(
        NPZ_PATH,
        m_t=combined.mean().astype(np.float32),
        sigma_t=combined.std().astype(np.float32),
        m_t_half0=m_half[0].astype(np.float32),
        m_t_half1=m_half[1].astype(np.float32),
        kinds=np.array(KINDS),
        layers=np.array(layers),
        n_samples=np.array([halves[0].n, halves[1].n]),
    )

    stats = split_half_stats(m_half[0], m_half[1], layers, kinds=KINDS)
    j_pearsons = [s["pearson"] for s in stats if s["kind"] == "J"]
    result = {
        "model_id": MODEL_ID,
        "lens": LENS_PATH,
        "n_texts": len(rows),
        "n_samples": [halves[0].n, halves[1].n],
        "split_half": stats,
        "min_J_pearson": min(j_pearsons),
    }
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"min J-lens split-half Pearson {min(j_pearsons):.4f}")
    print(f"wrote {NPZ_PATH} + {JSON_PATH}")


if __name__ == "__main__":
    main()
