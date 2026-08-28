"""Step 4.0/4a support: m_t/sigma_t battery on Pythia-1.4B (our fitted lens).

Mirror of src.offset_battery for the Route A arm: same fresh seed-2 document
selection (identical rows -- pick_fresh_rows is deterministic), same
document-level split-half null, same score space (pre-softmax logits; here
`lm.unembed` = final LayerNorm INCLUDING beta + output head, all fp32).
Kinds are (logit, J) only: no released R-lens exists for Pythia.

The lens is our own results/step2/pythia_jlens.pt (passed the step-2 sanity
battery 3/3), fitted layers 0-21. Split-half agreement is reported per layer;
the pre-registered 4.0 gate lives on the Qwen run, this one is the Route A
arm's replication check.

Run: .venv/bin/python -m src.offset_battery_pythia   (~5 min; writes
results/step4/step4_mt_pythia.npz + results/step4/step4_noise_null_pythia.json)
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
import torch
from datasets import load_dataset
from jlens.hooks import ActivationRecorder

from src.fit_pythia import load_lens_model
from src.junk_survey import DATASET_ID, MAX_SEQ_LEN, SKIP_FIRST
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

LENS_PATH = "results/step2/pythia_jlens.pt"
KINDS = ("logit", "J")
NPZ_PATH = "results/step4/step4_mt_pythia.npz"
JSON_PATH = "results/step4/step4_noise_null_pythia.json"


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

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lm = load_lens_model(device)
    ckpt = torch.load(LENS_PATH, map_location="cpu", weights_only=True)
    lens = {layer: m.float().to(device) for layer, m in ckpt["J"].items()}
    layers = sorted(lens)  # 0..21

    halves: list[Accumulator] | None = None  # sized lazily from the first score batch

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
        "meta": {
            "dataset": DATASET_ID,
            "model": "EleutherAI/pythia-1.4b",
            "lens": LENS_PATH,
            "seed": args.seed,
            "rows": rows,
            "n_texts": args.n_texts,
            "n_positions": args.n_positions,
            "samples_per_half": [halves[0].n, halves[1].n],
            "skip_first": SKIP_FIRST,
            "max_seq_len": MAX_SEQ_LEN,
            "vocab_width": vocab,
            "score_space": "pre-softmax logits via lm.unembed (LayerNorm incl. beta, fp32)",
            "npz": NPZ_PATH,
        },
        "min_J_pearson": min(j_pearsons),
        "split_half": stats,
    }
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"min J-lens split-half Pearson {min(j_pearsons):.4f}")
    print(f"wrote {NPZ_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
