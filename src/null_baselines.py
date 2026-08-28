"""Step 2 baselines: what junk rate would something that is NOT a lens give?

Three reference points for the junk figure, so "the J-lens reads out ~20%
impossible tokens early" can be read against a yardstick instead of in a
vacuum. All use the same junk rule as the survey (byte_fragment | non_latin;
see src.flags).

  1. `uniform_vocab`   -- draw 10 tokens uniformly from Qwen's 248k-token
     vocabulary. Upper reference: 43% of the vocabulary is non-Latin, so an
     uninformative readout lands here.
  2. `text_base_rate`  -- the tokens the model is actually trying to predict,
     on the same 25 surveyed pile texts. Lower reference (the floor a perfect
     instrument would hit). Its `punctuation` companion is the reference for
     the punctuation panel: punctuation is scored as clean, so the honest
     question there is "does the instrument predict it at the rate real text
     uses it?", not "how low can it go?".
  3. `rotation_null`   -- replace the fitted J_l with a random ROTATION
     matrix and read out through the identical final-norm -> W_U path. This
     is the step-8 pre-registered null ("shuffled-J / random-rotation"),
     pulled forward: it answers "is this junk rate just what any direction
     gives?". A rotation (not an arbitrary Gaussian) is the right null
     because it preserves the activation's length and destroys only its
     alignment with W_U.

Rotation-null sampling: one Q per seed, reused across layers -- so the
spread across seeds measures matrix-draw variability (the thing one draw
could get lucky on), while variation across layers within a seed is driven
by the activations. Layers are strided to keep this to ~2 min; the null is
flat with depth by construction, so a dense sweep buys nothing.

Run: .venv/bin/python -m src.null_baselines  (writes results/step2/step2_baselines.json)
"""

from __future__ import annotations

import json

import torch
from datasets import load_dataset

from src.flags import token_flags
from src.junk_survey import DATASET_ID, MAX_SEQ_LEN, pick_positions, pick_rows
from src.lens import Instrument

OUT_PATH = "results/step2/step2_baselines.json"
NULL_SEEDS = [0, 1, 2, 3, 4]
NULL_LAYERS = [0, 5, 10, 15, 20, 25, 30]
N_TEXTS = 25
SEED = 0
N_POSITIONS = 5
TOP_K = 10


def junk_rate(tokens: list[str]) -> float:
    return sum(token_flags(t)["is_junk"] for t in tokens) / len(tokens)


def main() -> None:
    inst = Instrument()
    texts = load_dataset(DATASET_ID, split="train")["text"]
    rows = pick_rows(N_TEXTS, SEED, len(texts))

    # --- 1. uniform over the vocabulary ---
    n_vocab = len(inst.tokenizer)
    n_junk = sum(
        token_flags(inst.tokenizer.decode([i]))["is_junk"] for i in range(n_vocab)
    )
    uniform = n_junk / n_vocab
    print(f"uniform-vocab junk rate: {uniform:.4f}  ({n_junk}/{n_vocab})")

    # --- 2. the tokens the model is actually trying to predict ---
    n_tok = n_tok_junk = n_tok_punct = 0
    for row in rows:
        ids = inst.tokenizer(texts[row], truncation=True, max_length=MAX_SEQ_LEN)[
            "input_ids"
        ]
        for i in ids:
            f = token_flags(inst.tokenizer.decode([i]))
            n_tok += 1
            n_tok_junk += f["is_junk"]
            n_tok_punct += f["punctuation"]
    text_rate = n_tok_junk / n_tok
    text_punct = n_tok_punct / n_tok
    print(f"text base rate: junk {text_rate:.4f}, punctuation {text_punct:.4f} "
          f"({n_tok} tokens)")

    # --- 3. random-rotation null ---
    d_model = inst.W_U.shape[1]
    per_seed: dict[int, dict[int, float]] = {}
    acts_cache = {}
    for row in rows:
        acts, ids = inst.residuals(texts[row], NULL_LAYERS, max_seq_len=MAX_SEQ_LEN)
        acts_cache[row] = (acts, pick_positions(ids.shape[1], N_POSITIONS))

    for seed in NULL_SEEDS:
        g = torch.Generator().manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(d_model, d_model, generator=g))
        Q = Q.to(inst.device)
        by_layer = {}
        for layer in NULL_LAYERS:
            rates = []
            for row in rows:
                acts, positions = acts_cache[row]
                for pos in positions:
                    h = acts[layer][pos].to(inst.device).float()
                    scores = inst.final_norm(h @ Q.T).float() @ inst.W_U.T
                    rates.append(
                        junk_rate([t for t, _ in inst.top_tokens(scores, TOP_K)])
                    )
            by_layer[layer] = sum(rates) / len(rates)
        per_seed[seed] = by_layer
        print(f"  seed {seed}: " + " ".join(f"L{l}={v:.3f}" for l, v in by_layer.items()))

    pooled = [v for s in per_seed.values() for v in s.values()]
    mean = sum(pooled) / len(pooled)
    seed_means = [sum(s.values()) / len(s) for s in per_seed.values()]
    spread = max(seed_means) - min(seed_means)
    print(f"rotation null: mean {mean:.4f}, seed means {[round(x,3) for x in seed_means]} "
          f"(spread {spread:.4f})")

    record = {
        "junk_rule": "byte_fragment | non_latin (punctuation excluded, see src.flags)",
        "uniform_vocab": {"rate": round(uniform, 4), "n_vocab": n_vocab},
        "text_base_rate": {
            "rate": round(text_rate, 4),
            "punctuation": round(text_punct, 4),
            "n_tokens": n_tok,
            "rows": rows,
        },
        "rotation_null": {
            "seeds": NULL_SEEDS,
            "layers": NULL_LAYERS,
            "n_samples_per_layer_per_seed": sum(
                len(p) for _, p in acts_cache.values()
            ),
            "by_seed": {str(s): {str(l): round(v, 4) for l, v in d.items()}
                        for s, d in per_seed.items()},
            "mean": round(mean, 4),
            "seed_mean_spread": round(spread, 4),
            "min": round(min(pooled), 4),
            "max": round(max(pooled), 4),
        },
    }
    with open(OUT_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
