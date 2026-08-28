"""Step 4.0 + m_t/sigma_t: the offset battery on Qwen3.5-4B.

For N_TEXTS pile-10k documents (seed 2; rows disjoint from BOTH the step-2
survey sample (seed 0) and the delta_t sample (seed 1), exclusions read back
from their results JSONs, not re-derived), sample N_POSITIONS positions >= 4
per document and read out every position through all three instruments
(J, R, logit) at every lensed layer (0-30).

Accumulated per (kind, layer, token), in the declared score space
(pre-softmax logits, src/lens.py):
  m_t     = mean score        -- the candidate offset
  sigma_t = std of score      -- high = content-dependent, low = constant

Noise null (step 4.0, runs on the same pass): documents are split into two
disjoint halves; m_t is computed independently on each; the halves are
correlated per layer. Pre-registered gate (set before any number was seen):
PASS iff split-half Pearson r >= 0.9 at EVERY layer for the J-lens.
No replication => no stable offset => all three public claims fail at once.

Run: .venv/bin/python -m src.offset_battery   (~10 min; writes
results/step4/step4_mt.npz + results/step4/step4_noise_null.json)
"""

from __future__ import annotations

import argparse
import json
import random

import numpy as np
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr

from src.junk_survey import DATASET_ID, MAX_SEQ_LEN, N_FIT_ROWS, SKIP_FIRST
from src.lens import KINDS, Instrument

SEED = 2  # step-2 survey used 0, delta_t used 1
N_TEXTS = 100
N_POSITIONS = 20
GATE_MIN_PEARSON = 0.9
PRIOR_ROW_FILES = ["results/step2/step2_readouts.json", "results/step3/step3_delta_t.json"]
NPZ_PATH = "results/step4/step4_mt.npz"
JSON_PATH = "results/step4/step4_noise_null.json"


def prior_rows() -> set[int]:
    """Rows already used by earlier steps, read from their results files."""
    used: set[int] = set()
    for path in PRIOR_ROW_FILES:
        with open(path) as f:
            data = json.load(f)
        used.update(data["meta"]["rows"] if "meta" in data else data["rows"])
    return used


def pick_fresh_rows(n_texts: int, seed: int, n_rows_total: int) -> list[int]:
    """Seeded sample from rows 25.. excluding every previously used row."""
    pool = sorted(set(range(N_FIT_ROWS, n_rows_total)) - prior_rows())
    return sorted(random.Random(seed).sample(pool, n_texts))


def pick_random_positions(seq_len: int, n_positions: int, rng: random.Random) -> list[int]:
    """Up to n_positions distinct positions in [SKIP_FIRST, seq_len - 1]."""
    valid = range(SKIP_FIRST, seq_len)
    return sorted(rng.sample(list(valid), min(n_positions, len(valid))))


class Accumulator:
    """Running sum / sum-of-squares per (kind, layer, token), float64."""

    def __init__(self, n_kinds: int, n_layers: int, vocab: int):
        self.total = np.zeros((n_kinds, n_layers, vocab))
        self.total_sq = np.zeros((n_kinds, n_layers, vocab))
        self.n = 0

    def add(self, kind_idx: int, layer_idx: int, scores: np.ndarray) -> None:
        """scores: [n_positions, vocab] fp32 for one (doc, kind, layer)."""
        s = scores.astype(np.float64)
        self.total[kind_idx, layer_idx] += s.sum(axis=0)
        self.total_sq[kind_idx, layer_idx] += (s * s).sum(axis=0)

    def mean(self) -> np.ndarray:
        return self.total / self.n

    def std(self) -> np.ndarray:
        var = self.total_sq / self.n - self.mean() ** 2
        return np.sqrt(np.clip(var, 0.0, None))


def split_half_stats(
    m_a: np.ndarray, m_b: np.ndarray, layers: list[int],
    kinds: tuple[str, ...] = KINDS, top_k: int = 100,
) -> list[dict]:
    """Per-layer agreement between the two halves' m_t, per kind."""
    out = []
    for kind_idx, kind in enumerate(kinds):
        for layer_idx, layer in enumerate(layers):
            a, b = m_a[kind_idx, layer_idx], m_b[kind_idx, layer_idx]
            top_a = set(np.argsort(a)[-top_k:].tolist())
            top_b = set(np.argsort(b)[-top_k:].tolist())
            out.append(
                {
                    "kind": kind,
                    "layer": layer,
                    "pearson": round(float(pearsonr(a, b)[0]), 6),
                    "spearman": round(float(spearmanr(a, b)[0]), 6),
                    f"top{top_k}_overlap": len(top_a & top_b) / top_k,
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-texts", type=int, default=N_TEXTS)
    parser.add_argument("--n-positions", type=int, default=N_POSITIONS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    all_texts = load_dataset(DATASET_ID, split="train")["text"]
    rows = pick_fresh_rows(args.n_texts, args.seed, len(all_texts))
    overlap = set(rows) & prior_rows()
    assert not overlap, f"row leakage into prior samples: {sorted(overlap)}"

    # Halves are disjoint SETS OF DOCUMENTS (never positions within a doc),
    # assigned by shuffling the row list with the same seed.
    shuffled = rows.copy()
    random.Random(args.seed).shuffle(shuffled)
    half_of = {row: (0 if i < len(shuffled) // 2 else 1) for i, row in enumerate(shuffled)}

    inst = Instrument()
    layers = inst.source_layers  # 0..30
    vocab = inst.W_U.shape[0]
    halves = [Accumulator(len(KINDS), len(layers), vocab) for _ in range(2)]

    pos_rng = random.Random(args.seed + 1000)
    positions_used = 0
    for i, row in enumerate(rows):
        acc = halves[half_of[row]]
        acts, input_ids = inst.residuals(all_texts[row], layers, max_seq_len=MAX_SEQ_LEN)
        positions = pick_random_positions(input_ids.shape[1], args.n_positions, pos_rng)
        for layer_idx, layer in enumerate(layers):
            h = acts[layer][positions]  # [P, d_model]
            for kind_idx, kind in enumerate(KINDS):
                scores = inst.score(h, layer, kind)  # [P, vocab] fp32
                acc.add(kind_idx, layer_idx, scores.cpu().numpy())
        acc.n += len(positions)
        positions_used += len(positions)
        print(f"[{i + 1}/{len(rows)}] row {row} half {half_of[row]} "
              f"+{len(positions)} pos (total {positions_used})", flush=True)

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

    stats = split_half_stats(m_half[0], m_half[1], layers)
    j_pearsons = [s["pearson"] for s in stats if s["kind"] == "J"]
    gate = {
        "criterion": f"split-half Pearson >= {GATE_MIN_PEARSON} at every layer, J-lens",
        "min_J_pearson": min(j_pearsons),
        "argmin_layer": layers[int(np.argmin(j_pearsons))],
        "pass": min(j_pearsons) >= GATE_MIN_PEARSON,
    }
    result = {
        "meta": {
            "dataset": DATASET_ID,
            "seed": args.seed,
            "rows": rows,
            "half_of": {str(r): h for r, h in half_of.items()},
            "excluded_prior_rows": sorted(prior_rows()),
            "n_texts": args.n_texts,
            "n_positions": args.n_positions,
            "positions_total": positions_used,
            "samples_per_half": [halves[0].n, halves[1].n],
            "skip_first": SKIP_FIRST,
            "max_seq_len": MAX_SEQ_LEN,
            "score_space": "pre-softmax logits (src/lens.py convention)",
            "npz": NPZ_PATH,
        },
        "gate": gate,
        "split_half": stats,
    }
    with open(JSON_PATH, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\nGATE {'PASS' if gate['pass'] else 'FAIL'}: "
          f"min J-lens split-half Pearson {gate['min_J_pearson']:.4f} "
          f"at layer {gate['argmin_layer']}")
    print(f"wrote {NPZ_PATH} and {JSON_PATH}")


if __name__ == "__main__":
    main()
