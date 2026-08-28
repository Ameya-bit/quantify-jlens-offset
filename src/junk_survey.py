"""Step 2 survey: top-k readouts across layers x positions x instruments.

Collects, for ~25 pile-10k texts, the top-10 readout tokens at several
positions and every lensed layer (0-30), through all three instruments
(J-lens, R-lens, logit-lens), plus junk flags per token. Raw evidence for
the step-2 setup figures; figures are derived from the JSON, never computed
fresh.

Text selection: seeded sample from pile-10k rows 25..9999. Rows 0..24 are
excluded because they are the fit corpus of BOTH the released Qwen lens
(provenance: docs_consumed=25 from row 0) and our Pythia fit (mirrored
recipe) -- reading out on the lens's own fit texts would be circular.

Positions: evenly spaced over [4, seq_len-1]; the first 4 positions are
skipped to match the lens fit (skip_first=4). Texts truncated to 128 tokens
to match the fit regime (t_max=128).

Run: .venv/bin/python -m src.junk_survey   (~5 min; writes
results/step2/step2_readouts.json)
"""

from __future__ import annotations

import argparse
import json
import random

from datasets import load_dataset

from src.flags import junk_fraction, token_flags
from src.lens import KINDS, Instrument

DATASET_ID = "NeelNanda/pile-10k"
N_FIT_ROWS = 25  # rows 0..24 reserved for lens fitting; never surveyed
SKIP_FIRST = 4
MAX_SEQ_LEN = 128
OUT_PATH = "results/step2/step2_readouts.json"


def pick_rows(n_texts: int, seed: int, n_rows_total: int) -> list[int]:
    rng = random.Random(seed)
    return sorted(rng.sample(range(N_FIT_ROWS, n_rows_total), n_texts))


def pick_positions(seq_len: int, n_positions: int) -> list[int]:
    """Evenly spaced positions in [SKIP_FIRST, seq_len - 1], deduplicated."""
    lo, hi = SKIP_FIRST, seq_len - 1
    if hi < lo:
        return []
    spaced = [round(lo + (hi - lo) * i / max(n_positions - 1, 1)) for i in range(n_positions)]
    return sorted(set(spaced))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-texts", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-positions", type=int, default=5)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args()

    texts = load_dataset(DATASET_ID, split="train")["text"]
    rows = pick_rows(args.n_texts, args.seed, len(texts))

    inst = Instrument()
    layers = inst.source_layers  # 0..30
    cells = []
    for row in rows:
        acts, input_ids = inst.residuals(texts[row], layers, max_seq_len=MAX_SEQ_LEN)
        seq_len = input_ids.shape[1]
        for pos in pick_positions(seq_len, args.n_positions):
            token_here = inst.tokenizer.decode([input_ids[0, pos].item()])
            for layer in layers:
                for kind in KINDS:
                    scores = inst.score(acts[layer][pos], layer, kind)
                    top = inst.top_tokens(scores, args.k)
                    top_strings = [t for t, _ in top]
                    cells.append(
                        {
                            "row": row,
                            "pos": pos,
                            "token_at_pos": token_here,
                            "layer": layer,
                            "kind": kind,
                            "top": [
                                {"t": t, "p": round(p, 5), **token_flags(t)}
                                for t, p in top
                            ],
                            "junk_fraction": junk_fraction(top_strings),
                        }
                    )
        print(f"row {row}: seq_len={seq_len}, cells so far {len(cells)}", flush=True)

    result = {
        "meta": {
            "dataset": DATASET_ID,
            "rows": rows,
            "excluded_fit_rows": N_FIT_ROWS,
            "seed": args.seed,
            "n_positions": args.n_positions,
            "k": args.k,
            "skip_first": SKIP_FIRST,
            "max_seq_len": MAX_SEQ_LEN,
            "layers": layers,
            "kinds": list(KINDS),
        },
        "cells": cells,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False)
    print(f"wrote {len(cells)} cells -> {args.out}")


if __name__ == "__main__":
    main()
