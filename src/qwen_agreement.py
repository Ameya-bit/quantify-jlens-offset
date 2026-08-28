"""Wide-sample Qwen J~logit agreement check, derived from the step-2 survey.

Retroactively upgrades step 1's 3-prompt late-layer agreement check to the
same 125-sample standard as the (widened) Pythia battery: for every surveyed
(text, position), top-10 overlap between the J-lens and logit-lens readouts
at the same layer. No forward passes -- pure re-read of the committed
results/step2/step2_readouts.json.

Run: .venv/bin/python -m src.qwen_agreement  (writes results/step2/qwen_agreement.json)
"""

from __future__ import annotations

import json
from collections import defaultdict

READOUTS_PATH = "results/step2/step2_readouts.json"
OUT_PATH = "results/step2/qwen_agreement.json"
LATE_LAYERS = [26, 28, 29, 30]  # step 1's AGREEMENT_LAYERS


def main() -> None:
    with open(READOUTS_PATH) as f:
        data = json.load(f)
    tops = {
        (c["kind"], c["layer"], c["row"], c["pos"]): {t["t"] for t in c["top"]}
        for c in data["cells"]
    }
    by_layer = defaultdict(list)
    for (kind, layer, row, pos), j_top in tops.items():
        if kind != "J":
            continue
        by_layer[layer].append(len(j_top & tops[("logit", layer, row, pos)]))
    means = {l: sum(v) / len(v) for l, v in sorted(by_layer.items())}
    n = len(by_layer[30])
    print(f"Qwen J~logit top-10 overlap, {n} samples/layer:")
    for l, m in means.items():
        marker = "  <-- step-1 late layer" if l in LATE_LAYERS else ""
        print(f"  L{l:>2}: {m:5.2f}{marker}")
    record = {
        "source": READOUTS_PATH,
        "n_samples_per_layer": n,
        "mean_overlap_by_layer": {str(l): round(m, 2) for l, m in means.items()},
        "step1_late_layers": LATE_LAYERS,
        "note": "step-1 gate analog: mean over layers 28-29 >= 5; layer 30 must be 10",
        "mean_28_29": round((means[28] + means[29]) / 2, 2),
        "layer_30_exact_10": means[30] == 10.0,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
