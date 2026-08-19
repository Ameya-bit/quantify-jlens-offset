"""Step 1 sanity checks: is the instrument loaded correctly?

Three checks (steps.md, Step 1):
  1. Anchor: J[target_layer] and R[target_layer] must be ~identity
     (the artifact guarantees the target-layer map is exactly I).
  2. Late layers: J-lens readout ~ logit-lens readout (top-10 overlap).
     At the target layer they must be identical (overlap 10/10).
  3. Known facts: the expected answer token appears in the J-lens top-10
     by mid-depth (layer <= 24 of 32) at the last prompt position.
     Each prompt is first validated against the model's true final logits,
     separating "lens broken" from "model doesn't know".

Run:  .venv/bin/python -m src.sanity
Output: human-readable report + results/step1_sanity.json. Exit 1 on failure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from src.lens import Instrument

TARGET_LAYER = 30
ANCHOR_TOL = 1e-3
AGREEMENT_PROMPTS = [
    "The quick brown fox jumps over the lazy",
    "In 1969, humans first landed on the",
    "Water is made of hydrogen and",
]
AGREEMENT_LAYERS = [26, 28, 29, 30]
MIN_MEAN_OVERLAP_28_29 = 5.0  # of 10; layer 30 must be exactly 10
FACT_PROMPTS = [
    ("The Eiffel Tower is in the city of", " Paris"),
    ("The capital of Japan is", " Tokyo"),
    ("The chemical symbol for gold is", " Au"),
]
FACT_LAYERS = [8, 12, 16, 20, 24, 28]
MID_DEPTH = 24
TOP_K = 10


def check_anchor(inst: Instrument) -> dict:
    """Max |M - I| at the target layer for both lenses."""
    out = {}
    eye = torch.eye(inst.lm.d_model, device=inst.device)
    for kind in ("J", "R"):
        diff = (inst.lenses[kind][TARGET_LAYER] - eye).abs().max().item()
        out[kind] = {"max_abs_diff_from_I": diff, "pass": diff < ANCHOR_TOL}
    return out


def _topk_ids(scores: torch.Tensor, k: int = TOP_K) -> set[int]:
    return set(scores.topk(k).indices.tolist())


def check_agreement(inst: Instrument) -> dict:
    """Top-10 overlap between J-lens and logit-lens at late layers,
    last position of each prompt."""
    rows = []
    for prompt in AGREEMENT_PROMPTS:
        acts, _ = inst.residuals(prompt, AGREEMENT_LAYERS)
        row = {"prompt": prompt, "overlap_by_layer": {}}
        for layer in AGREEMENT_LAYERS:
            h = acts[layer][-1]
            overlap = len(
                _topk_ids(inst.score(h, layer, "J"))
                & _topk_ids(inst.score(h, layer, "logit"))
            )
            row["overlap_by_layer"][layer] = overlap
        rows.append(row)
    at_target = [r["overlap_by_layer"][TARGET_LAYER] for r in rows]
    late = [r["overlap_by_layer"][l] for r in rows for l in (28, 29)]
    mean_late = sum(late) / len(late)
    return {
        "rows": rows,
        "target_layer_all_exact": all(o == TOP_K for o in at_target),
        "mean_overlap_28_29": mean_late,
        "pass": all(o == TOP_K for o in at_target)
        and mean_late >= MIN_MEAN_OVERLAP_28_29,
    }


def check_facts(inst: Instrument) -> dict:
    """Expected answer token in J-lens top-10 by mid-depth, last position."""
    rows = []
    for prompt, answer in FACT_PROMPTS:
        answer_id = inst.tokenizer.encode(answer)[0]
        # Validate the prompt against the model's true final logits first.
        input_ids = inst.lm.encode(prompt)
        final_logits = inst.model(input_ids=input_ids).logits[0, -1].float()
        model_top5 = final_logits.topk(5).indices.tolist()
        model_knows = answer_id in model_top5

        acts, _ = inst.residuals(prompt, FACT_LAYERS)
        found_at, tops = None, {}
        for layer in FACT_LAYERS:
            scores = inst.score(acts[layer][-1], layer, "J")
            tops[layer] = [t for t, _ in inst.top_tokens(scores, TOP_K)]
            if found_at is None and answer_id in _topk_ids(scores):
                found_at = layer
        rows.append(
            {
                "prompt": prompt,
                "answer": answer,
                "model_knows_top5": model_knows,
                "jlens_found_at_layer": found_at,
                "jlens_top10_by_layer": tops,
                "pass": model_knows and found_at is not None and found_at <= MID_DEPTH,
            }
        )
    n_pass = sum(r["pass"] for r in rows)
    return {"rows": rows, "n_pass": n_pass, "pass": n_pass >= 2}


def main() -> int:
    inst = Instrument()
    print(f"Loaded {inst.lm!r} on {inst.device}")
    for kind in ("J", "R"):
        print(f"{kind}-lens provenance: {inst.provenance[kind]}")

    results = {
        "provenance": inst.provenance,
        "anchor": check_anchor(inst),
        "agreement": check_agreement(inst),
        "facts": check_facts(inst),
    }
    results["all_pass"] = all(
        results[c]["pass"] if c == "agreement" else all_pass(results[c])
        for c in ("anchor", "agreement", "facts")
    )

    Path("results").mkdir(exist_ok=True)
    with open("results/step1_sanity.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n=== Step 1 sanity report ===")
    a = results["anchor"]
    print(
        f"1. Anchor (layer {TARGET_LAYER}): "
        f"J max|J-I|={a['J']['max_abs_diff_from_I']:.2e} "
        f"({'PASS' if a['J']['pass'] else 'FAIL'}); "
        f"R max|R-I|={a['R']['max_abs_diff_from_I']:.2e} "
        f"({'PASS' if a['R']['pass'] else 'FAIL'})"
    )
    g = results["agreement"]
    for row in g["rows"]:
        print(f"2. overlap {row['overlap_by_layer']}  <- {row['prompt']!r}")
    print(
        f"2. J~logit late-layer agreement: target exact={g['target_layer_all_exact']}, "
        f"mean@28-29={g['mean_overlap_28_29']:.1f}/10 "
        f"({'PASS' if g['pass'] else 'FAIL'})"
    )
    for row in results["facts"]["rows"]:
        print(
            f"3. {row['prompt']!r} -> {row['answer']!r}: "
            f"model_knows={row['model_knows_top5']}, "
            f"J-lens top-10 hit at layer {row['jlens_found_at_layer']} "
            f"({'PASS' if row['pass'] else 'FAIL'})"
        )
    print(
        f"3. facts: {results['facts']['n_pass']}/{len(FACT_PROMPTS)} "
        f"({'PASS' if results['facts']['pass'] else 'FAIL'})"
    )
    print(f"\nALL {'PASS' if results['all_pass'] else 'FAIL'}")
    return 0 if results["all_pass"] else 1


def all_pass(section: dict) -> bool:
    """A section passes if its own 'pass' or every sub-entry's 'pass' is True."""
    if "pass" in section:
        return bool(section["pass"])
    return all(v.get("pass", False) for v in section.values() if isinstance(v, dict))


if __name__ == "__main__":
    sys.exit(main())
