"""Sanity battery for OUR self-fitted Qwen3-1.7B J-lens (taboo-arm gate).

Mirror of src.sanity_pythia for the taboo experiment's base lens: the lens
came from our own `src.fit_qwen3` run, so it must pass the same style of
checks as step 1 before src.taboo_eval is allowed to use it. The artifact
stores layers 0-25 (destination layer 26); the wiring check stands in for
the missing anchor-identity assertion, exactly as on Pythia.

Checks (same gates as sanity_pythia, layer indices shifted for 28 layers):
  1. Wiring: final-block logit-lens readout == the model's true next-token
     logits (top-10 set match), fp32 throughout.
  2. Late-layer agreement: J-lens vs logit-lens top-10 overlap at the last
     fitted layers (23-25), 25 seeded pile texts x 5 positions (125
     samples/layer). Gate: mean overlap at layer 25 >= 5/10.
  3. Known facts: answer token in the J-lens top-10 by some fitted layer
     for >= 2/3 easy prompts; logit-lens first-hit layer recorded as the
     baseline.

Run: .venv/bin/python -m src.sanity_qwen3   (writes results/taboo/qwen3_sanity.json)
"""

from __future__ import annotations

import json

import torch
from datasets import load_dataset
from jlens.hooks import ActivationRecorder

from src.fit_pythia import DATASET_ID
from src.fit_qwen3 import MODEL_ID, load_lens_model
from src.junk_survey import pick_positions, pick_rows

LENS_PATH = "results/taboo/qwen3_1.7b_jlens.pt"
OUT_PATH = "results/taboo/qwen3_sanity.json"
AGREEMENT_LAYERS = [23, 24, 25]
AGREEMENT_N_TEXTS = 25
AGREEMENT_SEED = 0
AGREEMENT_N_POSITIONS = 5
AGREEMENT_GATE = 5.0  # mean top-10 overlap at the last fitted layer
GATE_LAYER = 25
FACTS = [
    ("The Eiffel Tower is located in the city of", "paris"),
    ("The capital city of Japan is", "tokyo"),
    ("Water is made of hydrogen and", "oxygen"),
]


def top10(scores: torch.Tensor, tokenizer) -> list[str]:
    return [tokenizer.decode([i]) for i in scores.topk(10).indices.tolist()]


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    lm = load_lens_model(device)
    ckpt = torch.load(LENS_PATH, map_location="cpu", weights_only=True)
    J = {layer: m.float().to(device) for layer, m in ckpt["J"].items()}
    fitted_layers = sorted(J)
    record: dict = {"lens_path": LENS_PATH, "model_id": MODEL_ID, "checks": {}}

    def residuals(text: str, layers: list[int]):
        input_ids = lm.encode(text, max_length=128)
        with ActivationRecorder(lm.layers, at=sorted(set(layers))) as rec:
            lm.forward(input_ids)
            return {l: rec.activations[l][0].detach() for l in set(layers)}, input_ids

    def score(h: torch.Tensor, layer: int | None) -> torch.Tensor:
        if layer is not None:
            h = h @ J[layer].T
        return lm.unembed(h)

    # --- Check 1: wiring ---
    text = "The quick brown fox jumps over the lazy dog because it"
    acts, input_ids = residuals(text, [lm.n_layers - 1])
    lens_top = top10(score(acts[lm.n_layers - 1][-1], None), lm.tokenizer)
    with torch.no_grad():
        true_logits = lm._hf_model(input_ids).logits[0, -1]
    true_top = top10(true_logits, lm.tokenizer)
    wiring_ok = set(lens_top) == set(true_top)
    record["checks"]["wiring"] = {
        "pass": wiring_ok, "lens_top10": lens_top, "true_top10": true_top,
    }
    print(f"1. wiring (final-block logit-lens == true logits): {'PASS' if wiring_ok else 'FAIL'}")

    # --- Check 2: late-layer agreement ---
    texts = load_dataset(DATASET_ID, split="train")["text"]
    rows = pick_rows(AGREEMENT_N_TEXTS, AGREEMENT_SEED, len(texts))
    overlaps: dict[int, list[int]] = {l: [] for l in AGREEMENT_LAYERS}
    for row in rows:
        acts, input_ids = residuals(texts[row], AGREEMENT_LAYERS)
        for pos in pick_positions(input_ids.shape[1], AGREEMENT_N_POSITIONS):
            for layer in AGREEMENT_LAYERS:
                h = acts[layer][pos]
                j_top = set(top10(score(h, layer), lm.tokenizer))
                logit_top = set(top10(score(h, None), lm.tokenizer))
                overlaps[layer].append(len(j_top & logit_top))
    means = {l: sum(v) / len(v) for l, v in overlaps.items()}
    n_samples = len(overlaps[AGREEMENT_LAYERS[0]])
    agree_ok = means[GATE_LAYER] >= AGREEMENT_GATE
    record["checks"]["late_agreement"] = {
        "pass": agree_ok, "rows": rows, "n_samples_per_layer": n_samples,
        "mean_overlap_by_layer": {str(l): round(m, 2) for l, m in means.items()},
        "gate": f"mean overlap at layer {GATE_LAYER} >= {AGREEMENT_GATE}",
    }
    print(f"2. late-layer J~logit agreement (n={n_samples}) "
          f"{ {l: round(m, 2) for l, m in means.items()} }: {'PASS' if agree_ok else 'FAIL'}")

    # --- Check 3: known facts through the J-lens ---
    hits = []
    for prompt, answer in FACTS:
        acts, _ = residuals(prompt, fitted_layers)
        found_layer = None
        logit_first = None
        for layer in fitted_layers:
            h = acts[layer][-1]
            if found_layer is None:
                tops = top10(score(h, layer), lm.tokenizer)
                if any(t.strip().lower() == answer for t in tops):
                    found_layer = layer
            if logit_first is None:
                tops = top10(score(h, None), lm.tokenizer)
                if any(t.strip().lower() == answer for t in tops):
                    logit_first = layer
        hits.append({"prompt": prompt, "answer": answer,
                     "first_layer": found_layer, "logit_first_layer": logit_first})
        print(f"   {answer!r}: first hit J-lens L{found_layer} vs logit-lens L{logit_first}")
    n_hits = sum(h["first_layer"] is not None for h in hits)
    facts_ok = n_hits >= 2
    record["checks"]["known_facts"] = {"pass": facts_ok, "hits": hits, "gate": ">=2/3"}
    print(f"3. known facts: {n_hits}/3: {'PASS' if facts_ok else 'FAIL'}")

    record["all_pass"] = wiring_ok and agree_ok and facts_ok
    with open(OUT_PATH, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nALL {'PASS' if record['all_pass'] else 'NOT PASSING'} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
