"""Fit a J-lens for Qwen3-1.7B (base for Cywinski's taboo adapters).

Same recipe as src/fit_pythia.py, which mirrors the released Qwen3.5-4B
lens provenance: NeelNanda/pile-10k docs in order, t_max=128, skip_first=4,
target = second-to-last layer, n_prompts=25. The lens is fitted on the
CLEAN base model, not the taboo fine-tunes: rank-16 LoRA barely perturbs
the network, and "lens fitted on the base, read out on the organism" is
the realistic monitoring setting the taboo experiment is meant to probe.

fp32 for the same reason as the Pythia fit (small offsets vs MPS half
precision). Layout auto-detects (Qwen3 is the standard `model` layout).

Examples:
  .venv/bin/python -m src.fit_qwen3 --smoke
  .venv/bin/python -m src.fit_qwen3 --n-prompts 25
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
from datasets import load_dataset
from jlens.fitting import fit
from jlens.hf import from_hf
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.fit_pythia import DATASET_ID, MAX_SEQ_LEN, N_PROMPTS, SKIP_FIRST

MODEL_ID = "Qwen/Qwen3-1.7B"
SMOKE_RESULT_PATH = "results/taboo/fit_smoke.json"


def load_lens_model(device: str):
    """Qwen3-1.7B in fp32, frozen, wrapped for the official tooling."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    hf_model.to(device).eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    return from_hf(hf_model, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="1-doc timing run")
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--out", default="results/taboo/qwen3_1.7b_jlens.pt")
    parser.add_argument("--checkpoint", default="results/taboo/qwen3_1.7b_fit_ckpt.pt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_lens_model(device)
    target_layer = model.n_layers - 2  # second-to-last, matches released recipe

    texts = [row["text"] for row in load_dataset(DATASET_ID, split="train")]
    n_docs = 1 if args.smoke else args.n_prompts
    prompts = texts[:n_docs]

    start = time.perf_counter()
    lens = fit(
        model,
        prompts,
        target_layer=target_layer,
        skip_first=SKIP_FIRST,
        max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=None if args.smoke else args.checkpoint,
    )
    elapsed = time.perf_counter() - start

    if args.smoke:
        result = {
            "model_id": MODEL_ID,
            "device": device,
            "n_layers": model.n_layers,
            "target_layer": target_layer,
            "skip_first": SKIP_FIRST,
            "max_seq_len": MAX_SEQ_LEN,
            "seconds_for_1_doc": round(elapsed, 1),
            "projected_minutes_25_docs": round(elapsed * 25 / 60, 1),
        }
        with open(SMOKE_RESULT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
        return

    lens.save(args.out)
    print(f"fitted {lens!r} on {n_docs} docs in {elapsed / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
