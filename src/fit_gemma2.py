"""Fit a J-lens for gemma-2-9b-it (base for Cywinski's headline taboo set).

Same released recipe as src/fit_pythia.py / src/fit_qwen3.py: pile-10k docs
in order, t_max=128, skip_first=4, target = second-to-last layer, n=25.
Fitted on the CLEAN google/gemma-2-9b-it base (gated: needs `hf auth login`
with an account that accepted the license), applied later to the LoRA-merged
taboo organisms — lens-on-base, readout-on-organism, as in D43/D44.

Gemma-2 notes: attn_implementation="eager" (softcapped attention; the
default sdpa path skips softcapping, and the fit differentiates THROUGH
attention, so eager is a correctness requirement, not a preference);
final logit softcapping is handled by jlens.hf natively. fp32 throughout,
same rationale as the other fits — sized for an 80GB CUDA card.

Examples (on the GPU box):
  python -m src.fit_gemma2 --smoke
  python -m src.fit_gemma2 --n-prompts 25
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

MODEL_ID = "google/gemma-2-9b-it"
SMOKE_RESULT_PATH = "results/taboo_gemma/fit_smoke.json"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_lens_model(device: str):
    """gemma-2-9b-it in fp32, eager attention, frozen, wrapped for jlens."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    )
    hf_model.to(device).eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    return from_hf(hf_model, tokenizer)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="1-doc timing run")
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--out", default="results/taboo_gemma/gemma2_9b_jlens.pt")
    parser.add_argument("--checkpoint", default="results/taboo_gemma/gemma2_9b_fit_ckpt.pt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    device = pick_device()
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
