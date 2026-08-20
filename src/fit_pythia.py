"""Fit a J-lens for Pythia-1.4B with the official `jlens.fit`.

Recipe mirrors the released Qwen3.5-4B lens artifact's provenance exactly:
same dataset (NeelNanda/pile-10k, docs consumed in order from row 0), same
truncation (t_max=128), same skip_first=4, target = second-to-last layer
(Qwen: 30 of 32 -> Pythia: 22 of 24), n_prompts=25 (or fewer via CLI).

Precision: the model runs fp32 (the study hunts a small offset; MPS
half-precision could eat it). Parameters are frozen -- the official
ActivationRecorder then builds the graph only from the earliest source
layer onward, which is both supported and cheaper. The final artifact is
saved fp16 via JacobianLens.save, matching the released lens format; the
fp32 running checkpoint stays on disk too (*.pt is gitignored).

Smoke mode (--smoke) fits on the single first doc and writes timing +
full-run projections to results/step2_pythia_smoke.json so we can decide
n before committing to the real run.

Examples:
  .venv/bin/python -m src.fit_pythia --smoke
  .venv/bin/python -m src.fit_pythia --n-prompts 25
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import torch
from datasets import load_dataset
from jlens.fitting import fit
from jlens.hf import Layout, from_hf
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "EleutherAI/pythia-1.4b"
DATASET_ID = "NeelNanda/pile-10k"
SKIP_FIRST = 4  # released Qwen lens provenance: skip_first=4
MAX_SEQ_LEN = 128  # provenance: t_max=128
N_PROMPTS = 25  # provenance: n_prompts=25
SMOKE_RESULT_PATH = "results/step2_pythia_smoke.json"


def load_lens_model(device: str):
    """Pythia-1.4B in fp32, frozen, wrapped for the official tooling."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    hf_model.to(device).eval()
    for p in hf_model.parameters():
        p.requires_grad_(False)
    # The official GPT-NeoX layout predates transformers 5.x, which renamed
    # the output head embed_out -> lm_head; pass the layout explicitly.
    layout = Layout("gpt_neox", norm="final_layer_norm", embed="embed_in")
    return from_hf(hf_model, tokenizer, layout=layout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="1-doc timing run")
    parser.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    parser.add_argument("--out", default="results/pythia_jlens.pt")
    parser.add_argument("--checkpoint", default="results/pythia_fit_ckpt.pt")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s"
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_lens_model(device)
    target_layer = model.n_layers - 2  # second-to-last, like released target 30/32

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
            "target_layer": target_layer,
            "skip_first": SKIP_FIRST,
            "max_seq_len": MAX_SEQ_LEN,
            "seconds_for_1_doc": round(elapsed, 1),
            "projected_minutes_10_docs": round(elapsed * 10 / 60, 1),
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
