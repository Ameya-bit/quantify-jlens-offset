"""Load Qwen3.5-4B + released J/R lenses; unified score() readout.

Conventions (matched to official `anthropics/jacobian-lens` tooling, which
fitted the released artifacts):
  - residual "at layer l" = output of decoder block l (forward hook on
    model.layers[l], via jlens.hooks.ActivationRecorder)
  - transport: J_l @ h, implemented as h @ J_l.T
  - readout: W_U @ finalnorm(transported), then softmax only for display

Deliberate deviation from official tooling: the official HFLensModel.unembed
casts the residual to the model dtype (bf16) before norm+unembed. Here the
norm -> unembed path runs in fp32 even though the model is bf16, because the
stable offset we are hunting is small and MPS bf16 could eat it
(mats plan: "fp32 for norm->unembed on MPS").

"score" = pre-softmax logits s(h) = W_U @ finalnorm(J_l @ h). All later
averaging (m_t, sigma_t) happens in this score space.
"""

from __future__ import annotations

import torch
from huggingface_hub import hf_hub_download
from jlens.hf import from_hf
from jlens.hooks import ActivationRecorder
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3.5-4B"
LENS_REPO = "camilablank/workspace-lenses"
LENS_FILES = {"J": "qwen3.5-4b/j-lens/lens.pt", "R": "qwen3.5-4b/r-lens/lens.pt"}
KINDS = ("logit", "J", "R")
DEFAULT_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_lens_matrices(kind: str, device: str) -> tuple[dict[int, torch.Tensor], dict]:
    """Download (or hit cache for) a released lens; return fp32 per-layer
    matrices on `device` plus the artifact's provenance dict."""
    path = hf_hub_download(LENS_REPO, LENS_FILES[kind])
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    matrices = {layer: m.float().to(device) for layer, m in ckpt["J"].items()}
    return matrices, ckpt.get("provenance", {})


class Instrument:
    """Qwen3.5-4B + both released lenses + one shared readout path."""

    def __init__(self, model_id: str = MODEL_ID, device: str = DEFAULT_DEVICE):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        model.to(device).eval()
        for p in model.parameters():
            p.requires_grad_(False)

        # Official wrapper: auto-locates the text decoder / layers / norm /
        # lm_head inside the (multimodal-wrapped) HF model.
        self.model = model  # full HF model (for true final logits in checks)
        self.lm = from_hf(model, self.tokenizer)
        layout = self.lm.layout
        text_module = model
        for attr in layout.path.split("."):
            text_module = getattr(text_module, attr)
        self.final_norm = getattr(text_module, layout.norm)
        # fp32 copy of the unembedding for the readout matmul (~2.5 GB).
        self.W_U = getattr(model, layout.lm_head).weight.detach().float().to(device)

        self.lenses: dict[str, dict[int, torch.Tensor]] = {}
        self.provenance: dict[str, dict] = {}
        for kind in ("J", "R"):
            self.lenses[kind], self.provenance[kind] = load_lens_matrices(kind, device)

        self.n_layers = self.lm.n_layers  # 32 blocks; lenses cover 0..30
        self.source_layers = sorted(self.lenses["J"])

    @torch.no_grad()
    def residuals(
        self, prompt: str, layers: list[int], max_seq_len: int = 512
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
        """Forward pass; return {layer: [seq_len, d_model]} residuals
        (output of each requested block) and the input_ids [1, seq_len]."""
        wanted = sorted(set(layers))
        input_ids = self.lm.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(self.lm.layers, at=wanted) as recorder:
            self.lm.forward(input_ids)
            acts = {l: recorder.activations[l][0].detach() for l in wanted}
        return acts, input_ids

    @torch.no_grad()
    def score(self, h: torch.Tensor, layer: int, kind: str) -> torch.Tensor:
        """s(h) for kind in {logit, J, R}: pre-softmax logits [..., vocab].

        One shared code path: the only difference between kinds is which
        (if any) transport matrix is applied before finalnorm -> W_U.
        norm -> unembed runs in fp32 (see module docstring).
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        h = h.to(self.device).float()
        if kind != "logit":
            if layer not in self.lenses[kind]:
                raise ValueError(
                    f"layer {layer} not fitted; lens covers {self.source_layers[0]}"
                    f"..{self.source_layers[-1]}"
                )
            h = h @ self.lenses[kind][layer].T
        normed = self.final_norm(h).float()  # RMSNorm upcasts internally; keep fp32
        return normed @ self.W_U.T

    def top_tokens(self, scores: torch.Tensor, k: int = 10) -> list[tuple[str, float]]:
        """Top-k (token_string, probability) for a single score vector."""
        probs = torch.softmax(scores.float(), dim=-1)
        vals, ids = probs.topk(k)
        return [
            (self.tokenizer.decode([tid]), val.item())
            for tid, val in zip(ids.tolist(), vals)
        ]
