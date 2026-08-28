"""Step 8a/8b/8c: domain-stratified m_t + third disjoint sample + nulls (D42).

Five arms, processed sequentially, rows mutually disjoint and disjoint from
every prior sample (step-2 survey, step-3 delta_t, step-4 battery, fit rows):

  8a  four pile-10k domains via meta.pile_set_name -- "Pile-CC" (web prose),
      "Github" (code), "PubMed Abstracts" (academic), "Wikipedia (en)"
      (encyclopedic); kinds logit/J/R. Question: is m_t a property of the
      lens, or of the texts it was averaged over?
      Gate (D42): Pearson(m_t) >= 0.8 between EVERY domain pair at a
      majority of layers 0-30, J-lens -> "context-independent" stands.
  8b  a fifth fresh MIXED sample (seed 3): correlate with the original
      step-4 m_t. Same >= 0.8 bar, majority of layers, J-lens.
  8c  on the mixed arm only, two extra readout kinds accumulated in the
      same pass (steps.md registered null, comparative -- no gate):
        rot    final_norm(h @ Q.T) @ W_U.T, Q = seeded random rotation
               (seed 0, QR of Gaussian; null_baselines.py convention)
        shufJ  J-lens matrix from the WRONG layer: fixed derangement
               perm(l) = (l + 15) mod 31 (31 odd -> no fixed point)
      Question: does an arbitrary map through the same norm->W_U path
      produce an offset as STABLE (split-half Pearson) and as BIG
      (D39 size: median top-100 m_t minus vocab median) as the fitted J?

Every arm gets the step-4 split-half treatment (halves = disjoint document
sets) so cross-domain correlations can be read against each arm's own
noise ceiling. Halves stored fp16 (correlation-safe), combined fp32.

Run: .venv/bin/python -m src.domain_mt   (~1 h MPS; writes
results/step8_arm_<arm>.npz + results/step8_domain_mt.json)
"""

from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import pearsonr

from src.junk_survey import DATASET_ID, MAX_SEQ_LEN, N_FIT_ROWS, SKIP_FIRST
from src.lens import KINDS, Instrument
from src.offset_battery import Accumulator, pick_random_positions
from src.offset_battery import prior_rows as step23_rows

DOMAINS = ("Pile-CC", "Github", "PubMed Abstracts", "Wikipedia (en)")
MIXED_ARM = "mixed"
ARMS = DOMAINS + (MIXED_ARM,)
NULL_KINDS = ("rot", "shufJ")
N_TEXTS = 100
N_POSITIONS = 20
SEED_BASE = 3  # step-2 used 0, delta_t 1, step-4 battery 2
ROT_SEED = 0
SHUF_SHIFT = 15
GATE_MIN_PEARSON = 0.8
STEP4_NULL_JSON = "results/step4_noise_null.json"
STEP4_MT_NPZ = "results/step4_mt.npz"
OUT_JSON = "results/step8_domain_mt.json"


def arm_npz(arm: str) -> str:
    return f"results/step8_arm_{arm.replace(' ', '_').replace('(', '').replace(')', '')}.npz"


def all_prior_rows() -> set[int]:
    with open(STEP4_NULL_JSON) as f:
        step4 = set(json.load(f)["meta"]["rows"])
    return step23_rows() | step4 | set(range(N_FIT_ROWS))


def null_score(inst: Instrument, h: torch.Tensor, layer: int, kind: str,
               Q: torch.Tensor) -> torch.Tensor:
    h = h.to(inst.device).float()
    if kind == "rot":
        h = h @ Q.T
    elif kind == "shufJ":
        perm_layer = (layer + SHUF_SHIFT) % len(inst.source_layers)
        h = h @ inst.lenses["J"][perm_layer].T
    else:
        raise ValueError(kind)
    return inst.final_norm(h).float() @ inst.W_U.T


def run_arm(inst: Instrument, arm: str, rows: list[int], texts: list[str],
            seed: int) -> None:
    kinds = KINDS + (NULL_KINDS if arm == MIXED_ARM else ())
    layers = inst.source_layers
    vocab = inst.W_U.shape[0]
    g = torch.Generator().manual_seed(ROT_SEED)
    Q, _ = torch.linalg.qr(torch.randn(inst.W_U.shape[1], inst.W_U.shape[1], generator=g))
    Q = Q.to(inst.device)

    shuffled = rows.copy()
    random.Random(seed).shuffle(shuffled)
    half_of = {row: (0 if i < len(shuffled) // 2 else 1) for i, row in enumerate(shuffled)}
    halves = [Accumulator(len(kinds), len(layers), vocab) for _ in range(2)]

    pos_rng = random.Random(seed + 1000)
    for i, row in enumerate(rows):
        acc = halves[half_of[row]]
        acts, input_ids = inst.residuals(texts[row], layers, max_seq_len=MAX_SEQ_LEN)
        positions = pick_random_positions(input_ids.shape[1], N_POSITIONS, pos_rng)
        for layer_idx, layer in enumerate(layers):
            h = acts[layer][positions]
            for kind_idx, kind in enumerate(kinds):
                if kind in KINDS:
                    scores = inst.score(h, layer, kind)
                else:
                    scores = null_score(inst, h, layer, kind, Q)
                acc.add(kind_idx, layer_idx, scores.cpu().numpy())
        acc.n += len(positions)
        if (i + 1) % 10 == 0 or i + 1 == len(rows):
            print(f"[{arm}] {i + 1}/{len(rows)} docs", flush=True)

    combined = halves[0].total + halves[1].total
    n_total = halves[0].n + halves[1].n
    np.savez(
        arm_npz(arm),
        m_t=(combined / n_total).astype(np.float32),
        m_t_half0=halves[0].mean().astype(np.float16),
        m_t_half1=halves[1].mean().astype(np.float16),
        kinds=np.array(kinds),
        layers=np.array(layers),
        n_samples=np.array([halves[0].n, halves[1].n]),
        rows=np.array(rows),
    )
    print(f"[{arm}] wrote {arm_npz(arm)} ({n_total} positions)", flush=True)


def per_layer_pearson(a: np.ndarray, b: np.ndarray) -> list[float]:
    """a, b: [n_layers, vocab] -> per-layer Pearson r."""
    return [round(float(pearsonr(a[i].astype(np.float64),
                                 b[i].astype(np.float64))[0]), 4)
            for i in range(a.shape[0])]


def d39_size(m: np.ndarray) -> list[float]:
    """[n_layers, vocab] -> per-layer median(top-100 m_t) - median(vocab)."""
    out = []
    for i in range(m.shape[0]):
        v = m[i].astype(np.float64)
        top = np.sort(v)[-100:]
        out.append(round(float(np.median(top) - np.median(v)), 4))
    return out


def analyse() -> dict:
    arms = {arm: np.load(arm_npz(arm), allow_pickle=True) for arm in ARMS}
    step4 = np.load(STEP4_MT_NPZ, allow_pickle=True)
    step4_kinds = list(step4["kinds"])
    n_layers = len(arms[ARMS[0]]["layers"])

    def kind_slice(z, kind):
        return z["m_t"][list(z["kinds"]).index(kind)]

    out: dict = {"split_half_J": {}, "cross_domain_J": {}, "vs_step4_J": {},
                 "nulls": {}, "gates": {}}

    for arm, z in arms.items():
        ki = list(z["kinds"]).index("J")
        out["split_half_J"][arm] = per_layer_pearson(
            z["m_t_half0"][ki].astype(np.float32), z["m_t_half1"][ki].astype(np.float32))

    pair_majorities = {}
    for i, a in enumerate(DOMAINS):
        for b in DOMAINS[i + 1:]:
            rs = per_layer_pearson(kind_slice(arms[a], "J"), kind_slice(arms[b], "J"))
            key = f"{a} vs {b}"
            out["cross_domain_J"][key] = rs
            pair_majorities[key] = sum(r >= GATE_MIN_PEARSON for r in rs)

    step4_J = step4["m_t"][step4_kinds.index("J")]
    for arm in ARMS:
        out["vs_step4_J"][arm] = per_layer_pearson(kind_slice(arms[arm], "J"), step4_J)

    mixed = arms[MIXED_ARM]
    for kind in ("J", "R", "logit") + NULL_KINDS:
        ki = list(mixed["kinds"]).index(kind)
        out["nulls"][kind] = {
            "split_half": per_layer_pearson(
                mixed["m_t_half0"][ki].astype(np.float32),
                mixed["m_t_half1"][ki].astype(np.float32)),
            "d39_size": d39_size(mixed["m_t"][ki]),
        }

    maj = n_layers // 2 + 1
    out["gates"]["8a"] = {
        "criterion": (f"Pearson >= {GATE_MIN_PEARSON} between every domain pair "
                      f"at >= {maj}/{n_layers} layers, J-lens"),
        "layers_passing_per_pair": pair_majorities,
        "pass": all(v >= maj for v in pair_majorities.values()),
    }
    rs_8b = out["vs_step4_J"][MIXED_ARM]
    out["gates"]["8b"] = {
        "criterion": (f"Pearson(mixed m_t, step-4 m_t) >= {GATE_MIN_PEARSON} "
                      f"at >= {maj}/{n_layers} layers, J-lens"),
        "layers_passing": sum(r >= GATE_MIN_PEARSON for r in rs_8b),
        "pass": sum(r >= GATE_MIN_PEARSON for r in rs_8b) >= maj,
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-texts", type=int, default=N_TEXTS)
    parser.add_argument("--analyse-only", action="store_true")
    args = parser.parse_args()

    if not args.analyse_only:
        ds = load_dataset(DATASET_ID, split="train")
        texts = ds["text"]
        domain_of = [m["pile_set_name"] for m in ds["meta"]]
        used = all_prior_rows()
        inst = Instrument()

        arm_rows: dict[str, list[int]] = {}
        for arm_idx, arm in enumerate(ARMS):
            pool = [r for r in range(len(texts))
                    if r not in used and (arm == MIXED_ARM or domain_of[r] == arm)]
            rows = sorted(random.Random(SEED_BASE + arm_idx).sample(
                pool, min(args.n_texts, len(pool))))
            used |= set(rows)
            arm_rows[arm] = rows
            print(f"[{arm}] {len(rows)} rows", flush=True)

        for arm_idx, arm in enumerate(ARMS):
            path = arm_npz(arm)
            if os.path.exists(path):
                stored = np.load(path, allow_pickle=True)["rows"].tolist()
                if stored == arm_rows[arm]:
                    print(f"[{arm}] completed arm on disk, skipping", flush=True)
                    continue
            run_arm(inst, arm, arm_rows[arm], texts, SEED_BASE + arm_idx)

        meta = {"arms": {a: arm_rows[a] for a in ARMS}}
    else:
        meta = {"arms": "see arm npz files (analyse-only rerun)"}

    result = {"meta": {"dataset": DATASET_ID, "n_positions": N_POSITIONS,
                       "skip_first": SKIP_FIRST, "max_seq_len": MAX_SEQ_LEN,
                       "seed_base": SEED_BASE, "rot_seed": ROT_SEED,
                       "shufJ_shift": SHUF_SHIFT, **meta},
              **analyse()}
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    for g in ("8a", "8b"):
        print(f"GATE {g}: {'PASS' if result['gates'][g]['pass'] else 'FAIL'}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
