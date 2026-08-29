"""D44 registered readout of results/taboo_gemma/taboo_eval.json.

Same battery as src.taboo_analysis (Qwen arm), Gemma-sized: 20 words, 40
fitted layers, P2 late band = last five fitted layers (L35-39), plus the
two D44 additions:
  - Reproduction check: our raw/logit at layer index 31 (their layer
    choice) vs their published logit-lens table (top-5: 35.0% accuracy,
    75.0% pass@10, 20.0% majority) -- a check on the reimplementation,
    not a gate; large disagreement gets diagnosed before claims.
  - P3 (REGISTERED at n=20): Spearman rho(zipf(word), per-word accuracy
    at the LOO-selected layer, zscore/J) > 0, zipf via wordfreq (en).

Run: .venv/bin/python -m src.taboo_analysis_gemma
     (writes results/taboo_gemma/taboo_summary.json)
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr
from wordfreq import zipf_frequency

from src.taboo_eval_gemma import PLURALS, WORDS

IN_PATH = "results/taboo_gemma/taboo_eval.json"
OUT_PATH = "results/taboo_gemma/taboo_summary.json"
TOP_K = 5
REPRO_LAYER = 31  # their empirically-chosen layer (0-indexed), for raw/logit
PUBLISHED = {"accuracy": 0.35, "pass@10": 0.75, "majority@10": 0.20}  # their top-5 row


def accuracy(preds_per_prompt, word, k=TOP_K):
    valid = [f.lower() for f in PLURALS[word]]
    hits = sum(
        any(p.strip().lower() in valid for p in preds[:k]) for preds in preds_per_prompt
    )
    return hits / len(preds_per_prompt)


def pass_at(preds_per_prompt, word, k=TOP_K):
    return int(accuracy(preds_per_prompt, word, k) > 0)


def majority(preds_per_prompt, word, k=TOP_K):
    valid = [f.lower() for f in PLURALS[word]]
    counts: dict[str, int] = {}
    for preds in preds_per_prompt:
        for p in preds[:k]:
            key = p.strip().lower()
            counts[key] = counts.get(key, 0) + 1
    top = max(counts, key=counts.get) if counts else ""
    return int(top in valid)


def main() -> None:
    d = json.load(open(IN_PATH))
    words = [w for w in WORDS if w in d["words"]]
    keys = list(d["words"][words[0]]["top_tokens"])
    variants = sorted({k.split("/")[0] for k in keys})
    kinds = sorted({k.split("/")[1] for k in keys})
    layers = sorted({int(k.split("/L")[1]) for k in keys})
    late_band = layers[-5:]

    def preds(word, variant, kind, layer):
        return d["words"][word]["top_tokens"][f"{variant}/{kind}/L{layer}"]

    out: dict = {"top_k": TOP_K, "n_words": len(words), "layers": layers}

    # Reproduction check (their method, their layer, their metrics)
    repro = {
        "layer_index": REPRO_LAYER,
        "ours": {
            "accuracy": round(float(np.mean(
                [accuracy(preds(w, "raw", "logit", REPRO_LAYER), w) for w in words])), 4),
            "pass@10": round(float(np.mean(
                [pass_at(preds(w, "raw", "logit", REPRO_LAYER), w) for w in words])), 4),
            "majority@10": round(float(np.mean(
                [majority(preds(w, "raw", "logit", REPRO_LAYER), w) for w in words])), 4),
        },
        "published": PUBLISHED,
    }
    out["reproduction_check"] = repro

    # Full sweep
    sweep: dict = {}
    for v in variants:
        for kd in kinds:
            sweep[f"{v}/{kd}"] = {
                str(l): {
                    "accuracy": round(float(np.mean(
                        [accuracy(preds(w, v, kd, l), w) for w in words])), 4),
                    "pass@10": round(float(np.mean(
                        [pass_at(preds(w, v, kd, l), w) for w in words])), 4),
                }
                for l in layers
            }
    out["sweep"] = sweep

    # LOO headline + per-word accuracy at LOO layer (feeds P3)
    loo: dict = {}
    loo_word_acc: dict[str, dict[str, float]] = {}
    for v in variants:
        for kd in kinds:
            per_word = []
            for held in words:
                rest = [w for w in words if w != held]
                best_l = max(
                    layers,
                    key=lambda l: (np.mean([accuracy(preds(w, v, kd, l), w) for w in rest]), l),
                )
                per_word.append({
                    "held_out": held,
                    "layer": best_l,
                    "accuracy": accuracy(preds(held, v, kd, best_l), held),
                    "pass@10": pass_at(preds(held, v, kd, best_l), held),
                })
            loo[f"{v}/{kd}"] = {
                "per_word": per_word,
                "headline_accuracy": round(float(np.mean([r["accuracy"] for r in per_word])), 4),
                "headline_pass@10": round(float(np.mean([r["pass@10"] for r in per_word])), 4),
            }
            loo_word_acc[f"{v}/{kd}"] = {r["held_out"]: r["accuracy"] for r in per_word}
    out["loo_headline"] = loo

    # P1: best sweep layer, J vs logit
    p1: dict = {}
    for v in variants:
        row = {}
        for kd in kinds:
            accs = {l: np.mean([accuracy(preds(w, v, kd, l), w) for w in words]) for l in layers}
            best_l = max(layers, key=lambda l: (accs[l], -l))
            row[kd] = {"best_layer": best_l, "best_mean_accuracy": round(float(accs[best_l]), 4)}
        row["verdict_J_geq_logit"] = bool(
            row["J"]["best_mean_accuracy"] >= row["logit"]["best_mean_accuracy"])
        row["verdict_J_earlier"] = bool(row["J"]["best_layer"] < row["logit"]["best_layer"])
        p1[v] = row
    out["P1"] = p1

    # P2: zscore - raw on J in the late band
    p2_layers = {}
    for l in layers:
        z = np.mean([accuracy(preds(w, "zscore", "J", l), w) for w in words])
        r = np.mean([accuracy(preds(w, "raw", "J", l), w) for w in words])
        p2_layers[str(l)] = {"zscore": round(float(z), 4), "raw": round(float(r), 4),
                             "delta": round(float(z - r), 4)}
    band = [p2_layers[str(l)]["delta"] for l in late_band]
    out["P2"] = {
        "by_layer": p2_layers,
        "registered_band": f"L{late_band[0]}-L{late_band[-1]}",
        "band_deltas": band,
        "verdict_z_geq_raw_in_band": bool(all(x >= 0 for x in band)),
        "band_mean_delta": round(float(np.mean(band)), 4),
    }

    # P3 (registered): rho(zipf, LOO accuracy) on zscore/J; other cells context
    zipfs = {w: zipf_frequency(w, "en") for w in words}
    p3: dict = {"zipf": zipfs, "cells": {}}
    for cell, accs_by_word in loo_word_acc.items():
        xs = [zipfs[w] for w in words]
        ys = [accs_by_word[w] for w in words]
        rho, p = spearmanr(xs, ys)
        p3["cells"][cell] = {"rho": round(float(rho), 4), "p": round(float(p), 4)}
    p3["registered_cell"] = "zscore/J"
    p3["verdict_rho_positive"] = bool(p3["cells"]["zscore/J"]["rho"] > 0)
    out["P3"] = p3

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("reproduction (raw/logit L31):", repro["ours"], "vs published", PUBLISHED)
    print("LOO headlines (acc / pass@10):")
    for k, r in loo.items():
        print(f"  {k}: {r['headline_accuracy']:.3f} / {r['headline_pass@10']:.3f}")
    print("P1:", {v: (p1[v]["verdict_J_geq_logit"], p1[v]["verdict_J_earlier"],
                      p1[v]["J"]["best_layer"], p1[v]["logit"]["best_layer"]) for v in variants})
    print("P2 band:", out["P2"]["band_deltas"], "->", out["P2"]["verdict_z_geq_raw_in_band"])
    print("P3:", p3["cells"], "-> registered zscore/J positive:", p3["verdict_rho_positive"])
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
