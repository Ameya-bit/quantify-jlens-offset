"""Step 8d: frequency slice of the step-7 headline (registered D42).

Attack on step 7's finding 1 ("the offset is mostly genuine signal at
working depths"): if subtracting m_t hurts because the frequency component
of the offset is real signal, the damage should scale with how frequent the
intermediate token is. Re-slices step7_ranks.npz by the wordfreq zipf of
each item's intermediate; no forward passes.

Damage per item x layer x instrument = log2(rank_corrected / rank_raw)
(positive = correction made the rank worse). Aggregated to the 25 distinct
intermediates by median over items and over the band, deliberately
collapsing template repeats (the step-7 independence caveat this
red-teams). Bands: working depths L17-23 (primary), early L0-4 (context).

Registered prediction (D42): Spearman rho(zipf, damage) > 0 over the 25
intermediates at working depths on J and R. Break criterion: the rare half
(bottom 12 by zipf) shows median damage < 0 (correction helps) at working
depths on J or R.

Run: .venv/bin/python -m src.freq_slice   (seconds; writes
results/step8/step8_freq_slice.json + step8_freq_slice.png)
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

RANKS_NPZ = "results/step7/step7_ranks.npz"
TWOHOP_JSON = "results/step3/step3_twohop.json"
WORDFREQ_NPZ = "results/step3/step3_wordfreq.npz"
OUT_JSON = "results/step8/step8_freq_slice.json"
OUT_PNG = "results/step8/step8_freq_slice.png"

BANDS = {"working_L17_23": range(17, 24), "early_L0_4": range(0, 5)}
PRIMARY_BAND = "working_L17_23"
KINDS = ("logit", "J", "R")


def load_items() -> list[dict]:
    with open(TWOHOP_JSON) as f:
        data = json.load(f)
    return next(v for v in data.values()
                if isinstance(v, list) and v and isinstance(v[0], dict))


def main() -> None:
    z = np.load(RANKS_NPZ, allow_pickle=True)
    assert tuple(z["kinds"]) == KINDS
    raw = z["intermediate_raw"].astype(np.float64)        # [138, 31, 3]
    corrected = z["intermediate_corrected"].astype(np.float64)
    headline = z["headline"].astype(bool)
    item_ids = list(z["item_ids"])

    items = {it["id"]: it for it in load_items()}
    zipf_all = np.load(WORDFREQ_NPZ)["wordfreq_zipf"]
    inter_of = [items[i]["intermediate"] for i in item_ids]
    zipf_of = [float(zipf_all[items[i]["intermediate_id"]]) for i in item_ids]

    damage = np.log2(corrected / raw)                      # [138, 31, 3]

    out: dict = {
        "design": {
            "registered": "D42 (pre-code, 27 Aug)",
            "damage": "log2(rank_corrected / rank_raw); positive = correction hurts",
            "aggregation": "median over items sharing an intermediate, then over band layers",
            "prediction": "Spearman rho(zipf, damage) > 0 at working depths, J and R",
            "break_criterion": ("rare half (bottom 12 of 25 by zipf) median damage < 0 "
                                "at working depths on J or R"),
        },
        "intermediates": {},
        "results": {},
    }

    for subset_name, mask in (("headline_72", headline),
                              ("all_138", np.ones(len(item_ids), bool))):
        subset_res: dict = {}
        for band_name, band in BANDS.items():
            band_idx = list(band)
            per_inter: dict[str, dict] = {}
            for n, inter in enumerate(inter_of):
                if not mask[n]:
                    continue
                rec = per_inter.setdefault(inter, {"zipf": zipf_of[n], "rows": []})
                rec["rows"].append(damage[n][band_idx])   # [band, 3]
            names = sorted(per_inter)
            zipfs = np.array([per_inter[i]["zipf"] for i in names])
            # median over (items x band layers) per intermediate -> [n_inter, 3]
            med = np.array([np.median(np.stack(per_inter[i]["rows"]), axis=(0, 1))
                            for i in names])
            order = np.argsort(zipfs)
            rare_half = order[: len(order) // 2 + len(order) % 2]
            freq_half = order[len(rare_half):]
            band_res: dict = {"n_intermediates": len(names)}
            for ki, kind in enumerate(KINDS):
                rho, p = spearmanr(zipfs, med[:, ki])
                band_res[kind] = {
                    "spearman_rho": round(float(rho), 3),
                    "spearman_p": float(p),
                    "median_damage_rare_half": round(float(np.median(med[rare_half, ki])), 3),
                    "median_damage_freq_half": round(float(np.median(med[freq_half, ki])), 3),
                }
            subset_res[band_name] = band_res
            if subset_name == "headline_72" and band_name == PRIMARY_BAND:
                out["intermediates"] = {
                    names[i]: {"zipf": round(float(zipfs[i]), 2),
                               **{k: round(float(med[i, ki]), 3)
                                  for ki, k in enumerate(KINDS)}}
                    for i in order}
        out["results"][subset_name] = subset_res

    prim = out["results"]["headline_72"][PRIMARY_BAND]
    out["verdicts"] = {
        "prediction_holds": {k: prim[k]["spearman_rho"] > 0 for k in ("J", "R")},
        "break_triggered": {k: prim[k]["median_damage_rare_half"] < 0 for k in ("J", "R")},
    }

    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)

    # figure: zipf vs working-depth damage, headline subset
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    inter = out["intermediates"]
    xs = [v["zipf"] for v in inter.values()]
    for ki, kind in enumerate(KINDS):
        ax = axes[ki]
        ys = [v[kind] for v in inter.values()]
        ax.axhline(0, color="grey", lw=0.8)
        ax.scatter(xs, ys, s=18)
        for name, v in inter.items():
            ax.annotate(name.strip(), (v["zipf"], v[kind]), fontsize=6, alpha=0.7)
        ax.set_title(f"{kind}  rho={prim.get(kind, out['results']['headline_72'][PRIMARY_BAND][kind])['spearman_rho'] if kind in ('J','R') else out['results']['headline_72'][PRIMARY_BAND][kind]['spearman_rho']}")
        ax.set_xlabel("intermediate zipf frequency")
    axes[0].set_ylabel("damage log2(corr/raw), median L17-23")
    fig.suptitle("Step 8d: correction damage vs intermediate frequency (headline 72)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(json.dumps(out["results"]["headline_72"], indent=1))
    print("verdicts:", json.dumps(out["verdicts"]))
    print(f"wrote {OUT_JSON} + {OUT_PNG}")


if __name__ == "__main__":
    main()
