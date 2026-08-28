"""Step 3 asset: two-hop prompts with a known intermediate (feeds step 7).

Step 7's payoff figure asks: at each layer, how highly ranked is the
*intermediate* entity of a two-hop question, before and after subtracting the
offset `m_t`? That needs prompts where we know what the intermediate is.

Shape of a two-hop item -- the intermediate is never named in the prompt:

    prompt        "The city of Munich is located in the country whose
                   capital city is"
    intermediate  " Germany"      <- what the model must retrieve mid-stack
    answer        " Berlin"       <- what it must say

Built as a cross-product of a hand-checked entity table with four templates,
then **filtered by the model itself**: an item is kept only if Qwen3.5-4B's
true final-layer top-1 next token is the answer. That filter also quietly
removes any row where our label is wrong, since a bad label shows up as a
model disagreement -- with the caveat that a label wrong in the same way the
model is wrong would survive. Facts were chosen to be unambiguous for that
reason: no multi-capital countries (South Africa), no contested "country"
(Scotland vs UK), no multilingual states in the language template
(Belgium, Switzerland, Ireland, Canada).

Both the intermediate and the answer must be a single Qwen token, so step 7
can track one rank rather than a sequence. Items failing that are dropped
before the model ever runs, and the count is recorded.

**Shortcut control.** A two-hop item is only two-hop if the answer actually
depends on the entity. Some templates leak: 70% of the `currency` items
answer " euro", so a model that ignores the city entirely still scores well.
Each template is therefore also run *blinded* -- the entity removed, nothing
else changed -- and every item carries two flags at different strictness:
  `shortcut_solvable`        blinded top-1 is already this item's answer
  `shortcut_in_blinded_top5` the answer merely appears in the blinded top-5
The lenient flag catches almost nothing for `currency`, because the blinded
top-1 there is a markdown artefact (" **") -- but " euro" sits at rank 2, so
the strict flag catches all 19 euro items. Both are recorded rather than one
being chosen here. Step 7 reports shortcut items separately or drops them;
they are kept in the file so the choice stays visible and reversible.

Run: .venv/bin/python -m src.twohop   -> results/step3/step3_twohop.json
"""

from __future__ import annotations

import json

import torch

from src.lens import MODEL_ID, Instrument

OUT_PATH = "results/step3/step3_twohop.json"

# (entity, kind, country, capital, currency, language)
# language=None where the country is genuinely multilingual.
ENTITIES = [
    ("Munich",           "city",     "Germany",        "Berlin",     "euro",   "German"),
    ("Hamburg",          "city",     "Germany",        "Berlin",     "euro",   "German"),
    ("Frankfurt",        "city",     "Germany",        "Berlin",     "euro",   "German"),
    ("Barcelona",        "city",     "Spain",          "Madrid",     "euro",   "Spanish"),
    ("Valencia",         "city",     "Spain",          "Madrid",     "euro",   "Spanish"),
    ("Milan",            "city",     "Italy",          "Rome",       "euro",   "Italian"),
    ("Venice",           "city",     "Italy",          "Rome",       "euro",   "Italian"),
    ("Naples",           "city",     "Italy",          "Rome",       "euro",   "Italian"),
    ("Lyon",             "city",     "France",         "Paris",      "euro",   "French"),
    ("Marseille",        "city",     "France",         "Paris",      "euro",   "French"),
    ("Osaka",            "city",     "Japan",          "Tokyo",      "yen",    "Japanese"),
    ("Kyoto",            "city",     "Japan",          "Tokyo",      "yen",    "Japanese"),
    ("Shanghai",         "city",     "China",          "Beijing",    "yuan",   "Chinese"),
    ("Toronto",          "city",     "Canada",         "Ottawa",     "dollar", None),
    ("Vancouver",        "city",     "Canada",         "Ottawa",     "dollar", None),
    ("Sydney",           "city",     "Australia",      "Canberra",   "dollar", "English"),
    ("Melbourne",        "city",     "Australia",      "Canberra",   "dollar", "English"),
    ("Mumbai",           "city",     "India",          "Delhi",      "rupee",  "Hindi"),
    ("Istanbul",         "city",     "Turkey",         "Ankara",     "lira",   "Turkish"),
    ("Porto",            "city",     "Portugal",       "Lisbon",     "euro",   "Portuguese"),
    ("Rotterdam",        "city",     "Netherlands",    "Amsterdam",  "euro",   "Dutch"),
    ("Krakow",           "city",     "Poland",         "Warsaw",     "zloty",  "Polish"),
    ("Gothenburg",       "city",     "Sweden",         "Stockholm",  "krona",  "Swedish"),
    ("Bergen",           "city",     "Norway",         "Oslo",       "krone",  "Norwegian"),
    ("Busan",            "city",     "Korea",          "Seoul",      "won",    "Korean"),
    ("Auckland",         "city",     "Zealand",        "Wellington", "dollar", "English"),
    ("Alexandria",       "city",     "Egypt",          "Cairo",      "pound",  "Arabic"),
    ("Casablanca",       "city",     "Morocco",        "Rabat",      "dirham", "Arabic"),
    ("Salzburg",         "city",     "Austria",        "Vienna",     "euro",   "German"),
    ("Chicago",          "city",     "America",        "Washington", "dollar", "English"),
    ("Boston",           "city",     "America",        "Washington", "dollar", "English"),
    ("Guadalajara",      "city",     "Mexico",         "Mexico",     "peso",   "Spanish"),
    ("Eiffel Tower",     "landmark", "France",         "Paris",      "euro",   "French"),
    ("Colosseum",        "landmark", "Italy",          "Rome",       "euro",   "Italian"),
    ("Taj Mahal",        "landmark", "India",          "Delhi",      "rupee",  "Hindi"),
    ("Great Wall",       "landmark", "China",          "Beijing",    "yuan",   "Chinese"),
    ("Statue of Liberty","landmark", "America",        "Washington", "dollar", "English"),
    ("Sagrada Familia",  "landmark", "Spain",          "Madrid",     "euro",   "Spanish"),
    ("Brandenburg Gate", "landmark", "Germany",        "Berlin",     "euro",   "German"),
    ("Acropolis",        "landmark", "Greece",         "Athens",     "euro",   "Greek"),
    ("Machu Picchu",     "landmark", "Peru",           "Lima",       "sol",    "Spanish"),
    ("Mount Fuji",       "landmark", "Japan",          "Tokyo",      "yen",    "Japanese"),
    ("Christ the Redeemer","landmark","Brazil",        "Brasilia",   "real",   "Portuguese"),
    ("Angkor Wat",       "landmark", "Cambodia",       "Phnom",      "riel",   "Khmer"),
    ("Neuschwanstein",   "landmark", "Germany",        "Berlin",     "euro",   "German"),
]

SUBJECT = {
    "city": "The city of {e}",
    "landmark": "The {e}",
}
# entity stripped out, everything else identical -- the shortcut control
BLIND_SUBJECT = {"city": "The city", "landmark": "The landmark"}
# template key -> (tail of the prompt, which column supplies the answer)
TEMPLATES = {
    "capital":  ("is located in the country whose capital city is", "capital"),
    "currency": ("is located in the country whose currency is the", "currency"),
    "language": ("is located in the country whose main language is", "language"),
    "capital_of": ("is located in a country, and the capital of that country is", "capital"),
}
COLUMNS = ("country", "capital", "currency", "language")


def build_candidates(tokenizer) -> tuple[list[dict], dict]:
    """Cross-product table x templates, keeping only items whose intermediate
    and answer are both a single Qwen token."""
    def single_token(word: str) -> int | None:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        return ids[0] if len(ids) == 1 else None

    items, dropped = [], {"multi_token_intermediate": 0, "multi_token_answer": 0,
                          "no_label": 0}
    for entity, kind, *cols in ENTITIES:
        row = dict(zip(COLUMNS, cols))
        inter_id = single_token(row["country"])
        if inter_id is None:
            dropped["multi_token_intermediate"] += len(TEMPLATES)
            continue
        for tname, (tail, answer_col) in TEMPLATES.items():
            answer = row[answer_col]
            if answer is None:
                dropped["no_label"] += 1
                continue
            answer_id = single_token(answer)
            if answer_id is None:
                dropped["multi_token_answer"] += 1
                continue
            items.append({
                "id": f"{entity}|{tname}".replace(" ", "_"),
                "entity": entity,
                "entity_kind": kind,
                "template": tname,
                "prompt": f"{SUBJECT[kind].format(e=entity)} {tail}",
                "intermediate": " " + row["country"],
                "intermediate_id": inter_id,
                "answer": " " + answer,
                "answer_id": answer_id,
            })
    return items, dropped


def main() -> None:
    inst = Instrument()
    items, dropped = build_candidates(inst.tokenizer)
    print(f"{len(items)} candidates after single-token filter; dropped {dropped}")

    def top5_of(prompt: str) -> list[int]:
        input_ids = inst.lm.encode(prompt)
        with torch.no_grad():
            return inst.model(input_ids=input_ids).logits[0, -1].float().topk(5).indices.tolist()

    # --- shortcut control: what does each template answer with no entity? ---
    blinded: dict[tuple[str, str], list[int]] = {}
    for tname, (tail, _) in TEMPLATES.items():
        for kind, subject in BLIND_SUBJECT.items():
            prompt = f"{subject} {tail}"
            blinded[(tname, kind)] = top5_of(prompt)
            print(f"   blinded [{tname}/{kind}]: {prompt!r} -> "
                  f"{[inst.tokenizer.decode([i]) for i in blinded[(tname, kind)]]}")

    kept, rejected = [], []
    for item in items:
        input_ids = inst.lm.encode(item["prompt"])
        with torch.no_grad():
            logits = inst.model(input_ids=input_ids).logits[0, -1].float()
        top5 = logits.topk(5).indices.tolist()
        blind = blinded[(item["template"], item["entity_kind"])]
        item = {
            **item,
            "model_top1": inst.tokenizer.decode([top5[0]]),
            "model_top5": [inst.tokenizer.decode([i]) for i in top5],
            "answer_rank_in_top5": (
                top5.index(item["answer_id"]) if item["answer_id"] in top5 else None
            ),
            "shortcut_solvable": blind[0] == item["answer_id"],
            "shortcut_in_blinded_top5": item["answer_id"] in blind,
            "blinded_top1": inst.tokenizer.decode([blind[0]]),
            "blinded_top5": [inst.tokenizer.decode([i]) for i in blind],
        }
        (kept if top5[0] == item["answer_id"] else rejected).append(item)

    by_template: dict[str, dict[str, int]] = {}
    for t in TEMPLATES:
        n_k = sum(i["template"] == t for i in kept)
        n_r = sum(i["template"] == t for i in rejected)
        n_short = sum(i["template"] == t and i["shortcut_solvable"] for i in kept)
        n_top5 = sum(i["template"] == t and i["shortcut_in_blinded_top5"] for i in kept)
        by_template[t] = {
            "kept": n_k, "rejected": n_r,
            "keep_rate": round(n_k / max(n_k + n_r, 1), 3),
            "shortcut_solvable_top1": n_short,
            "shortcut_in_blinded_top5": n_top5,
            "two_hop_strict": n_k - n_top5,
        }

    n_short = sum(i["shortcut_solvable"] for i in kept)
    n_top5 = sum(i["shortcut_in_blinded_top5"] for i in kept)
    record = {
        "model_id": MODEL_ID,
        "n_shortcut_solvable_top1": n_short,
        "n_shortcut_in_blinded_top5": n_top5,
        "n_two_hop_lenient": len(kept) - n_short,
        "n_two_hop_strict": len(kept) - n_top5,
        "filter": "kept iff the model's true final-layer top-1 next token == answer",
        "n_entities": len(ENTITIES),
        "n_candidates": len(items),
        "n_kept": len(kept),
        "n_rejected": len(rejected),
        "dropped_before_model": dropped,
        "by_template": by_template,
        "n_distinct_intermediates": len({i["intermediate"] for i in kept}),
        "items": kept,
        "rejected_items": rejected,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    print(f"kept {len(kept)}/{len(items)}: shortcut by blinded-top1 {n_short}, "
          f"by blinded-top5 {n_top5} -> two-hop {len(kept) - n_short} lenient / "
          f"{len(kept) - n_top5} strict")
    for t, v in by_template.items():
        print(f"   {t:>11}: kept {v['kept']:>3}  shortcut(top1) "
              f"{v['shortcut_solvable_top1']:>3}  shortcut(top5) "
              f"{v['shortcut_in_blinded_top5']:>3}  strict two-hop {v['two_hop_strict']:>3}")
    print(f"distinct intermediates among kept: {record['n_distinct_intermediates']}")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
