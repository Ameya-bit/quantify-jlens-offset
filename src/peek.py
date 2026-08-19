"""Peek at what the model is 'leaning toward' at each layer, for any prompt.

Reviewer tool: lets a human fact-check lens readouts by eye. Prints the
top-k tokens at the last position of the prompt, layer by layer, through
the chosen lens (J, R, or the plain logit-lens baseline).

Examples:
  .venv/bin/python -m src.peek "The Eiffel Tower is in the city of"
  .venv/bin/python -m src.peek "Water is made of hydrogen and" --kind logit
  .venv/bin/python -m src.peek "some text" --layers 0 2 4 6 8 --k 15
"""

from __future__ import annotations

import argparse

from src.lens import KINDS, Instrument

DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 30]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--kind", choices=KINDS, default="J")
    parser.add_argument("--layers", type=int, nargs="+", default=DEFAULT_LAYERS)
    parser.add_argument("--k", type=int, default=10, help="tokens per layer")
    args = parser.parse_args()

    inst = Instrument()
    acts, input_ids = inst.residuals(args.prompt, args.layers)
    last_token = inst.tokenizer.decode([input_ids[0, -1].item()])
    print(f"\nprompt: {args.prompt!r}")
    print(f"reading out at the LAST token ({last_token!r}) through the {args.kind}-lens\n")
    for layer in args.layers:
        scores = inst.score(acts[layer][-1], layer, args.kind)
        tokens = [repr(t) for t, _ in inst.top_tokens(scores, args.k)]
        print(f"L{layer:>2}: {'  '.join(tokens)}")


if __name__ == "__main__":
    main()
