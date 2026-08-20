"""Token-string flags for the junk taxonomy (used by steps 2 and 3).

A token counts as *junk* when it could not be a correct continuation of the
surveyed English text at all: it contains undecodable bytes, or non-Latin
script. Base rate of those two in the surveyed pile-10k texts is exactly
0.000 (3125 tokens), so any such readout is unambiguously wrong and the
floor for a perfect instrument is zero.

`punctuation` is recorded but is NOT junk (changed 20 Aug, see devlog 0.0.2
addendum 4). Real pile text is 15.0% punctuation tokens and four of the
corpus's ten most frequent tokens are punctuation, so a perfect unigram
predictor would score 0.4 under the old rule -- the flag punished
correctness. Worse, punctuation *rises* with depth exactly where the
anomaly rate falls (J-lens L10-13 -> L16-19: punct 0.033 -> 0.100 while
anomaly 0.112 -> 0.075, both p<0.01 paired by text), so summing the two
manufactured a spurious mid-depth plateau. It is now reported as its own
series against the 0.150 text base rate.

Scoping to "concept slots" (dividing by 10 - n_punctuation) was rejected:
the denominator would shrink precisely where punctuation peaks, re-inflating
the same layers by ~12%. A fixed denominator of 10 keeps layers, instruments
and baselines comparable, and counting punctuation as clean makes the
reported junk rate a conservative lower bound.

Remaining known limit (unchanged): obscene-but-well-formed English words and
Latin word-fragments ('oooo', 'fictiona') are not flagged, so the rate
undercounts the J-lens's characteristic junk style. Existence claims survive
an undercounting detector; instrument rankings do not rest on this proxy.

`leading_space` is recorded but is NOT junk evidence -- normal English word
tokens start with a space; step 3's frequency work wants the flag.

Self-check battery: `.venv/bin/python -m src.flags` (asserts on examples).
"""

from __future__ import annotations

# Latin script (incl. Latin-1 and Latin Extended-A/B) ends at U+024F; any
# letter beyond that is counted as non-Latin (CJK, Cyrillic, Greek, Arabic...).
LAST_LATIN_CODEPOINT = 0x024F
REPLACEMENT_CHAR = "�"  # what decode() emits for undecodable byte pieces


def token_flags(token: str) -> dict[str, bool]:
    """Flags for one decoded token string. Keys: leading_space, punctuation,
    byte_fragment, non_latin, is_junk. `is_junk` = byte_fragment or
    non_latin; punctuation is recorded separately (see module docstring)."""
    stripped = token.strip()
    has_alnum = any(c.isalnum() for c in token)
    flags = {
        "leading_space": token[:1].isspace(),
        # symbol/punctuation/whitespace run: no letters or digits at all
        # (catches "____", "!!!", "((", pure-whitespace tokens)
        "punctuation": not has_alnum,
        "byte_fragment": REPLACEMENT_CHAR in token,
        "non_latin": any(
            c.isalpha() and ord(c) > LAST_LATIN_CODEPOINT for c in stripped
        ),
    }
    flags["is_junk"] = flags["byte_fragment"] or flags["non_latin"]
    return flags


def junk_fraction(tokens: list[str]) -> float:
    """Fraction of tokens flagged as junk (for a top-k readout list).
    Denominator is always len(tokens) -- punctuation slots count as clean,
    not as excluded (see module docstring)."""
    if not tokens:
        raise ValueError("empty token list")
    return sum(token_flags(t)["is_junk"] for t in tokens) / len(tokens)


def _self_check() -> None:
    cases = {
        # token -> expected is_junk
        " Paris": False,
        "Paris": False,
        "zinho": False,  # subword fragment, but well-formed Latin: not junk
        " 1998": False,
        # punctuation is a legitimate prediction (15.0% of real pile tokens):
        "____": False,
        "!!!": False,
        " ((": False,
        "\n\n": False,
        "��": True,
        "你好": True,  # non-Latin script (junk *as a proxy*, see docstring)
        " привет": True,
        "café": False,  # Latin-1 accents stay Latin
    }
    for token, expected in cases.items():
        got = token_flags(token)["is_junk"]
        assert got == expected, f"{token!r}: is_junk={got}, expected {expected}"
    assert token_flags(" Paris")["leading_space"]
    assert not token_flags("Paris")["leading_space"]
    # punctuation still *recorded*, just not junk
    for t in ("____", "!!!", " ((", "\n\n"):
        assert token_flags(t)["punctuation"], t
    assert not token_flags(" Paris")["punctuation"]
    assert junk_fraction(["\u4f60\u597d", " Paris"]) == 0.5
    assert junk_fraction(["____", " Paris"]) == 0.0
    print(f"flags self-check OK ({len(cases)} cases)")


if __name__ == "__main__":
    _self_check()
