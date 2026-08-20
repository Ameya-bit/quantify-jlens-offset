"""Token-string flags for the junk taxonomy (used by steps 2 and 3).

A token counts as *junk* for the step-2 setup figure when it is
punctuation/symbol/whitespace-only, contains undecodable bytes, or contains
non-Latin script. This is a proxy: it cannot catch obscene-but-well-formed
English words (that axis arrives with the step-5 assets); the qualitative
grid panel covers those by eye. The proxy is applied identically to all
three instruments and all layers, so depth trends and lens-vs-lens
comparisons remain meaningful even where the proxy is imperfect.

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
    byte_fragment, non_latin, is_junk."""
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
    flags["is_junk"] = (
        flags["punctuation"] or flags["byte_fragment"] or flags["non_latin"]
    )
    return flags


def junk_fraction(tokens: list[str]) -> float:
    """Fraction of tokens flagged as junk (for a top-k readout list)."""
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
        "____": True,
        "!!!": True,
        " ((": True,
        "\n\n": True,
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
    assert junk_fraction(["____", " Paris"]) == 0.5
    print(f"flags self-check OK ({len(cases)} cases)")


if __name__ == "__main__":
    _self_check()
