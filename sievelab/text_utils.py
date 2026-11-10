"""Unicode-aware text normalisation and tokenisation shared by all stages.

Design constraints:
* Must work for Arabic-script languages (Urdu, Persian, Pashto, Sindhi), Indic
  scripts and Latin. ``\\w``-based regexes split Devanagari words on vowel signs,
  so tokenisation is "whitespace split after mapping punctuation/symbols to
  spaces" instead.
* Urdu text in the wild mixes Arabic and Urdu code points for the same letter
  (Arabic yeh U+064A vs Farsi yeh U+06CC, Arabic kaf U+0643 vs keheh U+06A9)
  depending on the keyboard. ``fold_arabic=True`` maps them together, which
  matters for exact/near-duplicate detection.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List

_PUNCT_TABLE = {
    cp: " " for cp in range(0x10000) if unicodedata.category(chr(cp))[0] in ("P", "S")
}
_ZERO_WIDTH = {0x200B: None, 0x200C: None, 0x200D: None, 0xFEFF: None, 0x00AD: None}
_ARABIC_FOLD = {0x064A: 0x06CC, 0x0649: 0x06CC, 0x0643: 0x06A9}
_DIACRITICS_RE = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E8\u06EA-\u06ED]")
_WS_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?\u06D4\u061F\u0964])\s+|\n+")


def strip_diacritics(text: str) -> str:
    """Remove Arabic-script harakat (zer, zabar, pesh, shadda ...)."""
    return _DIACRITICS_RE.sub("", text)


def normalize_text(
    text: str,
    lowercase: bool = True,
    strip_punct: bool = True,
    fold_arabic: bool = False,
    normalize_digits: bool = False,
) -> str:
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    if fold_arabic:
        text = strip_diacritics(text).translate(_ARABIC_FOLD)
    if lowercase:
        text = text.lower()
    if strip_punct:
        text = text.translate(_PUNCT_TABLE)
    if normalize_digits:
        text = _DIGIT_RE.sub("0", text)
    return _WS_RE.sub(" ", text).strip()


def words(text: str, **kwargs) -> List[str]:
    norm = normalize_text(text, **kwargs)
    return norm.split() if norm else []


def raw_tokens(text: str) -> List[str]:
    """Whitespace tokens without any normalisation (for symbol/caps statistics)."""
    return text.split()


def sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s and s.strip()]


def lines(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]
