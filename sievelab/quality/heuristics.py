"""Rule-based document quality filters (Gopher / C4 / FineWeb style).

Rules are deliberately language-aware: stop-word checks only apply to
languages with a stop-word list, and the uppercase rule only to cased scripts.
Applying English-tuned thresholds blindly to Urdu over-filters it (Urdu words
are short, and Arabic script has no case), which is one of the failure modes
this project measures.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from sievelab.lang.langid import STOPWORDS
from sievelab.text_utils import lines, raw_tokens, words

SYMBOLS = ("#", "...", "\u2026")
BULLETS = ("-", "*", "\u2022", "\u25aa", "\u2023", "\u27a2")
CASED_LANGS = {"en", "fr", "es", "de", "pt", "it", "sw", "id", "tr", "ur_Latn"}


@dataclass
class HeuristicConfig:
    min_words: int = 25
    max_words: int = 100_000
    min_mean_word_len: float = 2.0
    max_mean_word_len: float = 12.0
    max_symbol_word_ratio: float = 0.1
    max_bullet_line_frac: float = 0.9
    max_ellipsis_line_frac: float = 0.3
    max_dup_line_frac: float = 0.3
    max_top_bigram_char_frac: float = 0.2
    min_alpha_word_frac: float = 0.8
    max_uppercase_frac: float = 0.3
    max_exclamation_per_100_words: float = 3.0
    min_stopwords: int = 2

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "HeuristicConfig":
        d = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        return cls(**d)


def compute_stats(text: str, lang: Optional[str] = None) -> Dict[str, float]:
    toks = words(text)
    raw = raw_tokens(text)
    lns = lines(text)
    n = max(len(toks), 1)
    line_counts = Counter(lns)
    dup_lines = sum(c for c in line_counts.values() if c > 1)
    bigrams = Counter(zip(toks, toks[1:]))
    top_bigram_chars = 0.0
    if bigrams:
        (w1, w2), c = bigrams.most_common(1)[0]
        top_bigram_chars = c * (len(w1) + len(w2)) / max(sum(len(t) for t in toks), 1)
    letters = [ch for ch in text if ch.isalpha()]
    upper = sum(ch.isupper() for ch in letters)
    stop = STOPWORDS.get(lang or "", set())
    return {
        "n_words": len(toks),
        "mean_word_len": sum(len(t) for t in toks) / n,
        "symbol_word_ratio": sum(text.count(s) for s in SYMBOLS) / n,
        "bullet_line_frac": sum(ln.startswith(BULLETS) for ln in lns) / max(len(lns), 1),
        "ellipsis_line_frac": sum(ln.endswith(("...", "\u2026")) for ln in lns) / max(len(lns), 1),
        "dup_line_frac": dup_lines / max(len(lns), 1),
        "top_bigram_char_frac": top_bigram_chars if len(toks) >= 10 else 0.0,
        "alpha_word_frac": sum(any(ch.isalpha() for ch in t) for t in raw) / max(len(raw), 1),
        "uppercase_frac": upper / max(len(letters), 1),
        "exclamation_per_100_words": 100 * text.count("!") / n,
        "stopword_count": sum(t in stop for t in toks) if stop else -1,
    }


def apply_rules(text: str, lang: Optional[str], cfg: HeuristicConfig) -> Tuple[bool, List[str], Dict[str, float]]:
    s = compute_stats(text, lang)
    reasons: List[str] = []
    if s["n_words"] < cfg.min_words:
        reasons.append("too_short")
    if s["n_words"] > cfg.max_words:
        reasons.append("too_long")
    if not (cfg.min_mean_word_len <= s["mean_word_len"] <= cfg.max_mean_word_len):
        reasons.append("mean_word_len")
    if s["symbol_word_ratio"] > cfg.max_symbol_word_ratio:
        reasons.append("symbol_ratio")
    if s["bullet_line_frac"] > cfg.max_bullet_line_frac:
        reasons.append("bullet_lines")
    if s["ellipsis_line_frac"] > cfg.max_ellipsis_line_frac:
        reasons.append("ellipsis_lines")
    if s["dup_line_frac"] > cfg.max_dup_line_frac:
        reasons.append("repeated_lines")
    if s["top_bigram_char_frac"] > cfg.max_top_bigram_char_frac:
        reasons.append("repeated_ngrams")
    if s["alpha_word_frac"] < cfg.min_alpha_word_frac:
        reasons.append("non_alpha_words")
    if lang in CASED_LANGS and s["uppercase_frac"] > cfg.max_uppercase_frac:
        reasons.append("uppercase")
    if s["exclamation_per_100_words"] > cfg.max_exclamation_per_100_words:
        reasons.append("exclamations")
    if s["stopword_count"] >= 0 and s["stopword_count"] < cfg.min_stopwords:
        reasons.append("few_stopwords")
    if "lorem ipsum" in text.lower():
        reasons.append("lorem_ipsum")
    return (not reasons), reasons, s
