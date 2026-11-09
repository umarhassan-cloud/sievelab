"""Language identification with first-class support for Arabic-script languages.

Off-the-shelf identifiers routinely confuse Urdu with Persian/Arabic and have
no notion of Roman Urdu (Urdu written in Latin script, extremely common on the
web and social media). The default ``heuristic`` backend is a transparent
two-step classifier:

1. **Script detection** over letter code points (Arabic, Latin, Devanagari ...).
2. **Within-script disambiguation** using language-specific letters
   (Urdu ٹ ڈ ڑ ں ے, Pashto ټ ډ ښ ګ, Sindhi ڄ ڃ ڪ ۾, Arabic ة ي ك) plus
   high-frequency function words.

Optional backends: ``langdetect`` (Naive Bayes over char n-grams) and
``fasttext`` (lid.176 / GlotLID ``.bin`` models).
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from dataforge.text_utils import strip_diacritics, words

SCRIPT_RANGES: List[Tuple[str, List[Tuple[int, int]]]] = [
    ("Latin", [(0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x1E00, 0x1EFF)]),
    ("Arabic", [(0x600, 0x6FF), (0x750, 0x77F), (0x8A0, 0x8FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]),
    ("Devanagari", [(0x900, 0x97F)]),
    ("Bengali", [(0x980, 0x9FF)]),
    ("Gurmukhi", [(0xA00, 0xA7F)]),
    ("Gujarati", [(0xA80, 0xAFF)]),
    ("Tamil", [(0xB80, 0xBFF)]),
    ("Telugu", [(0xC00, 0xC7F)]),
    ("Sinhala", [(0xD80, 0xDFF)]),
    ("Thai", [(0xE00, 0xE7F)]),
    ("Ethiopic", [(0x1200, 0x137F)]),
    ("Cyrillic", [(0x400, 0x4FF)]),
    ("Greek", [(0x370, 0x3FF)]),
    ("Hebrew", [(0x590, 0x5FF)]),
    ("Hangul", [(0xAC00, 0xD7AF), (0x1100, 0x11FF)]),
    ("Kana", [(0x3040, 0x30FF)]),
    ("Han", [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)]),
]

SCRIPT_DEFAULT_LANG = {
    "Devanagari": "hi", "Bengali": "bn", "Gurmukhi": "pa", "Gujarati": "gu", "Tamil": "ta",
    "Telugu": "te", "Sinhala": "si", "Thai": "th", "Ethiopic": "am", "Cyrillic": "ru",
    "Greek": "el", "Hebrew": "he", "Hangul": "ko", "Kana": "ja", "Han": "zh",
}

ARABIC_SCRIPT: Dict[str, Dict[str, object]] = {
    "ur": {
        "chars": "\u0679\u0688\u0691\u06ba\u06d2\u06d3\u06c1\u06be",  # ٹ ڈ ڑ ں ے ۓ ہ ھ
        "stop": {"کے", "کی", "کا", "ہے", "ہیں", "میں", "اور", "سے", "کو", "نے", "پر", "یہ",
                 "ایک", "بھی", "تھا", "تھی", "گیا", "جو", "کہ", "لیے", "ہو", "وہ", "کر"},
    },
    "fa": {
        "chars": "\u0647\u0698",  # ه ژ (Urdu uses ہ instead of ه)
        "stop": {"است", "این", "که", "را", "در", "با", "از", "برای", "می", "شود", "های", "یک",
                 "آن", "به", "هر", "شده", "نیز", "او", "ما"},
    },
    "ar": {
        "chars": "\u0629\u064a\u0643\u0649\u0623\u0625",  # ة ي ك ى أ إ
        "stop": {"في", "من", "على", "إلى", "عن", "هذا", "هذه", "التي", "الذي", "أن", "كان",
                 "مع", "هو", "هي", "ما", "لا", "كل", "بين"},
    },
    "ps": {
        "chars": "\u067c\u0689\u0693\u0696\u069a\u06ab\u06bc\u06d0\u06cd\u0681\u0685",  # ټ ډ ړ ږ ښ ګ ڼ ې ۍ ځ څ
        "stop": {"د", "په", "او", "دی", "ده", "چې", "ته", "له", "هم", "یو", "دا", "څخه", "لپاره"},
    },
    "sd": {
        "chars": "\u0684\u0683\u0687\u068a\u068c\u068d\u068f\u0699\u06a6\u06bb\u067a\u067f\u0680\u06b3\u06b1\u06aa\u06fe",
        "stop": {"جي", "۾", "آهي", "۽", "کي", "تي", "جو", "جا", "هڪ", "ٿو", "ٿي", "ته", "سان"},
    },
}

LATIN_STOPWORDS: Dict[str, set] = {
    "en": {"the", "and", "of", "to", "in", "is", "that", "it", "for", "was", "on", "are", "with",
           "as", "by", "this", "be", "from", "at", "or", "which", "an", "have", "they", "you", "we"},
    "fr": {"le", "la", "les", "des", "et", "est", "une", "un", "du", "dans", "que", "qui", "pour",
           "pas", "sur", "au", "avec", "ce", "sont", "par", "il", "elle", "nous", "l", "d"},
    "es": {"el", "la", "los", "las", "de", "y", "que", "en", "es", "un", "una", "por", "con",
           "para", "del", "se", "al", "lo", "como", "más", "su", "son", "está"},
    "de": {"der", "die", "das", "und", "ist", "nicht", "ein", "eine", "zu", "den", "mit", "von",
           "sich", "auf", "für", "des", "dem", "im", "wir", "auch"},
    "pt": {"o", "os", "as", "de", "e", "que", "do", "da", "em", "um", "uma", "para", "com",
           "não", "por", "mais", "dos", "das", "foi"},
    "it": {"il", "lo", "gli", "di", "e", "che", "è", "un", "una", "per", "con", "non", "del",
           "della", "sono", "nel", "alla"},
    "sw": {"na", "ya", "wa", "ni", "kwa", "za", "katika", "kuwa", "hii", "cha", "vya", "kama",
           "sana", "zaidi", "hiyo", "yake", "ambayo", "vyote", "mara", "kila", "husaidia"},
    "id": {"yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "dengan", "tidak", "adalah", "dalam"},
    "tr": {"ve", "bir", "bu", "için", "ile", "çok", "olarak", "daha", "gibi", "ama"},
    # Roman Urdu: Urdu written in Latin script (ISO 639-3 urd, script Latn).
    "ur_Latn": {"hai", "hain", "ka", "ki", "ke", "aur", "mein", "se", "ko", "nahi", "kya", "yeh",
                "ye", "woh", "bhi", "tha", "thi", "ho", "raha", "rahi", "kar", "yaar", "bohat",
                "kahan", "gaya", "gayi", "acha", "hoon", "tum", "ap", "aap", "kal", "aaj", "phir"},
}

STOPWORDS: Dict[str, set] = {**{k: v["stop"] for k, v in ARABIC_SCRIPT.items()}, **LATIN_STOPWORDS}  # type: ignore[misc]


@lru_cache(maxsize=65536)
def script_of(ch: str) -> Optional[str]:
    cp = ord(ch)
    for name, ranges in SCRIPT_RANGES:
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return name
    return None


def script_histogram(text: str) -> Counter:
    hist: Counter = Counter()
    for ch in text:
        if ch.isalpha():
            s = script_of(ch)
            if s:
                hist[s] += 1
    return hist


def _best(scores: Dict[str, float]) -> Tuple[Optional[str], float]:
    total = sum(scores.values())
    if total <= 0:
        return None, 0.0
    lang = max(scores, key=scores.get)  # type: ignore[arg-type]
    return lang, scores[lang] / total


class LanguageIdentifier:
    """``predict(text) -> (lang, score)`` where score is in [0, 1]."""

    def __init__(self, backend: str = "heuristic", fasttext_model: Optional[str] = None, max_chars: int = 2000):
        self.backend = backend
        self.max_chars = max_chars
        self._ft = None
        if backend == "langdetect":
            from langdetect import DetectorFactory  # noqa: F401

            DetectorFactory.seed = 0
        elif backend == "fasttext":
            if not fasttext_model:
                raise ValueError("fasttext backend requires `fasttext_model` path (e.g. lid.176.bin)")
            import fasttext  # type: ignore

            self._ft = fasttext.load_model(fasttext_model)
        elif backend != "heuristic":
            raise ValueError(f"Unknown langid backend: {backend}")

    def predict(self, text: str) -> Tuple[str, float]:
        text = text[: self.max_chars]
        if self.backend == "langdetect":
            return self._predict_langdetect(text)
        if self.backend == "fasttext":
            return self._predict_fasttext(text)
        return self._predict_heuristic(text)

    def _predict_heuristic(self, text: str) -> Tuple[str, float]:
        hist = script_histogram(text)
        n_letters = sum(hist.values())
        if n_letters < 10:
            return "und", 0.0
        script, count = hist.most_common(1)[0]
        if "Kana" in hist and script == "Han":
            script, count = "Kana", hist["Han"] + hist["Kana"]
        share = count / n_letters
        length_factor = min(1.0, n_letters / 40)

        if script == "Arabic":
            clean = strip_diacritics(text)
            toks = words(clean)
            n_tok = max(len(toks), 1)
            scores: Dict[str, float] = {}
            for lang, spec in ARABIC_SCRIPT.items():
                stop_frac = sum(t in spec["stop"] for t in toks) / n_tok  # type: ignore[operator]
                char_frac = sum(clean.count(c) for c in spec["chars"]) / count  # type: ignore[union-attr]
                scores[lang] = 3.0 * stop_frac + 5.0 * char_frac
            lang, conf = _best(scores)
            if lang is None:
                return "ar", 0.3 * share
            return lang, round(share * conf * length_factor, 4)

        if script == "Latin":
            toks = words(text)
            n_tok = max(len(toks), 1)
            scores = {lang: sum(t in sw for t in toks) / n_tok for lang, sw in LATIN_STOPWORDS.items()}
            lang, conf = _best(scores)
            if lang is None:
                return "und", 0.0
            coverage = min(1.0, scores[lang] / 0.15)
            return lang, round(share * (0.5 * conf + 0.5 * coverage) * length_factor, 4)

        lang = SCRIPT_DEFAULT_LANG.get(script, "und")
        return lang, round(share * length_factor, 4)

    def _predict_langdetect(self, text: str) -> Tuple[str, float]:
        from langdetect import detect_langs
        from langdetect.lang_detect_exception import LangDetectException

        try:
            best = detect_langs(text)[0]
        except LangDetectException:
            return "und", 0.0
        return best.lang.split("-")[0], round(float(best.prob), 4)

    _ISO3 = {"urd": "ur", "eng": "en", "arb": "ar", "ara": "ar", "pes": "fa", "fas": "fa", "hin": "hi",
             "fra": "fr", "spa": "es", "swh": "sw", "swa": "sw", "pbt": "ps", "pus": "ps", "snd": "sd",
             "pan": "pa", "ben": "bn", "deu": "de", "por": "pt", "ita": "it"}

    def _predict_fasttext(self, text: str) -> Tuple[str, float]:
        labels, probs = self._ft.predict(text.replace("\n", " "), k=1)  # type: ignore[union-attr]
        label = labels[0].replace("__label__", "")
        if "_" in label:
            code, script = label.split("_", 1)
            code = self._ISO3.get(code, code)
            label = f"{code}_Latn" if (code == "ur" and script == "Latn") else code
        return label, round(float(probs[0]), 4)
