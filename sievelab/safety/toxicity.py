"""Toxicity / hate-speech filtering.

Two scorers, combined with ``max``:

* ``LexiconToxicityScorer`` — weighted multilingual lexicon (English, Urdu,
  Roman Urdu) with severity tiers. Scoring is *sentence-level*: a document's
  score is the maximum over its sentences, so a single abusive comment under
  an otherwise clean article is caught without letting long documents dilute
  it. Supports per-sentence redaction as a retention policy.
* ``ToxicityClassifier`` — optional TF-IDF + logistic regression model that
  can be trained on labelled data to catch paraphrases the lexicon misses.

The lexicon intentionally contains insults, threats and dehumanising phrases
but no slurs; production deployments should load a vetted lexicon via
``LexiconToxicityScorer.from_json``.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sievelab.text_utils import normalize_text, sentences, strip_diacritics

# severity: 1 = insult/profanity, 2 = harassment, 3 = threat / dehumanisation
DEFAULT_LEXICON: Dict[str, Dict[str, int]] = {
    "en": {
        "idiot": 1, "stupid": 1, "moron": 1, "loser": 1, "dumb": 1, "pathetic": 1, "worthless": 2,
        "trash": 1, "shut up": 1, "nobody wants you": 2, "get lost": 1, "disgusting": 1,
        "i will kill you": 3, "kill you": 3, "i will hurt you": 3, "you deserve to die": 3,
        "are vermin": 3, "subhuman": 3, "should be driven out": 3, "go back to where": 2,
    },
    "ur": {
        "بیوقوف": 1, "بے وقوف": 1, "گدھا": 1, "گدھے": 1, "جاہل": 1, "کمینہ": 2, "کمینے": 2,
        "بے غیرت": 2, "الو کا پٹھا": 2, "چپ کرو": 1, "نکمے": 1, "جان سے مار": 3, "مار دوں گا": 3,
        "کیڑے مکوڑے": 3, "ملک سے نکال": 3,
    },
    "ur_Latn": {
        "bewakoof": 1, "gadha": 1, "gadhay": 1, "jahil": 1, "kameena": 2, "kameeney": 2,
        "beghairat": 2, "ullu ka pattha": 2, "chup kar": 1, "maar dunga": 3, "jaan se maar": 3,
    },
}

_LEET = str.maketrans({"@": "a", "0": "o", "1": "i", "3": "e", "$": "s", "4": "a"})


def _norm(text: str) -> str:
    text = strip_diacritics(text).lower().translate(_LEET)
    return " " + normalize_text(text, lowercase=False) + " "


@dataclass
class ToxicityResult:
    score: float
    hits: List[Tuple[str, int]]
    toxic_sentences: List[int]


class LexiconToxicityScorer:
    def __init__(self, lexicon: Optional[Dict[str, Dict[str, int]]] = None, saturation: float = 2.0):
        lexicon = lexicon or DEFAULT_LEXICON
        self.terms: Dict[str, int] = {}
        for entries in lexicon.values():
            for term, sev in entries.items():
                key = _norm(term).strip()
                self.terms[key] = max(sev, self.terms.get(key, 0))
        self.saturation = saturation
        ordered = sorted(self.terms, key=len, reverse=True)
        self._re = re.compile(r"(?<= )(" + "|".join(re.escape(t) for t in ordered) + r")(?= )")

    @classmethod
    def from_json(cls, path: str, **kw) -> "LexiconToxicityScorer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")), **kw)

    def sentence_score(self, sentence: str) -> Tuple[float, List[Tuple[str, int]]]:
        norm = _norm(sentence)
        hits = [(m, self.terms[m]) for m in self._re.findall(norm)]
        if not hits:
            return 0.0, []
        if max(s for _, s in hits) >= 3:
            return 1.0, hits
        return min(1.0, sum(s for _, s in hits) / self.saturation), hits

    def score(self, text: str, sentence_threshold: float = 0.5) -> ToxicityResult:
        best, all_hits, toxic_idx = 0.0, [], []
        for i, sent in enumerate(sentences(text)):
            s, hits = self.sentence_score(sent)
            all_hits.extend(hits)
            if s >= sentence_threshold:
                toxic_idx.append(i)
            best = max(best, s)
        return ToxicityResult(best, all_hits, toxic_idx)

    def redact_sentences(self, text: str, sentence_threshold: float = 0.5) -> str:
        """Drop toxic sentences, keep the rest of the document (paragraph breaks preserved)."""
        out_paras = []
        for para in text.split("\n"):
            kept = [s for s in sentences(para) if self.sentence_score(s)[0] < sentence_threshold]
            if kept:
                out_paras.append(" ".join(kept))
        return "\n".join(out_paras)


class ToxicityClassifier:
    """Optional learned scorer (char n-gram TF-IDF + logistic regression)."""

    def __init__(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        self.model = Pipeline(
            [
                ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=1)),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> "ToxicityClassifier":
        self.model.fit(list(texts), list(labels))
        return self

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.predict_proba(list(texts))[:, 1]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str) -> "ToxicityClassifier":
        with open(path, "rb") as fh:
            return pickle.load(fh)
