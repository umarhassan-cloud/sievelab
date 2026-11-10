"""Heuristic "teacher" that scores educational value on FineWeb-Edu's additive 0-5 rubric.

FineWeb-Edu prompts Llama-3-70B with an additive rubric (one point each for:
basic relevant information; addresses elements pertinent to education;
coherent and appropriate for educational use; highly relevant and beneficial
for learning at school level; outstanding educational value), then distils
those ~450k annotations into a small classifier that is cheap enough to run on
billions of documents.

Here an explicit, bilingual (English / Urdu / Roman Urdu) rule set plays the
role of the LLM annotator so the whole loop runs offline and is fully
reproducible. The student classifier (``classifier.py``) is trained on these
labels — exactly the teacher → student distillation FineWeb-Edu uses. Swapping
in an LLM teacher means replacing ``score()``; nothing downstream changes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sievelab.text_utils import sentences, words

EDU_TERMS = {
    # English
    "learn", "learning", "lesson", "study", "studies", "students", "student", "teacher", "teachers",
    "theory", "process", "concept", "definition", "defined", "explain", "explains", "experiment",
    "scientists", "science", "history", "historians", "mathematics", "number", "element", "atomic",
    "energy", "cells", "organism", "organisms", "system", "function", "chapter", "research",
    "evidence", "principle", "law", "laws", "equation", "physics", "chemistry", "biology",
    "geography", "economics", "government", "civilization", "century", "empire", "body", "blood",
    "molecules", "planet", "earth", "atmosphere", "temperature", "measure", "important", "known",
    "called", "means", "result", "causes", "structure", "properties", "knowledge", "understand",
    "sum", "product", "multiplied", "equals", "equal", "area", "perimeter", "percent", "metres",
    "kilometres", "prime", "divisors", "divided", "capital", "planet", "solar", "classify",
    "biologists", "invented", "developed", "degrees", "celsius", "mammal", "reptile", "species",
    # Urdu
    "سبق", "سیکھیں", "سیکھنے", "مطالعہ", "طلبہ", "اساتذہ", "نظریہ", "عمل", "تصور", "تعریف",
    "وضاحت", "تجربہ", "تجربے", "تجربات", "سائنس", "تاریخ", "ریاضی", "عنصر", "جوہری", "توانائی",
    "خلیہ", "خلیوں", "نظام", "باب", "تحقیق", "اصول", "قانون", "قوانین", "مساوات", "طبیعیات", "کیمیا",
    "حیاتیات", "جغرافیہ", "معیشت", "حکومت", "تہذیب", "صدی", "سلطنت", "جسم", "خون", "مالیکیولز",
    "زمین", "درجہ", "حرارت", "اہم", "مورخین", "نتیجہ", "ساخت", "علم", "سمجھنا", "عدد", "قوت", "مشاہدے",
    "مجموعہ", "ضرب", "حاصل", "رقبہ", "احاطہ", "فیصد", "میٹر", "کلومیٹر", "مفرد", "تقسیم", "دارالحکومت",
    "سیارہ", "شمسی", "ماہرین", "ایجاد", "ڈگری", "برابر", "ممالیہ", "جمع",
    # Roman Urdu
    "sabaq", "parhai", "ilm", "tajurba", "science", "tareekh",
}
EXPLANATION_MARKERS = (
    "for example", "because", "therefore", "this means", "as a result", "in other words",
    "this is why", "such as", "in simple terms", "by definition",
    "مثال کے طور پر", "کیونکہ", "اس لیے", "لہذا", "اس کا مطلب", "نتیجہ", "آسان الفاظ میں",
    "تعریف کے مطابق", "اسی طرح", "مثلاً",
)
SPAM_TERMS = {
    "buy", "cheap", "discount", "offer", "click", "free", "sale", "deal", "deals", "price",
    "casino", "bonus", "earn", "order", "shipping", "limited", "guaranteed", "subscribe", "win",
    "خریدیں", "سستے", "رعایت", "کلک", "مفت", "سیل", "پیشکش", "کمائیں", "چھوٹ", "آرڈر", "ڈیلیوری",
}
CASUAL_TERMS = {
    "lol", "omg", "haha", "hahaha", "lmao", "brb", "btw", "idk", "u", "ur", "gonna", "wanna",
    "yaar", "bro", "lol", "ہاہاہا", "یار", "لول", "واہ", "مزہ",
}
_URL_RE = re.compile(r"https?://|www\.")
_EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")


@dataclass
class AnnotatorFeatures:
    n_words: int
    n_sentences: int
    avg_sentence_words: float
    edu_density: float
    explanation_hits: int
    spam_hits: int
    casual_hits: int
    unique_ratio: float
    exclamations: int
    urls: int
    emojis: int


def extract_features(text: str) -> AnnotatorFeatures:
    toks = words(text)
    n = len(toks)
    sents = sentences(text)
    lower = text.lower()
    return AnnotatorFeatures(
        n_words=n,
        n_sentences=len(sents),
        avg_sentence_words=n / max(len(sents), 1),
        edu_density=sum(t in EDU_TERMS for t in toks) / max(n, 1),
        explanation_hits=sum(lower.count(m) for m in EXPLANATION_MARKERS),
        spam_hits=sum(t in SPAM_TERMS for t in toks),
        casual_hits=sum(t in CASUAL_TERMS for t in toks),
        unique_ratio=len(set(toks)) / max(n, 1),
        exclamations=text.count("!"),
        urls=len(_URL_RE.findall(lower)),
        emojis=len(_EMOJI_RE.findall(text)),
    )


class EduHeuristicAnnotator:
    """Additive 0-5 educational-value rubric."""

    def score_with_reasons(self, text: str) -> Tuple[int, List[str], Dict[str, float]]:
        f = extract_features(text)
        reasons: List[str] = []
        spammy = f.spam_hits >= 3 or (f.urls >= 1 and f.spam_hits >= 1) or f.exclamations >= 4
        if spammy:
            return 0, ["spam"], f.__dict__
        points = 0
        if f.n_words >= 30 and f.edu_density > 0:
            points += 1
            reasons.append("basic_information")
        if f.edu_density >= 0.03 and 6 <= f.avg_sentence_words <= 45:
            points += 1
            reasons.append("educational_elements")
        if f.unique_ratio >= 0.45 and f.casual_hits == 0 and f.emojis == 0 and f.exclamations <= 1:
            points += 1
            reasons.append("coherent")
        if f.edu_density >= 0.06 and f.n_words >= 80 and points == 3:
            points += 1
            reasons.append("highly_relevant")
        if f.explanation_hits >= 2 and f.n_words >= 140 and points == 4:
            points += 1
            reasons.append("outstanding")
        return points, reasons, f.__dict__

    def score(self, text: str) -> int:
        return self.score_with_reasons(text)[0]
