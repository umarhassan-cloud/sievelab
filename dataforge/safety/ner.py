"""Person-name detection (NER stub).

A trigger-word detector: names are only flagged when introduced by an
honorific/title (``Dr.``, ``Mr.``, ``جناب``, ``محترمہ`` ...), which keeps
precision high on encyclopedic text full of public figures. It is a stub for
a real NER model; ``SpacyNameDetector`` shows the drop-in interface and is used
when spaCy and a model are installed.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_EN_TRIGGER = re.compile(
    r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam|Engr)\.?\s+((?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+){0,2})"
)
_UR_TRIGGER = re.compile(
    r"(?:جناب|محترمہ|محترم|ڈاکٹر|پروفیسر|بیگم|انجینئر)\s+([\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+)?)"
)
_UR_STOP = {"کے", "کی", "کا", "نے", "سے", "کو", "ہے", "ہیں", "اور", "میں", "پر"}


class NameDetector:
    def detect(self, text: str) -> List[Tuple[int, int, str]]:
        spans: List[Tuple[int, int, str]] = []
        for m in _EN_TRIGGER.finditer(text):
            spans.append((m.start(1), m.end(1), m.group(1)))
        for m in _UR_TRIGGER.finditer(text):
            name_tokens = m.group(1).split()
            while name_tokens and name_tokens[-1] in _UR_STOP:
                name_tokens.pop()
            if not name_tokens:
                continue
            name = " ".join(name_tokens)
            spans.append((m.start(1), m.start(1) + len(name), name))
        return spans


class SpacyNameDetector:
    """Optional backend: ``SpacyNameDetector("en_core_web_sm")``."""

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy  # type: ignore

        self.nlp = spacy.load(model)

    def detect(self, text: str) -> List[Tuple[int, int, str]]:
        return [(e.start_char, e.end_char, e.text) for e in self.nlp(text).ents if e.label_ == "PERSON"]
