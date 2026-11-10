"""PII detection and redaction.

Regex detectors with validation (Luhn for cards, octet ranges for IPv4) and
Pakistan-specific identifiers (CNIC ``#####-#######-#``, ``+92 3xx`` mobiles,
PK IBANs). Python's ``\\d`` matches any Unicode decimal digit, so the same
patterns catch numbers written in Urdu/Extended Arabic-Indic digits
(``۰۳۰۰-۱۲۳۴۵۶۷``); ``int()`` handles them in the Luhn check too.

Overlapping matches are resolved by detector priority (most specific first).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sievelab.safety.ner import NameDetector


@dataclass
class PIIMatch:
    type: str
    start: int
    end: int
    value: str


def luhn_valid(number: str) -> bool:
    digits = [int(c) for c in number if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _ipv4_valid(s: str) -> bool:
    parts = s.split(".")
    return len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 and (p == "0" or not p.startswith("0")) for p in parts)


_D = r"\d"
DETECTORS: List[Tuple[str, "re.Pattern[str]", Optional[Callable[[str], bool]]]] = [
    ("EMAIL", re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"), None),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{4}){3,7}(?:[ ]?[A-Z0-9]{1,4})?\b"), None),
    ("CNIC", re.compile(rf"(?<!{_D}){_D}{{5}}-{_D}{{7}}-{_D}(?!{_D})"), None),
    ("CREDIT_CARD", re.compile(rf"(?<!{_D})(?:{_D}[ -]?){{12,18}}{_D}(?!{_D})"), luhn_valid),
    ("PHONE", re.compile(rf"(?<![\w+])(?:\+92|0092|[0\u06f0])[ -]?[3\u06f3]{_D}{{2}}[ -]?{_D}{{7}}(?!{_D})"), None),
    ("PHONE", re.compile(rf"(?<![\w+])\+{_D}{{1,3}}[ -]?\(?{_D}{{1,4}}\)?(?:[ -]?{_D}{{2,4}}){{2,4}}(?!{_D})"), None),
    ("PHONE", re.compile(rf"(?<!{_D})\(?{_D}{{3}}\)?[ .-]{_D}{{3}}[ .-]{_D}{{4}}(?!{_D})"), None),
    ("IP_ADDRESS", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\.?\d)"), _ipv4_valid),
]

PLACEHOLDER = "<{type}>"


class PIIDetector:
    def __init__(self, enable_names: bool = True, types: Optional[Sequence[str]] = None):
        self.types = set(types) if types else None
        self.names = NameDetector() if enable_names else None

    def detect(self, text: str) -> List[PIIMatch]:
        found: List[PIIMatch] = []
        taken: List[Tuple[int, int]] = []

        def free(s: int, e: int) -> bool:
            return all(e <= ts or s >= te for ts, te in taken)

        for ptype, pattern, validator in DETECTORS:
            if self.types and ptype not in self.types:
                continue
            for m in pattern.finditer(text):
                s, e = m.span()
                if validator and not validator(m.group()):
                    continue
                if free(s, e):
                    found.append(PIIMatch(ptype, s, e, m.group()))
                    taken.append((s, e))
        if self.names and (not self.types or "PERSON" in self.types):
            for s, e, val in self.names.detect(text):
                if free(s, e):
                    found.append(PIIMatch("PERSON", s, e, val))
                    taken.append((s, e))
        return sorted(found, key=lambda m: m.start)


def redact(text: str, matches: Sequence[PIIMatch], types: Optional[Sequence[str]] = None) -> str:
    """Replace matched spans (optionally only of ``types``) with ``<TYPE>`` tokens."""
    out, last = [], 0
    for m in sorted(matches, key=lambda m: m.start):
        if types is not None and m.type not in types:
            continue
        out.append(text[last : m.start])
        out.append(PLACEHOLDER.format(type=m.type))
        last = m.end
    out.append(text[last:])
    return "".join(out)


def count_by_type(matches: Sequence[PIIMatch]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in matches:
        counts[m.type] = counts.get(m.type, 0) + 1
    return counts
