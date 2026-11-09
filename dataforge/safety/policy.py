"""Configurable retention policies for PII and toxicity.

PII actions (per entity type):
  ``redact``  replace span with ``<TYPE>`` and keep the document
  ``drop``    remove the whole document
  ``flag``    keep text unchanged, record the match in ``meta["pii"]``
  ``keep``    ignore entirely

Toxicity actions:
  ``drop``             remove documents with score >= threshold
  ``redact_sentences`` remove only the offending sentences, keep the rest
  ``flag``             keep, annotate ``meta["toxicity"]``
  ``keep``             no-op
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

PII_ACTIONS = {"redact", "drop", "flag", "keep"}
TOX_ACTIONS = {"drop", "redact_sentences", "flag", "keep"}

DEFAULT_PII_ACTIONS = {
    "EMAIL": "redact",
    "PHONE": "redact",
    "CNIC": "redact",
    "CREDIT_CARD": "redact",
    "IBAN": "redact",
    "IP_ADDRESS": "redact",
    "PERSON": "flag",
}


@dataclass
class PIIPolicy:
    actions: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PII_ACTIONS))
    max_pii_per_100_words: Optional[float] = 5.0
    enable_names: bool = True

    def __post_init__(self) -> None:
        bad = {a for a in self.actions.values() if a not in PII_ACTIONS}
        if bad:
            raise ValueError(f"Unknown PII action(s): {bad}")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "PIIPolicy":
        d = d or {}
        actions = dict(DEFAULT_PII_ACTIONS)
        actions.update(d.get("actions") or {})
        return cls(actions=actions, max_pii_per_100_words=d.get("max_pii_per_100_words", 5.0),
                   enable_names=d.get("enable_names", True))


@dataclass
class ToxicityPolicy:
    action: str = "drop"
    threshold: float = 0.5
    classifier_path: Optional[str] = None
    lexicon_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.action not in TOX_ACTIONS:
            raise ValueError(f"Unknown toxicity action: {self.action}")

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "ToxicityPolicy":
        d = d or {}
        return cls(action=d.get("action", "drop"), threshold=d.get("threshold", 0.5),
                   classifier_path=d.get("classifier_path"), lexicon_path=d.get("lexicon_path"))
