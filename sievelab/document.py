from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """A single unit of text flowing through the pipeline.

    ``meta`` is free-form; stages write their annotations under their own key
    (``meta["quality"]``, ``meta["pii"]`` ...). Ground-truth labels for synthetic
    corpora live under ``meta["_gold"]`` and are never read by filtering stages.
    """

    id: str
    text: str
    url: Optional[str] = None
    source: Optional[str] = None
    lang: Optional[str] = None
    lang_score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Document":
        return cls(
            id=str(d["id"]),
            text=d.get("text", ""),
            url=d.get("url"),
            source=d.get("source"),
            lang=d.get("lang"),
            lang_score=d.get("lang_score"),
            meta=dict(d.get("meta") or {}),
        )

    @property
    def gold(self) -> Dict[str, Any]:
        return self.meta.get("_gold", {})
