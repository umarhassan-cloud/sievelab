from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

from sievelab.text_utils import normalize_text


def content_hash(text: str) -> str:
    norm = normalize_text(text, fold_arabic=True, normalize_digits=False)
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


class ExactDeduplicator:
    """Keeps the first occurrence of each normalised-content hash.

    Normalisation (NFKC, case-fold, punctuation/whitespace collapse, Arabic
    letter folding) means copies differing only in markup residue or keyboard
    layout are treated as exact duplicates.
    """

    def __init__(self) -> None:
        self.seen: Dict[str, str] = {}

    def run(self, ids: Sequence[str], texts: Sequence[str]) -> Tuple[List[int], Dict[int, str]]:
        """Returns ``(kept_indices, {removed_index: original_id})``."""
        kept: List[int] = []
        removed: Dict[int, str] = {}
        for i, (doc_id, text) in enumerate(zip(ids, texts)):
            h = content_hash(text)
            if h in self.seen:
                removed[i] = self.seen[h]
            else:
                self.seen[h] = doc_id
                kept.append(i)
        return kept, removed
