from __future__ import annotations

from typing import Sequence

from sievelab.text_utils import words


def ngram_overlap(train_texts: Sequence[str], eval_texts: Sequence[str], n: int = 13) -> float:
    """Fraction of evaluation word n-grams that also occur in the training texts
    (the GPT-3 / Llama-style 13-gram contamination check)."""
    train = set()
    for t in train_texts:
        toks = words(t)
        train.update(tuple(toks[i : i + n]) for i in range(len(toks) - n + 1))
    hits = total = 0
    for t in eval_texts:
        toks = words(t)
        for i in range(len(toks) - n + 1):
            total += 1
            hits += tuple(toks[i : i + n]) in train
    return hits / total if total else 0.0
