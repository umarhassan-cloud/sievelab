"""Character-level interpolated Kneser-Ney n-gram language model.

Character-level modelling with a *shared, fixed vocabulary* makes perplexities
comparable across models trained on different corpora (word-level perplexity
is not comparable when vocabularies differ). Unknown characters fall back to a
uniform distribution over the shared vocabulary, so every model assigns
non-zero probability to every evaluation character.

    P_KN(w | h) = max(c(h,w) - D, 0) / c(h)  +  D * N1+(h, .) / c(h) * P_KN(w | h')

Highest order uses raw counts; lower orders use continuation counts
N1+(., h', w). Single absolute discount ``D`` per order.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

BOS = "\x02"
EOS = "\x03"


class CharKneserNeyLM:
    def __init__(self, order: int = 5, discount: float = 0.75):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.D = discount
        # counts[k][(context, char)] for k = 1..order (k = n-gram length)
        self.counts: List[Dict[tuple, int]] = [dict() for _ in range(order + 1)]
        self.ctx_total: List[Dict[str, int]] = [dict() for _ in range(order + 1)]
        self.ctx_types: List[Dict[str, int]] = [dict() for _ in range(order + 1)]
        self.vocab: set = set()
        self.V = 1

    def _pad(self, text: str) -> str:
        return BOS * (self.order - 1) + text + EOS

    def fit(self, texts: Iterable[str], vocab: Sequence[str] = ()) -> "CharKneserNeyLM":
        n = self.order
        top: Dict[tuple, int] = defaultdict(int)
        for t in texts:
            s = self._pad(t)
            self.vocab.update(s)
            for i in range(n - 1, len(s)):
                top[(s[i - n + 1 : i], s[i])] += 1
        self.counts[n] = dict(top)
        # continuation counts for lower orders: N1+(. h' w) = #distinct left-extensions
        for k in range(n - 1, 0, -1):
            cont: Dict[tuple, int] = defaultdict(int)
            for (ctx, w) in self.counts[k + 1]:
                cont[(ctx[1:], w)] += 1
            self.counts[k] = dict(cont)
        for k in range(1, n + 1):
            tot: Dict[str, int] = defaultdict(int)
            typ: Dict[str, int] = defaultdict(int)
            for (ctx, _w), c in self.counts[k].items():
                tot[ctx] += c
                typ[ctx] += 1
            self.ctx_total[k], self.ctx_types[k] = dict(tot), dict(typ)
        self.vocab.update(vocab)
        self.V = len(self.vocab) + 1  # +1 for unseen characters
        return self

    def prob(self, ctx: str, w: str) -> float:
        p = 1.0 / self.V
        for k in range(1, self.order + 1):
            h = ctx[len(ctx) - (k - 1) :] if k > 1 else ""
            total = self.ctx_total[k].get(h)
            if not total:
                continue
            c = self.counts[k].get((h, w), 0)
            p = max(c - self.D, 0.0) / total + self.D * self.ctx_types[k][h] / total * p
        return p

    def _contexts(self, text: str):
        s = self._pad(text)
        n = self.order
        for i in range(n - 1, len(s)):
            yield s[i - n + 1 : i], s[i]

    def evaluate(self, texts: Sequence[str], acc_positions: int = 3000, seed: int = 0) -> Dict[str, float]:
        """Bits-per-character, char perplexity and (sampled) top-1 next-char accuracy."""
        nll, n = 0.0, 0
        positions = []
        for t in texts:
            for ctx, w in self._contexts(t):
                nll -= math.log2(self.prob(ctx, w))
                n += 1
                positions.append((ctx, w))
        rng = random.Random(seed)
        sample = rng.sample(positions, min(acc_positions, len(positions)))
        cand = sorted(self.vocab - {BOS})
        correct = sum(max(cand, key=lambda c: self.prob(ctx, c)) == w for ctx, w in sample)
        bpc = nll / max(n, 1)
        return {"bpc": round(bpc, 4), "char_ppl": round(2**bpc, 4), "top1_acc": round(correct / max(len(sample), 1), 4), "n_chars": n}
