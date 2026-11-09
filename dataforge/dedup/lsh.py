"""Locality-sensitive hashing over MinHash signatures (banding technique).

With ``b`` bands of ``r`` rows, two documents with Jaccard ``s`` become
candidates with probability ``1 - (1 - s^r)^b`` — an S-curve whose inflection
is roughly ``(1/b)^(1/r)``. ``optimal_param`` picks ``(b, r)`` minimising the
weighted false-positive + false-negative area around the threshold, the same
objective datasketch uses.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Hashable, Iterator, List, Set, Tuple

import numpy as np


_trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")


def _integrate(f, a: float, b: float, n: int = 256) -> float:
    if b <= a:
        return 0.0
    xs = np.linspace(a, b, n)
    return float(_trapezoid(f(xs), xs))


def candidate_probability(s: float, bands: int, rows: int) -> float:
    return 1.0 - (1.0 - s**rows) ** bands


def optimal_param(threshold: float, num_perm: int, fp_weight: float = 0.5, fn_weight: float = 0.5) -> Tuple[int, int]:
    best, min_err = (1, num_perm), float("inf")
    for b in range(1, num_perm + 1):
        for r in range(1, num_perm // b + 1):
            fp = _integrate(lambda s: 1 - (1 - s**r) ** b, 0.0, threshold)
            fn = _integrate(lambda s: (1 - s**r) ** b, threshold, 1.0)
            err = fp_weight * fp + fn_weight * fn
            if err < min_err:
                best, min_err = (b, r), err
    return best


class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[Hashable, Hashable] = {}

    def find(self, x: Hashable) -> Hashable:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # keep the smaller key as root so "first seen" wins for integer keys
            if rb < ra:  # type: ignore[operator]
                ra, rb = rb, ra
            self.parent[rb] = ra


class MinHashLSH:
    def __init__(self, threshold: float = 0.8, num_perm: int = 128, bands: int = None, rows: int = None):
        if bands is None or rows is None:
            bands, rows = optimal_param(threshold, num_perm)
        if bands * rows > num_perm:
            raise ValueError("bands * rows must be <= num_perm")
        self.threshold = threshold
        self.num_perm = num_perm
        self.bands = bands
        self.rows = rows
        self.tables: List[Dict[bytes, List[Hashable]]] = [defaultdict(list) for _ in range(bands)]

    def band_keys(self, sig: np.ndarray) -> List[bytes]:
        r = self.rows
        return [sig[i * r : (i + 1) * r].tobytes() for i in range(self.bands)]

    def insert(self, key: Hashable, sig: np.ndarray) -> None:
        for table, bk in zip(self.tables, self.band_keys(sig)):
            table[bk].append(key)

    def query(self, sig: np.ndarray) -> Set[Hashable]:
        out: Set[Hashable] = set()
        for table, bk in zip(self.tables, self.band_keys(sig)):
            out.update(table.get(bk, ()))
        return out

    def buckets(self) -> Iterator[List[Hashable]]:
        for table in self.tables:
            for members in table.values():
                if len(members) > 1:
                    yield members
