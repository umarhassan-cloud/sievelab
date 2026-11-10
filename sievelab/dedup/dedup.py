"""Near-duplicate removal with MinHash + LSH.

Two execution modes with identical semantics (keep the first document of each
duplicate cluster):

* ``streaming`` — query-then-insert over a single in-memory LSH index. Simple,
  low memory for candidate pairs, natural for single-node runs.
* ``mapreduce`` — the formulation used on Spark/Ray at Common Crawl scale:
  ``map`` each doc to ``(band_id, band_hash) -> doc``, ``groupBy`` key to get
  buckets, emit verified edges, then connected components. Implemented here
  with plain Python data structures so it can be unit-tested against
  ``streaming``.

Candidate pairs are verified with the signature-estimated Jaccard
(``verify="minhash"``) or exact shingle Jaccard (``verify="exact"``) before
being merged, which removes most LSH false positives.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from sievelab.dedup.lsh import MinHashLSH, UnionFind, optimal_param
from sievelab.dedup.minhash import MinHasher, estimate_jaccard, jaccard, shingles
from sievelab.dedup.parallel import compute_signatures


@dataclass
class DedupResult:
    kept: List[int]
    removed: Dict[int, int]  # removed index -> index of kept representative
    similarity: Dict[int, float] = field(default_factory=dict)
    n_candidate_pairs: int = 0
    n_verified_pairs: int = 0
    bands: int = 0
    rows: int = 0

    @property
    def n_clusters_with_dups(self) -> int:
        return len(set(self.removed.values()))


class NearDuplicateDeduplicator:
    def __init__(
        self,
        threshold: float = 0.7,
        num_perm: int = 128,
        ngram: int = 5,
        seed: int = 1,
        char_level: bool = False,
        mode: str = "streaming",
        verify: str = "minhash",
        workers: int = 1,
        parallel_backend: str = "multiprocessing",
        backend: str = "python",
        bands: Optional[int] = None,
        rows: Optional[int] = None,
    ):
        self.threshold = threshold
        self.hasher = MinHasher(num_perm=num_perm, ngram=ngram, seed=seed, char_level=char_level)
        self.mode = mode
        self.verify = verify
        self.workers = workers
        self.parallel_backend = parallel_backend
        self.backend = backend
        if bands is None or rows is None:
            bands, rows = optimal_param(threshold, num_perm)
        self.bands, self.rows = bands, rows

    def signatures(self, texts: Sequence[str]) -> np.ndarray:
        if self.backend == "datasketch":
            return self._datasketch_signatures(texts)
        return compute_signatures(texts, self.hasher, self.workers, self.parallel_backend)

    def _datasketch_signatures(self, texts: Sequence[str]) -> np.ndarray:
        from datasketch import MinHash  # type: ignore

        sigs = []
        for t in texts:
            m = MinHash(num_perm=self.hasher.num_perm, seed=self.hasher.seed)
            m.update_batch([s.encode("utf-8") for s in shingles(t, self.hasher.ngram, self.hasher.char_level)])
            sigs.append(m.hashvalues.astype(np.uint64))
        return np.stack(sigs) if sigs else np.empty((0, self.hasher.num_perm), dtype=np.uint64)

    def _similar(self, i: int, j: int, sigs: np.ndarray, texts: Sequence[str], cache: Dict[int, set]) -> float:
        if self.verify == "exact":
            for k in (i, j):
                if k not in cache:
                    cache[k] = shingles(texts[k], self.hasher.ngram, self.hasher.char_level)
            return jaccard(cache[i], cache[j])
        return estimate_jaccard(sigs[i], sigs[j])

    def run(self, texts: Sequence[str]) -> DedupResult:
        sigs = self.signatures(texts)
        if self.mode == "mapreduce":
            return self._run_mapreduce(texts, sigs)
        return self._run_streaming(texts, sigs)

    def _run_streaming(self, texts: Sequence[str], sigs: np.ndarray) -> DedupResult:
        lsh = MinHashLSH(self.threshold, self.hasher.num_perm, self.bands, self.rows)
        uf = UnionFind()
        cache: Dict[int, set] = {}
        kept: List[int] = []
        removed: Dict[int, int] = {}
        sim: Dict[int, float] = {}
        n_cand = n_ver = 0
        for i in range(len(texts)):
            cands = lsh.query(sigs[i])
            n_cand += len(cands)
            best_j, best_s = None, -1.0
            for j in cands:
                s = self._similar(i, j, sigs, texts, cache)
                if s >= self.threshold and s > best_s:
                    best_j, best_s = j, s
            if best_j is not None:
                n_ver += 1
                uf.union(best_j, i)
                removed[i] = uf.find(i)
                sim[i] = best_s
            else:
                uf.find(i)
                kept.append(i)
            lsh.insert(i, sigs[i])
        return DedupResult(kept, removed, sim, n_cand, n_ver, self.bands, self.rows)

    def _run_mapreduce(self, texts: Sequence[str], sigs: np.ndarray) -> DedupResult:
        r = self.rows
        # map: (band, band_hash) -> [doc ...]   (a Spark groupByKey / Ray shuffle)
        buckets: Dict[tuple, List[int]] = defaultdict(list)
        for i in range(len(texts)):
            for b in range(self.bands):
                buckets[(b, sigs[i, b * r : (b + 1) * r].tobytes())].append(i)
        # reduce: candidate edges (all pairs within small buckets, star edges for large ones)
        edges = set()
        for members in buckets.values():
            if len(members) < 2:
                continue
            if len(members) <= 64:
                edges.update((a, b) for x, a in enumerate(members) for b in members[x + 1 :])
            else:
                edges.update((members[0], m) for m in members[1:])
        cache: Dict[int, set] = {}
        uf = UnionFind()
        best_sim: Dict[int, float] = {}
        n_ver = 0
        for a, b in edges:
            s = self._similar(a, b, sigs, texts, cache)
            if s >= self.threshold:
                n_ver += 1
                uf.union(a, b)
                hi = max(a, b)
                best_sim[hi] = max(best_sim.get(hi, 0.0), s)
        kept, removed = [], {}
        for i in range(len(texts)):
            root = uf.find(i)
            if root == i:
                kept.append(i)
            else:
                removed[i] = root
        sim = {i: best_sim.get(i, self.threshold) for i in removed}
        return DedupResult(kept, removed, sim, len(edges), n_ver, self.bands, self.rows)
