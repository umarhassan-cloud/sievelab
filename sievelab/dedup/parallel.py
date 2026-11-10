"""Parallel MinHash signature computation.

Signature computation is embarrassingly parallel and dominates dedup cost
(hashing every shingle ``num_perm`` times), so it is the part distributed
across processes (``multiprocessing``) or a cluster (``ray``). LSH banding and
connected components are cheap by comparison; see ``docs/SYSTEM_DESIGN.md``
for how they shard on Spark/Ray at Common Crawl scale.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from sievelab.dedup.minhash import MinHasher

_WORKER_HASHER = None


def _init_worker(params: dict) -> None:
    global _WORKER_HASHER
    _WORKER_HASHER = MinHasher(**params)


def _signature_chunk(texts: List[str]) -> np.ndarray:
    assert _WORKER_HASHER is not None
    return np.stack([_WORKER_HASHER.signature(t) for t in texts]) if texts else np.empty((0, _WORKER_HASHER.num_perm), dtype=np.uint64)


def _chunks(seq: Sequence[str], n: int) -> List[List[str]]:
    size = max(1, (len(seq) + n - 1) // n)
    return [list(seq[i : i + size]) for i in range(0, len(seq), size)]


def compute_signatures(
    texts: Sequence[str], hasher: MinHasher, workers: int = 1, backend: str = "multiprocessing"
) -> np.ndarray:
    """Return a ``(len(texts), num_perm)`` uint64 matrix of signatures."""
    if not texts:
        return np.empty((0, hasher.num_perm), dtype=np.uint64)
    if workers <= 1 or len(texts) < 200:
        return np.stack([hasher.signature(t) for t in texts])

    chunks = _chunks(texts, workers * 4)
    if backend == "ray":
        import ray  # type: ignore

        if not ray.is_initialized():
            ray.init(num_cpus=workers, ignore_reinit_error=True, log_to_driver=False)
        params = hasher.params()

        @ray.remote
        def _ray_chunk(chunk: List[str]) -> np.ndarray:
            h = MinHasher(**params)
            return np.stack([h.signature(t) for t in chunk])

        return np.concatenate(ray.get([_ray_chunk.remote(c) for c in chunks]))

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(hasher.params(),)) as pool:
        parts = pool.map(_signature_chunk, chunks)
    return np.concatenate(parts)
