"""MinHash signatures over word (or character) n-gram shingles.

Hashing follows the classic Broder construction used by datasketch and the
FineWeb / text-dedup pipelines: each shingle is hashed to 32 bits (SHA-1
prefix), then ``num_perm`` universal hash functions
``h_i(x) = ((a_i * x + b_i) mod p) & (2^32 - 1)`` with ``p = 2^61 - 1`` simulate
random permutations. The signature is the element-wise minimum over shingles,
and ``P[sig_A[i] == sig_B[i]] = Jaccard(A, B)``.

uint64 multiplication intentionally wraps (identical to datasketch), so
signatures produced here are bit-compatible with ``datasketch.MinHash`` given
the same seed and hash function.
"""

from __future__ import annotations

import hashlib
import struct
from typing import Iterable, Set

import numpy as np

from sievelab.text_utils import normalize_text, words

MERSENNE_PRIME = np.uint64((1 << 61) - 1)
MAX_HASH = np.uint64((1 << 32) - 1)
_CHUNK = 8192


def sha1_hash32(data: bytes) -> int:
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]


def shingles(text: str, ngram: int = 5, char_level: bool = False) -> Set[str]:
    """Normalised n-gram shingles. Digits are mapped to 0 and Arabic-script
    letter variants are folded so trivially re-typed copies collide."""
    if char_level:
        norm = normalize_text(text, fold_arabic=True, normalize_digits=True)
        if not norm:
            return set()
        if len(norm) <= ngram:
            return {norm}
        return {norm[i : i + ngram] for i in range(len(norm) - ngram + 1)}
    toks = words(text, fold_arabic=True, normalize_digits=True)
    if not toks:
        return set()
    if len(toks) < ngram:
        return {" ".join(toks)}
    return {" ".join(toks[i : i + ngram]) for i in range(len(toks) - ngram + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    def __init__(self, num_perm: int = 128, ngram: int = 5, seed: int = 1, char_level: bool = False):
        if num_perm < 2:
            raise ValueError("num_perm must be >= 2")
        self.num_perm = num_perm
        self.ngram = ngram
        self.seed = seed
        self.char_level = char_level
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)
        self.b = rng.randint(0, MERSENNE_PRIME, size=num_perm, dtype=np.uint64)

    def params(self) -> dict:
        return {"num_perm": self.num_perm, "ngram": self.ngram, "seed": self.seed, "char_level": self.char_level}

    def signature_from_shingles(self, sh: Iterable[str]) -> np.ndarray:
        sh = list(sh)
        sig = np.full(self.num_perm, MAX_HASH, dtype=np.uint64)
        if not sh:
            return sig
        hv_all = np.fromiter((sha1_hash32(s.encode("utf-8")) for s in sh), dtype=np.uint64, count=len(sh))
        with np.errstate(over="ignore"):
            for start in range(0, len(hv_all), _CHUNK):
                hv = hv_all[start : start + _CHUNK]
                phv = ((hv[:, None] * self.a[None, :] + self.b[None, :]) % MERSENNE_PRIME) & MAX_HASH
                sig = np.minimum(sig, phv.min(axis=0))
        return sig

    def signature(self, text: str) -> np.ndarray:
        return self.signature_from_shingles(shingles(text, self.ngram, self.char_level))


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    return float(np.mean(sig_a == sig_b))
