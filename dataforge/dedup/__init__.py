from dataforge.dedup.dedup import DedupResult, NearDuplicateDeduplicator
from dataforge.dedup.exact import ExactDeduplicator, content_hash
from dataforge.dedup.lsh import MinHashLSH, UnionFind, candidate_probability, optimal_param
from dataforge.dedup.minhash import MinHasher, estimate_jaccard, jaccard, shingles

__all__ = [
    "DedupResult", "NearDuplicateDeduplicator", "ExactDeduplicator", "content_hash", "MinHashLSH",
    "UnionFind", "candidate_probability", "optimal_param", "MinHasher", "estimate_jaccard", "jaccard", "shingles",
]
