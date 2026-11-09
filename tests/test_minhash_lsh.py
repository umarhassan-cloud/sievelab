import random

import numpy as np
import pytest

from dataforge.dedup import (
    ExactDeduplicator,
    MinHasher,
    MinHashLSH,
    NearDuplicateDeduplicator,
    UnionFind,
    candidate_probability,
    estimate_jaccard,
    jaccard,
    optimal_param,
    shingles,
)

WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho sigma tau "
         "upsilon phi chi psi omega river mountain ocean forest desert valley").split()


def rand_doc(rng, n=120):
    return " ".join(rng.choice(WORDS) for _ in range(n))


def mutate(rng, text, n_edits):
    toks = text.split()
    for _ in range(n_edits):
        toks[rng.randrange(len(toks))] = rng.choice(WORDS)
    return " ".join(toks)


def test_signature_shape_and_determinism():
    h = MinHasher(num_perm=64, seed=3)
    a = h.signature("the quick brown fox jumps over the lazy dog again and again")
    b = MinHasher(num_perm=64, seed=3).signature("the quick brown fox jumps over the lazy dog again and again")
    assert a.shape == (64,) and a.dtype == np.uint64
    assert np.array_equal(a, b)


def test_minhash_estimate_tracks_true_jaccard():
    rng = random.Random(0)
    h = MinHasher(num_perm=256, ngram=3)
    errs = []
    for _ in range(30):
        a = rand_doc(rng)
        b = mutate(rng, a, rng.randint(1, 40))
        true = jaccard(shingles(a, 3), shingles(b, 3))
        est = estimate_jaccard(h.signature(a), h.signature(b))
        errs.append(abs(true - est))
    # standard error of the estimator is sqrt(J(1-J)/k) <= 0.031 for k=256
    assert np.mean(errs) < 0.03
    assert max(errs) < 0.12


def test_shingle_normalisation_folds_arabic_variants_and_digits():
    urdu_farsi_yeh = "یہ ایک کتاب ہے جو طلبہ کے لیے لکھی گئی ہے"
    arabic_yeh = urdu_farsi_yeh.replace("\u06cc", "\u064a")
    assert shingles(urdu_farsi_yeh) == shingles(arabic_yeh)
    assert shingles("call 0300 1234567 today for the best offer") == shingles("call 0311 7654321 today for the best offer")


def test_optimal_param_and_s_curve():
    b, r = optimal_param(0.7, 128)
    assert b * r <= 128
    assert candidate_probability(0.95, b, r) > 0.99
    assert candidate_probability(0.3, b, r) < 0.05


def test_lsh_query_finds_similar():
    rng = random.Random(1)
    h = MinHasher(num_perm=128)
    lsh = MinHashLSH(threshold=0.7, num_perm=128)
    base = rand_doc(rng, 200)
    lsh.insert("base", h.signature(base))
    for i in range(20):
        lsh.insert(f"other{i}", h.signature(rand_doc(rng, 200)))
    assert "base" in lsh.query(h.signature(mutate(rng, base, 2)))


def test_union_find():
    uf = UnionFind()
    uf.union(3, 1)
    uf.union(1, 2)
    uf.union(5, 4)
    assert uf.find(3) == uf.find(2) == 1
    assert uf.find(5) == 4


def test_exact_dedup_normalises_whitespace_and_case():
    kept, removed = ExactDeduplicator().run(["a", "b", "c"], ["Hello   World!", "hello world", "different text"])
    assert kept == [0, 2] and removed == {1: "a"}


@pytest.mark.parametrize("mode", ["streaming", "mapreduce"])
def test_near_dedup_recall_and_precision(mode):
    rng = random.Random(7)
    originals = [rand_doc(rng, 150) for _ in range(40)]
    near = [mutate(rng, originals[i], 2) for i in range(0, 40, 4)]
    texts = originals + near
    res = NearDuplicateDeduplicator(threshold=0.7, mode=mode).run(texts)
    removed = set(res.removed)
    assert removed == set(range(40, 50))
    for i in removed:
        assert res.removed[i] == (i - 40) * 4


def test_streaming_and_mapreduce_agree():
    rng = random.Random(11)
    base = [rand_doc(rng, 100) for _ in range(30)]
    texts = base + [mutate(rng, base[i], 1) for i in range(0, 30, 3)] + [base[5]]
    a = NearDuplicateDeduplicator(mode="streaming").run(texts)
    b = NearDuplicateDeduplicator(mode="mapreduce").run(texts)
    assert set(a.kept) == set(b.kept)


def test_parallel_signatures_match_serial():
    from dataforge.dedup.parallel import compute_signatures

    rng = random.Random(2)
    texts = [rand_doc(rng, 50) for _ in range(250)]
    h = MinHasher(num_perm=32)
    serial = compute_signatures(texts, h, workers=1)
    parallel = compute_signatures(texts, h, workers=2)
    assert np.array_equal(serial, parallel)
