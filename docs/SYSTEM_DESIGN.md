# System design: scaling DataForge to Common Crawl

The repository runs on a laptop-sized sample, but every stage is written so it maps
onto a distributed engine without changing semantics. This document sketches the
production design for one Common Crawl snapshot (~3 B pages, ~100 TB WARC / ~8 TB WET,
roughly 90k WET files).

## 1. Shape of the workload

| Stage | Parallelism | State | Notes |
|---|---|---|---|
| Extract, LID, heuristics, quality, PII, toxicity | per document | none | embarrassingly parallel map over WARC/WET shards |
| Exact dedup | global | hash → first id | shuffle on hash |
| Near dedup | global | LSH buckets | shuffle on (band, band-hash) |

Only dedup needs a shuffle. Everything else is a `map` over shards, so the pipeline is
two jobs:

1. **Per-shard job** (map only): read WARC/WET → extract → LID → heuristics → compute
   SHA-256 and MinHash signature → write Parquet with `(doc_id, text, lang, sig[128], sha256, ...)`.
2. **Global dedup job** (shuffle) → list of doc ids to drop.
3. **Per-shard job** again: anti-join with the drop list → quality classifier → PII →
   toxicity → final Parquet, partitioned by language.

Quality scoring runs *after* dedup on purpose: at FineWeb scale the classifier is the
most expensive model in the pipeline, and dedup removes a large fraction of documents
(FineWeb reports MinHash dedup removing a substantial share of each dump).

## 2. MinHash LSH on Spark

`dataforge/dedup/dedup.py::_run_mapreduce` is the single-node reference for this job.

```text
docs: (doc_id, sig[0..127])
  └─ flatMap: for b in 0..B-1: emit ((b, hash(sig[b*r:(b+1)*r])), doc_id)       # B = 14 bands, r = 9 rows
       └─ groupByKey → buckets
            └─ flatMap: emit star edges (min_id, other) for each bucket        # avoids O(k²) blow-up
                 └─ distinct → edges
                      └─ (optional) verify: join both signatures, keep est. Jaccard ≥ τ
                           └─ connected components (GraphFrames / iterative min-label propagation)
                                └─ keep min doc_id per component; others → drop list
```

Design decisions:

* **Band parameters.** `optimal_param(τ, num_perm)` minimises the weighted area of false
  positives below τ and false negatives above τ. For τ = 0.7 and 128 permutations it picks
  14 bands × 9 rows, close to FineWeb's 14 × 8 (112 permutations). More bands raise recall
  at the cost of shuffle volume (`B × N` records).
* **Star edges instead of all pairs.** A bucket of size k produces k-1 edges instead of
  k(k-1)/2. Connected components are unchanged; only verification granularity changes.
* **Verification.** Optional; on Spark it costs one more join. Without it, transitive
  chains (A≈B, B≈C, A≉C) can over-merge. DataForge verifies by default; in the sample the
  exact-Jaccard precision of removals is ~0.99.
* **Shuffle volume.** 3 B docs × 14 bands × ~16 B ≈ 670 GB, which is fine for a Spark cluster.
  Partition by `band` first so each band's groupBy is independent and can be run
  sequentially if memory is tight (the approach of `text-dedup`'s Spark script).
* **Connected components.** Iterative min-label propagation converges in O(diameter)
  rounds; duplicate graphs are shallow (diameter < 10 in practice).

## 3. Ray variant

`dataforge/dedup/parallel.py` already distributes signature computation with Ray
(`parallel_backend: ray`). For the global step:

* Ray Data `map_batches` computes signatures and emits `(band, hash, id)` rows.
* `groupby("band", "hash")` performs the shuffle (push-based shuffle for >1 TB).
* Union-find is run per band partition and the partial forests are merged on the driver
  (forests are small: only documents that have at least one duplicate appear).

## 4. Cost model (one CC snapshot, order-of-magnitude)

| Step | Throughput / core | Core-hours |
|---|---|---|
| WARC → text (trafilatura-class extractor) | ~50 docs/s | ~17k |
| LID + heuristics (this repo: 0.15 ms LID) | ~2k docs/s | ~400 |
| MinHash 128 perms (numpy, word 5-grams) | ~1-2k docs/s | ~600 |
| Edu classifier, TF-IDF + linear | ~5k docs/s | ~150 |
| Edu classifier, small transformer (FineWeb-Edu style) | GPU: ~2k docs/s/GPU | ~400 GPU-h |

Extraction dominates, so the first thing to do at scale is to start from WET files
(pre-extracted) or a faster extractor (resiliparse). MinHash is the next bottleneck;
switching `sha1_hash32` to xxhash and vectorising across documents gives ~5×.

## 5. Incremental snapshots

New snapshots should dedup against everything already released:

* Persist `(band, band_hash) → representative_id` tables per language (≈ 14 × 16 B per kept doc).
* For a new snapshot, compute signatures, look up each band in the persisted tables
  (a broadcast/semi-join), and only run connected components on new documents.
* Exact hashes go into a Bloom filter (1 % FPR ≈ 1.2 GB per billion docs).

## 6. Reproducibility and auditability

* Every run writes `manifest.json` with the full resolved config, per-stage counts,
  rejection reasons and timing. Configs are versioned in `configs/`.
* Rejected documents are kept with `stage` and `reason`, which allows threshold changes to be
  re-evaluated without re-running upstream stages, and supports data-subject deletion
  requests (search `rejected.jsonl` and `clean.parquet` by URL).
* All randomness (MinHash permutations, synthetic corpus, subsampling) is seeded.

## 7. Low-resource languages at scale

Global thresholds tuned on English remove disproportionately more Urdu (shorter
words, no letter case, script-mixed numerals, no stop-word list in most tools). In
production:

* Calibrate quality thresholds **per language** by percentile of the classifier score on a
  human-rated sample, not by a single global cut-off (`quality.thresholds` is per language).
* Track per-language retention in the manifest (`lang_mix_in` → `lang_mix_out`) and alert if
  a language's retention drops relative to the previous snapshot.
* Prefer `redact_sentences` over `drop` for toxicity in languages where every good
  document matters.
