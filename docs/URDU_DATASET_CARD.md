---
language:
  - ur
license: mit
pretty_name: DataForge Urdu-Edu (sample)
size_categories:
  - n<1K
task_categories:
  - text-generation
task_ids:
  - language-modeling
source_datasets:
  - original
tags:
  - urdu
  - low-resource
  - educational
  - pretraining
  - synthetic
configs:
  - config_name: default
    data_files: outputs/urdu/clean.parquet
---

# Dataset card: DataForge Urdu-Edu (sample scale)

## Summary

**DataForge Urdu-Edu** is an educational-quality Urdu (`ur`, Arabic script, Nastaliq)
pretraining corpus produced by the DataForge pipeline (`configs/urdu.yaml`). It
follows the FineWeb-Edu recipe (language ID → quality heuristics → MinHash dedup →
educational-value classifier) and adds PII redaction and sentence-level toxicity
removal tuned for Urdu.

This release is **sample scale** (273 documents, ~135k characters). It exists to
document the recipe, the per-stage filtering rates and the known failure modes, so
the same card can be regenerated unchanged when the pipeline is pointed at real
Common Crawl WET files.

| | |
|---|---|
| Curated by | Umar Hassan |
| Language | Urdu (`ur`), Arabic script |
| Documents | 273 |
| Characters | 134,561 |
| Whitespace tokens | ~30,400 (median 113 per document) |
| Format | JSONL and Parquet (`id, text, url, source, lang, lang_score, edu_score, meta`) |
| License | MIT (all text is original or generated from public-domain facts) |
| Reproduce | `python scripts/run_pipeline.py --config configs/urdu.yaml` |

## Uses

**Intended:** continued pretraining or ablations of small models on educational
Urdu; testing data-curation tooling for Arabic-script languages; teaching material for
data-centric ML.

**Out of scope:** benchmarking downstream Urdu capability (the sample is too small and
partly templated); any use that relies on the factual statements being exhaustive or
current.

## Dataset structure

```json
{
  "id": "doc-01234",
  "text": "اس سبق میں ہم کیمیا کے بارے میں سیکھیں گے۔ ...",
  "url": "https://ur.example.org/edu/1234",
  "source": "data/samples/corpus.jsonl",
  "lang": "ur",
  "lang_score": 0.97,
  "meta": {
    "heuristics": {"failed": [], "n_words": 142},
    "quality": {"edu_score": 3.84, "edu_int": 4},
    "pii": {"counts": {"PHONE": 1}, "redacted": 1},
    "toxicity": {"score": 0.0, "hits": []}
  }
}
```

PII spans are replaced with typed placeholders: `<EMAIL>`, `<PHONE>`, `<CNIC>`,
`<CREDIT_CARD>`, `<IBAN>`, `<IP_ADDRESS>`.

## Source data

The input is DataForge's sample crawl: a JSONL corpus, a WARC file with full HTML page
chrome, a WET file and stand-alone HTML pages, in 11 languages (2,707 documents in
total). Urdu content consists of:

* hand-written explanatory sentences (science, history, geography, literature);
* fact tables rendered through several templates (chemical elements, organs, concepts,
  historical events, capitals, planets, animals, inventions);
* procedurally generated, verifiably correct statements (arithmetic, unit conversion,
  geometry, primes, temperature conversion);
* deliberately low-quality text: casual chat, spam/SEO pages, navigation boilerplate,
  injected duplicates, PII and abusive comments.

Sources of the final documents: 267 from the JSONL corpus, 4 from WARC pages, 1 from the
WET file and 1 hand-written HTML article.

## Curation pipeline and filtering rates

| Stage | In | Out | Removed | % of stage input | Main reasons |
|---|---:|---:|---:|---:|---|
| Ingest (all languages) | 2,707 | 2,707 | 0 | 0.0% | — |
| Language ID (keep `ur`, score ≥ 0.4) | 2,707 | 1,030 | 1,677 | 62.0% | en 967, hi 95, Roman Urdu 93, fr 88 … |
| Heuristic rules | 1,030 | 667 | 363 | 35.2% | exclamation spam 178, non-alphabetic tokens 119, repeated lines 29 |
| Dedup (exact + MinHash, τ = 0.7) | 667 | 551 | 116 | 17.4% | near-duplicate 68, exact 48 |
| Educational classifier (score ≥ 2.5) | 551 | 273 | 278 | 50.5% | low educational value |
| PII | 273 | 273 | 0 | 0.0% | 47 documents redacted, none dropped |
| Toxicity (`redact_sentences`) | 273 | 273 | 0 | 0.0% | 16 documents had abusive sentences removed |
| **Final** | | **273** | | **26.5 % of Urdu input** | |

Educational score distribution of kept documents (rounded): 3 → 92, 4 → 176, 5 → 5.

### Measured quality of each filter (against injected ground truth)

| Filter | Metric | Value |
|---|---|---:|
| Language ID (document level) | accuracy on Urdu documents | 1.000 |
| Language ID (sentence level, `scripts/eval_langid.py`) | Urdu accuracy | 0.982 |
| Dedup | recall on injected duplicates | 0.970 |
| Dedup | precision (exact Jaccard ≥ τ) | 0.979 |
| Edu classifier | quadratic-weighted κ vs gold (Urdu) | 0.881 |
| Edu classifier | keep-decision precision / recall | 0.970 / 0.996 |
| PII (injected) | recall by type (email, phone, CNIC, card) | 1.00 |
| Toxicity (injected) | precision / recall | 1.00 / 1.00 |

Final composition by gold type: 177 long educational, 82 short educational, 8 news-style
documents; no spam, casual chat or boilerplate survived.

## Urdu-specific processing decisions

* **Urdu vs Persian vs Arabic vs Pashto vs Sindhi.** All five share the Arabic script.
  The language identifier uses Urdu-only letters (ٹ ڈ ڑ ں ے ہ ھ) and Urdu function words
  (کے، کی، ہے، میں …); generic tools often confuse these languages. On the sentence-level
  benchmark langdetect labels every Pashto and Sindhi sentence as Persian.
* **Roman Urdu is a separate language code (`ur_Latn`)** and is excluded from this
  Arabic-script release. It is common on Pakistani social media and is scored as a
  distinct class rather than being misread as English or Indonesian.
* **Normalisation for dedup.** Arabic yeh/kaf (ي ك) are folded into Urdu forms (ی ک) and
  diacritics (zer, zabar, pesh) are stripped before hashing, so copies typed on different
  keyboards are detected as duplicates.
* **Urdu digits.** PII detectors match Extended Arabic-Indic digits (۰-۹), e.g.
  `۰۳۰۰-۱۲۳۴۵۶۷`.
* **Numerals.** Gopher's rule that 80 % of words must contain a letter removed 41 % of
  math-heavy educational documents in the sample; DataForge uses 0.6 (0 % of educational
  Urdu documents removed).
* **Toxicity retention.** `redact_sentences` keeps the article and removes only the abusive
  comment: 16 documents kept that a `drop` policy would have discarded.

## Bias, risks and limitations

* **Synthetic and templated.** Much of the Urdu text comes from templates and procedural
  generators. It is grammatical and factually correct, but less diverse in style and
  register than web Urdu. Treat it as a recipe demonstration, not a representative corpus.
* **Teacher bias.** Educational labels come from a rule-based annotator distilled into a
  classifier. Keywords favour school-science and history registers and under-score
  poetry, religious scholarship and journalism, which are major genres of real Urdu web text.
* **Lexicon toxicity is brittle.** On the hand-written challenge set the lexicon has 0.4
  precision and 0.4 recall. It misses paraphrased threats and flags literal uses such as
  گدھا (donkey) and جاہل (uneducated). See `experiments/results/safety_challenge.json`.
* **PII regexes miss unformatted identifiers** (CNICs without dashes, spelled-out numbers):
  challenge-set entity recall is 0.64.
* **Dialect and script coverage.** No Shahmukhi Punjabi, Saraiki or Hindko, which are often
  mislabelled as Urdu on the web.

## Recommendations

Re-calibrate the educational threshold per language on a human-rated Urdu sample;
replace the lexicon with a learned multilingual toxicity classifier; and add a
human-rated audit of 200+ documents before using a real-crawl version for training.

## Citation

```bibtex
@software{hassan2026dataforge,
  author = {Hassan, Umar},
  title  = {DataForge: Web-Scale Data Curation with First-Class Urdu Support},
  year   = {2026}
}
```
