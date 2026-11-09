# Sample data

All files are generated deterministically by `python scripts/make_sample_corpus.py`
except `safety_challenge.jsonl` (hand-written) and `html/` (the generator writes
three hand-authored pages).

| File | Docs | Purpose |
|---|---:|---|
| `corpus.jsonl` | 2,594 | Crawl-like multilingual corpus with ground truth in `meta._gold` (language, type, 0-5 quality, duplicate group, injected PII, toxicity). Contains ~10 % exact and ~8 % near duplicates. |
| `cc_sample.warc.gz` | 70 | WARC `response` records with full page chrome (nav, cookie banner, sidebar, footer, scripts). 40 re-host articles from `corpus.jsonl`, so dedup must work across sources. |
| `cc_sample.wet` | 40 | WET `conversion` records (plain text) in 9 languages. |
| `html/` | 3 | Stand-alone HTML pages (English article, Urdu RTL article, spam page). |
| `quality_train.jsonl` | 1,600 | Training pool for the educational-quality classifier (en / ur / Roman Urdu). |
| `heldout_eval.jsonl` | 240 | Held-out educational documents (120 en + 120 ur) built only from the *eval* split of sentences and fact rows; used for the LM experiment. |
| `safety_challenge.jsonl` | 40 | Hand-written out-of-distribution PII and toxicity cases (obfuscated PII, paraphrased abuse, benign uses of lexicon words). |

## Languages

English (`en`), Urdu (`ur`), Roman Urdu (`ur_Latn`), Arabic (`ar`), Persian (`fa`),
Pashto (`ps`), Sindhi (`sd`), Hindi (`hi`), Swahili (`sw`), French (`fr`), Spanish (`es`).

## Provenance and license

All text was written for this project or generated from templates over public-domain
facts (arithmetic, unit conversions, capitals, planets, animals, dates of inventions).
No third-party web content is included. Released under the repository's MIT license.

The toxicity examples are deliberately mild (insults, threats, dehumanising phrases)
and contain no slurs. PII values are random and do not belong to real people; credit
card numbers are random Luhn-valid numbers and test numbers (4111…, 5500…).
