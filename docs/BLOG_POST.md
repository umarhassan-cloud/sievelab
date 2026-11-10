# Most of the crawl is not worth training on: building a measurable curation pipeline, with Urdu in the loop

*Umar Hassan*

The pretraining results of the last two years (FineWeb-Edu, DCLM, Dolma, Phi's
"textbooks") all say the same thing: at a fixed compute budget, **which tokens you
train on matters more than most architecture tweaks**. FineWeb-Edu kept roughly a tenth of
FineWeb and matched or beat the full set on knowledge benchmarks. DCLM showed that a
fastText quality classifier is worth more than many filtering heuristics combined.

Those results come from pipelines that are expensive to run and hard to inspect. I
wanted a pipeline small enough to run on a laptop in seconds, but *instrumented* well
enough that every stage reports precision and recall, not just "removed 37 %". I also
wanted to see what happens to a low-resource language, in this case Urdu, when it passes
through filters designed around English.

This post covers what I built (SieveLab), the measurements, and five things that
surprised me.

## The setup

SieveLab runs `ingest → language ID → heuristics → dedup → educational classifier → PII →
toxicity → export`, with a manifest of counts per stage. To measure each filter, the
sample crawl is **synthetic with ground truth**: 2,707 documents in 11 languages, in
JSONL, WARC (full HTML page chrome) and WET formats. Every document carries hidden labels
(true language, document type, 0-5 educational value, duplicate group, injected PII,
injected abuse) that the filters never see. After a run, the pipeline joins its decisions
with those labels.

Synthetic data is a deliberate trade-off. It is easier than the real web, and I report
separate hand-written out-of-distribution tests to keep the numbers honest. In exchange,
every stage gets a real confusion matrix, which you almost never have for a crawl.

## The results

| Stage | Removed (% of stage input) | Measured quality |
|---|---:|---|
| Language ID (keep en, ur) | 26.6 % | 99.9 % doc accuracy; 96.6 % on hard sentence-level set |
| Heuristics | 23.8 % | all spam removed, 0 % of educational docs lost (after one fix, below) |
| Dedup (exact + MinHash) | 17.0 % | recall 0.97 on injected dups, precision 0.99 (exact Jaccard) |
| Educational classifier | 43.8 % | κ = 0.86 vs gold; keep-decision F1 0.98 |
| PII | 0 % dropped, 102 docs redacted | 1.00 recall injected; **0.64** on challenge set |
| Toxicity | 4.1 % | 1.00 on injected; **0.40 / 0.40** P/R on challenge set |

Overall, 25 % of documents survive. Then the question that matters: **is the surviving
data actually better to train on?**

## Does cleaning help? A fixed-budget LM experiment

I built five training sets from the same crawl (`raw`, `+lang`, `+heuristics`, `+dedup`,
`full`), subsampled each to the same number of characters (the size of `full`), trained
a character-level 5-gram Kneser-Ney model on each, and evaluated on held-out educational
English and Urdu. Character level with a shared alphabet makes perplexities comparable
across training sets, which word-level perplexity with different vocabularies is not.

| Training data | Bits/char ↓ | Char perplexity ↓ | Δ vs raw |
|---|---:|---:|---:|
| raw | 1.552 | 2.932 | — |
| +lang | 1.526 | 2.880 | −1.8 % |
| +heuristics | 1.490 | 2.809 | −4.2 % |
| +dedup | 1.483 | 2.795 | −4.7 % |
| full | **1.401** | **2.642** | **−9.9 %** |

Every stage helps, and the educational classifier contributes the most, the same
ordering FineWeb-Edu reports. The most striking row isn't in the table: training on
**all** of the raw data (3× more characters) still gives 1.549 bits/char, worse than the
curated set at one third of the size.

## Five things I didn't expect

### 1. My first LM experiment was measuring repetition, not quality

The first version of the corpus had only ~200 unique educational sentences per language.
Bits-per-character got *worse* as I added clean data, which looked like a bug in my
Kneser-Ney implementation. It wasn't. More data meant more copies of the same sentences,
and the model became overconfident on them. A controlled ablation (fixed budget, fewer
unique documents) makes the effect explicit:

| Unique docs | Repeats | Δ perplexity |
|---:|---:|---:|
| 400 | 1× | — |
| 100 | 3.8× | +15.6 % |
| 25 | 15× | +49.8 % |

That is the argument for deduplication in one table. (Transformers tolerate a few epochs
better than n-gram models; Muennighoff et al. find ~4 repeats nearly free. Fifteen are not.)

### 2. A standard Gopher rule deleted 41 % of the maths

Gopher requires 80 % of words to contain a letter. Educational text about arithmetic,
unit conversion and geometry ("A rectangle 12 metres long and 7 metres wide has an area
of 84 square metres") fails that test. At 0.8 the rule removed **41 % of educational
documents** while catching no spam that other rules didn't already catch. At 0.6 it
removes 0 %. A rule copied from a paper encodes that paper's data distribution.

### 3. Off-the-shelf language ID doesn't know Pakistan's languages

On a sentence-level benchmark, langdetect labelled **every** Pashto and Sindhi sentence as
Persian, and Roman Urdu (Urdu typed in Latin script, the default on Pakistani social media)
as Indonesian, Somali or Swahili. A transparent two-step identifier (script detection, then
language-specific letters such as ٹ ڈ ڑ ں ے for Urdu and ټ ډ ښ for Pashto, plus function
words) scored 96.6 % vs 87.4 % and ran ~150× faster. langdetect is still better on short
English, so in production I would ensemble the two rather than pick one.

### 4. Distillation beat its own teacher

The educational labels come from a rule-based 0-5 rubric standing in for FineWeb-Edu's
LLM annotator. A TF-IDF ridge "student" trained on those labels agrees with the true
labels *more* than the teacher does (κ 0.890 vs 0.882). Regressing on many noisy labels
averages out idiosyncratic teacher mistakes. This is a small-scale version of why the
FineWeb-Edu classifier is usable even though the Llama-3 annotations are noisy.

### 5. Lexicons are worse than they look, especially in Urdu

On injected abuse the lexicon filter is perfect, because the abuse and the lexicon come
from the same mind. On 20 hand-written challenge cases it gets **0.40 precision and 0.40
recall**. It misses paraphrased threats ("I hope something terrible happens to your
family") and flags benign text: medicine ("smoking can kill you slowly"), recycling
("trash"), and in Urdu the literal word for donkey (گدھا), which is also a common insult.
Sentence-level scoring and phrase entries ("are vermin" rather than "vermin") help, as
does the `redact_sentences` policy, which keeps an article and removes only the abusive
comment underneath it. The real fix is a learned multilingual classifier.

## What I'd do with a real budget

1. Point it at 10 CC WET files, replace the rule teacher with an LLM annotator on 50k
   documents, and distil into a multilingual encoder classifier.
2. Calibrate the educational threshold **per language** by percentile against a
   human-rated Urdu sample. English-tuned thresholds are the main way low-resource
   languages silently disappear from web corpora.
3. Replace the n-gram/tiny-GPT proxy with 150M-parameter models on a few billion tokens,
   evaluated on Urdu and English knowledge benchmarks.
4. Run dedup on Spark with the same band/bucket/connected-components algorithm
   (`docs/SYSTEM_DESIGN.md`) and dedup incrementally across snapshots.

The code, data generator, configs and every number above are in the repository and
reproduce with four commands.
