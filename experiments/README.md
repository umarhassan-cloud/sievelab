# Experiments

All results in `results/` are produced by scripts in this repository on the bundled
sample data. Everything is CPU-only and seeded.

| Experiment | Command | Output |
|---|---|---|
| Raw vs. cleaned LM (main proof) | `python scripts/prove_cleaning_helps.py` | `results/lm_comparison.{md,json}` |
| Repetition ablation (why dedup) | `python experiments/repetition_ablation.py` | `results/repetition_ablation.{md,json}` |
| Quality classifier distillation | `python scripts/train_quality_clf.py` | `results/quality_classifier.json` |
| Sentence-level language ID | `python scripts/eval_langid.py` | `results/langid_benchmark.json` |
| Safety challenge set | `python scripts/eval_safety_challenge.py` | `results/safety_challenge.json` |

## 1. Raw vs. cleaned training data

**Question.** At a *fixed data budget*, does curated data give a better language model
than the raw crawl it was curated from?

**Protocol.**

1. Build five training sets from the same ingested crawl: `raw`, `+lang`, `+heuristics`,
   `+dedup` and `full` (+ edu classifier, PII, toxicity). Each is the actual pipeline output
   with later stages disabled.
2. Subsample every set to the character count of `full` (3 seeds for the non-full sets).
3. Train a character-level 5-gram interpolated Kneser-Ney LM per set, with a shared
   vocabulary (union of all training and eval characters plus an unknown symbol), so
   perplexities are directly comparable.
4. Evaluate on `data/samples/heldout_eval.jsonl`: 240 educational documents (120 English,
   120 Urdu) built only from held-out sentences and fact rows. The 13-gram overlap
   between eval and train is ~1.5 % for every condition (shared lesson-intro templates),
   so the comparison is not driven by contamination.
5. Repeat the raw-vs-full comparison with a tiny character-level GPT at identical compute
   (same architecture, steps, batch size and context).

**Why character level?** Word-level perplexities are not comparable across training
sets with different vocabularies. Characters with a shared, fixed alphabet are.

**What it does and does not show.** The eval set is educational English/Urdu, which is
the distribution the pipeline is designed to select for. This is the FineWeb-Edu claim
(better on knowledge/educational benchmarks at equal tokens), not a claim that curation
helps on every distribution: a model evaluated on Roman-Urdu chat would prefer `raw`.

## 2. Repetition ablation

Fixed budget, varying the number of unique educational documents (fewer unique docs =
more repeats). This isolates the cost of duplication that dedup removes. Caveat: fewer
unique documents also means less unique information, and n-gram models are more
sensitive to repetition than transformers (Muennighoff et al., 2023, find that up to ~4
epochs of repeated data is nearly as good as fresh data for LLMs).

## 3. Quality classifier distillation

Teacher = additive 0-5 heuristic rubric (FineWeb-Edu's LLM prompt, made explicit).
Student = TF-IDF (char 2-4 + word 1-2 grams) → ridge regression, trained on teacher
labels. The student agrees with the generator's gold labels *more* than the teacher does
(quadratic κ 0.890 vs 0.882): regression on many noisy labels smooths per-document
teacher errors.

## 4. Language ID and 5. Safety challenge set

Out-of-distribution checks that keep the in-pipeline numbers (which are near-perfect on
in-distribution synthetic data) honest. See the main README for the tables.
