"""Does curation help? Train small LMs on RAW vs progressively CLEANED data.

Protocol (fixed data budget, the setting FineWeb/DCLM report):
  * Build training sets from the same crawl at increasing curation levels:
      raw            everything ingested (after HTML/WARC text extraction)
      +lang          language filter only (en, ur)
      +heuristics    + Gopher/C4-style rules
      +dedup         + exact & MinHash near-dup removal
      full           + educational-quality classifier + PII + toxicity
  * Subsample every condition to the SAME number of characters (the size of
    the fully curated set), over several seeds.
  * Train a character-level 5-gram Kneser-Ney LM (and optionally a tiny GPT)
    with a shared vocabulary, and evaluate bits-per-character / perplexity /
    next-char accuracy on a held-out educational set (English + Urdu) whose
    source sentences never appear in training.

    python scripts/prove_cleaning_helps.py               # n-gram + tiny GPT (if torch)
    python scripts/prove_cleaning_helps.py --no-gpt
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sievelab.eval.contamination import ngram_overlap  # noqa: E402
from sievelab.eval.ngram_lm import CharKneserNeyLM  # noqa: E402
from sievelab.pipeline.config import load_config  # noqa: E402
from sievelab.pipeline.io import read_jsonl  # noqa: E402
from sievelab.pipeline.runner import Pipeline  # noqa: E402

CONDITIONS = [
    ("raw", None),
    ("+lang", {"heuristics": {"enabled": False}, "dedup": {"exact": False, "near": False}, "quality": {"enabled": False},
               "safety": {"pii": {"enabled": False}, "toxicity": {"enabled": False}}}),
    ("+heuristics", {"dedup": {"exact": False, "near": False}, "quality": {"enabled": False},
                     "safety": {"pii": {"enabled": False}, "toxicity": {"enabled": False}}}),
    ("+dedup", {"quality": {"enabled": False}, "safety": {"pii": {"enabled": False}, "toxicity": {"enabled": False}}}),
    ("full", {}),
]


def subsample_to_budget(texts: Sequence[str], budget: int, seed: int) -> List[str]:
    order = list(range(len(texts)))
    random.Random(seed).shuffle(order)
    out, used = [], 0
    for i in order:
        if used >= budget:
            break
        t = texts[i][: budget - used]
        out.append(t)
        used += len(t)
    return out


def eval_ngram(train: Sequence[str], evals: Dict[str, List[str]], vocab: set, order: int, acc_positions: int) -> Dict:
    lm = CharKneserNeyLM(order=order).fit(train, vocab=sorted(vocab))
    res = {lang: lm.evaluate(txts, acc_positions=acc_positions) for lang, txts in evals.items()}
    n = sum(r["n_chars"] for r in res.values())
    bpc = sum(r["bpc"] * r["n_chars"] for r in res.values()) / n
    acc = sum(r["top1_acc"] for r in res.values()) / len(res)
    res["all"] = {"bpc": round(bpc, 4), "char_ppl": round(2**bpc, 4), "top1_acc": round(acc, 4), "n_chars": n}
    return res


def mean_std(xs: Sequence[float]) -> str:
    return f"{np.mean(xs):.4f} ± {np.std(xs):.4f}" if len(xs) > 1 else f"{xs[0]:.4f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--eval", default=str(ROOT / "data/samples/heldout_eval.jsonl"))
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--acc-positions", type=int, default=1500)
    ap.add_argument("--gpt", dest="gpt", action="store_true", default=None)
    ap.add_argument("--no-gpt", dest="gpt", action="store_false")
    ap.add_argument("--gpt-steps", type=int, default=300)
    ap.add_argument("--gpt-dim", type=int, default=64)
    ap.add_argument("--gpt-layers", type=int, default=2)
    ap.add_argument("--gpt-block", type=int, default=64)
    ap.add_argument("--out-dir", default=str(ROOT / "experiments/results"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    base_cfg = load_config(args.config)
    base_cfg["export"] = {"formats": []}
    t0 = time.time()

    # ---- build training sets
    raw_docs = Pipeline(base_cfg, ROOT).ingest()
    train_sets: Dict[str, List[str]] = {"raw": [d.text for d in raw_docs if d.text.strip()]}
    for name, override in CONDITIONS[1:]:
        from sievelab.pipeline.config import deep_merge

        cfg = deep_merge(copy.deepcopy(base_cfg), override)
        res = Pipeline(cfg, ROOT).run(export=False)
        train_sets[name] = [d.text for d in res.documents]
    sizes = {k: (len(v), sum(map(len, v))) for k, v in train_sets.items()}
    budget = sizes["full"][1]
    print("training sets (docs, chars):", sizes, f"\nfixed budget = {budget:,} chars")

    eval_docs = list(read_jsonl(args.eval))
    evals = {lang: [d.text for d in eval_docs if d.gold["lang"] == lang] for lang in ("en", "ur")}
    vocab = set("".join(train_sets["raw"])) | set("".join(t for ts in evals.values() for t in ts))
    contamination = {k: round(ngram_overlap(v, [t for ts in evals.values() for t in ts], n=13), 4) for k, v in train_sets.items()}

    # ---- n-gram LM, fixed budget, multiple seeds
    results: Dict[str, Dict] = {}
    for name, _ in CONDITIONS:
        runs = [eval_ngram(subsample_to_budget(train_sets[name], budget, s), evals, vocab, args.order, args.acc_positions)
                for s in range(args.seeds if name != "full" else 1)]
        results[name] = {
            "train_docs_available": sizes[name][0],
            "train_chars_available": sizes[name][1],
            **{f"{lang}_bpc": [r[lang]["bpc"] for r in runs] for lang in ("en", "ur", "all")},
            **{f"{lang}_acc": [r[lang]["top1_acc"] for r in runs] for lang in ("en", "ur", "all")},
            "all_ppl": [r["all"]["char_ppl"] for r in runs],
        }
        print(f"[ngram] {name:12s} bpc(all)={mean_std(results[name]['all_bpc'])}  acc={mean_std(results[name]['all_acc'])}")
    raw_full = eval_ngram(train_sets["raw"], evals, vocab, args.order, args.acc_positions)
    print(f"[ngram] raw (unbudgeted, {sizes['raw'][1]:,} chars) bpc(all)={raw_full['all']['bpc']:.4f}")

    # ---- tiny GPT, fixed compute
    gpt_results = {}
    try:
        import torch  # noqa: F401

        have_torch = True
    except ImportError:
        have_torch = False
    if (args.gpt is None and have_torch) or args.gpt:
        from sievelab.eval import tiny_gpt

        cv = tiny_gpt.CharVocab(sorted(vocab))
        ev_all = [t for ts in evals.values() for t in ts]
        for name in ("raw", "full"):
            ts = time.time()
            model = tiny_gpt.train(subsample_to_budget(train_sets[name], budget, 0), cv, steps=args.gpt_steps, seed=0,
                                   block_size=args.gpt_block, n_embd=args.gpt_dim, n_layer=args.gpt_layers)
            gpt_results[name] = {lang: tiny_gpt.evaluate(model, txts, cv)
                                 for lang, txts in (("all", ev_all), ("en", evals["en"]), ("ur", evals["ur"]))}
            gpt_results[name]["train_loss_final"] = round(model.final_train_loss, 4)
            print(f"[gpt]   {name:12s} bpc(all)={gpt_results[name]['all']['bpc']:.4f} "
                  f"acc={gpt_results[name]['all']['top1_acc']:.4f} ({time.time() - ts:.0f}s)")

    # ---- report
    def m(name, key):
        return float(np.mean(results[name][key]))

    raw_ppl, full_ppl = m("raw", "all_ppl"), m("full", "all_ppl")
    summary = {
        "budget_chars": budget,
        "ngram_order": args.order,
        "seeds": args.seeds,
        "ngram": results,
        "ngram_raw_unbudgeted": raw_full,
        "gpt": gpt_results,
        "eval_13gram_overlap_with_train": contamination,
        "headline": {
            "ngram_char_ppl_raw": round(raw_ppl, 4),
            "ngram_char_ppl_clean": round(full_ppl, 4),
            "ngram_ppl_reduction_pct": round(100 * (raw_ppl - full_ppl) / raw_ppl, 2),
            "ngram_bpc_reduction_pct": round(100 * (m("raw", "all_bpc") - m("full", "all_bpc")) / m("raw", "all_bpc"), 2),
        },
        "seconds": round(time.time() - t0, 1),
    }
    if gpt_results:
        g_raw, g_full = gpt_results["raw"]["all"]["char_ppl"], gpt_results["full"]["all"]["char_ppl"]
        summary["headline"]["gpt_char_ppl_raw"] = g_raw
        summary["headline"]["gpt_char_ppl_clean"] = g_full
        summary["headline"]["gpt_ppl_reduction_pct"] = round(100 * (g_raw - g_full) / g_raw, 2)

    lines = [
        "# Raw vs. cleaned training data — LM comparison",
        "",
        f"Fixed budget: **{budget:,} characters** per condition (= size of the fully curated set); "
        f"char-level {args.order}-gram Kneser-Ney, shared vocabulary ({len(vocab)} symbols), "
        f"{args.seeds} subsampling seeds (mean ± std). Eval: {sum(len(v) for v in evals.values())} held-out educational "
        "documents (English + Urdu) built from sentences never used in training.",
        "",
        "| Training data | Docs available | BPC (all) ↓ | Char PPL (all) ↓ | BPC en ↓ | BPC ur ↓ | Next-char acc ↑ | Δ PPL vs raw |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, _ in CONDITIONS:
        r = results[name]
        dppl = 100 * (np.mean(r["all_ppl"]) - raw_ppl) / raw_ppl
        lines.append(
            f"| {name} | {r['train_docs_available']:,} | {mean_std(r['all_bpc'])} | {np.mean(r['all_ppl']):.3f} | "
            f"{np.mean(r['en_bpc']):.4f} | {np.mean(r['ur_bpc']):.4f} | {np.mean(r['all_acc']):.4f} | {dppl:+.1f}% |"
        )
    lines += ["", f"Raw, unbudgeted ({sizes['raw'][1]:,} chars, {sizes['raw'][1] / budget:.1f}× more data): "
              f"BPC {raw_full['all']['bpc']:.4f}, char PPL {raw_full['all']['char_ppl']:.3f}."]
    if gpt_results:
        lines += ["", f"**Tiny GPT** ({args.gpt_layers} layers, d={args.gpt_dim}, {args.gpt_steps} steps × 32 × "
                  f"{args.gpt_block} chars, identical compute for both):", "",
                  "| Training data | BPC (all) ↓ | Char PPL ↓ | BPC en ↓ | BPC ur ↓ | Next-char acc ↑ |", "|---|---:|---:|---:|---:|---:|"]
        for name in ("raw", "full"):
            g = gpt_results[name]
            lines.append(f"| {name} | {g['all']['bpc']:.4f} | {g['all']['char_ppl']:.3f} | {g['en']['bpc']:.4f} | "
                         f"{g['ur']['bpc']:.4f} | {g['all']['top1_acc']:.4f} |")
    lines += ["", "13-gram overlap between eval set and training data (contamination check): "
              + ", ".join(f"{k} {v:.2%}" for k, v in contamination.items()), ""]
    md = "\n".join(lines)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lm_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "lm_comparison.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print(json.dumps(summary["headline"], indent=2))


if __name__ == "__main__":
    main()
