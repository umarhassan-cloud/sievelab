"""Why deduplicate? Repetition at a fixed character budget hurts held-out perplexity.

Train a character 5-gram KN model on a fixed budget of educational text built
from ``U`` unique documents each repeated ``budget / U`` times, and evaluate on
held-out educational text. Fewer unique documents == more duplication.

    python experiments/repetition_ablation.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sievelab.eval.ngram_lm import CharKneserNeyLM  # noqa: E402
from sievelab.pipeline.io import read_jsonl  # noqa: E402


def main() -> None:
    docs = [d.text for d in read_jsonl(ROOT / "data/samples/quality_train.jsonl") if d.gold["type"] == "edu"]
    random.Random(0).shuffle(docs)
    evals = [d.text for d in read_jsonl(ROOT / "data/samples/heldout_eval.jsonl")]
    budget = sum(map(len, docs[:400]))
    rows = []
    for n_unique in (400, 200, 100, 50, 25):
        uniq = docs[:n_unique]
        train, used = [], 0
        while used < budget:
            for t in uniq:
                if used >= budget:
                    break
                t = t[: budget - used]
                train.append(t)
                used += len(t)
        lm = CharKneserNeyLM(order=5).fit(train)
        r = lm.evaluate(evals, acc_positions=0)
        rows.append({"unique_docs": n_unique, "repeats": round(budget / sum(map(len, uniq)), 2), "bpc": r["bpc"], "char_ppl": r["char_ppl"]})
        print(rows[-1], flush=True)
    base = rows[0]["char_ppl"]
    md = ["| Unique docs | Avg. repeats | BPC ↓ | Char PPL ↓ | Δ PPL vs. no repetition |", "|---:|---:|---:|---:|---:|"]
    md += [f"| {r['unique_docs']} | {r['repeats']}× | {r['bpc']:.4f} | {r['char_ppl']:.3f} | {100 * (r['char_ppl'] - base) / base:+.1f}% |" for r in rows]
    out = ROOT / "experiments/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "repetition_ablation.json").write_text(json.dumps({"budget_chars": budget, "rows": rows}, indent=2), encoding="utf-8")
    (out / "repetition_ablation.md").write_text(f"Fixed budget {budget:,} chars, char 5-gram KN.\n\n" + "\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
