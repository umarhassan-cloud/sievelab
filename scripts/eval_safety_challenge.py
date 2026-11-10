"""Stress-test PII and toxicity filters on a hand-written, out-of-distribution challenge set.

The in-pipeline evaluation uses injected PII/toxicity drawn from the same
templates the detectors were designed around, so it is optimistic. This set
contains obfuscated PII, toxic paraphrases outside the lexicon, and benign
uses of lexicon words (literal "donkey" in Urdu, "kill" in medicine ...).

    python scripts/eval_safety_challenge.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sievelab.safety import LexiconToxicityScorer, PIIDetector  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    items = [json.loads(l) for l in (ROOT / "data/samples/safety_challenge.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    det, tox = PIIDetector(), LexiconToxicityScorer()

    pii_items = [it for it in items if it["id"].startswith("pii")]
    gold_ents = found_ents = 0
    fp_items, misses = [], []
    for it in pii_items:
        found = {m.type for m in det.detect(it["text"])} - {"PERSON"}
        gold = set(it["pii"])
        gold_ents += len(gold)
        found_ents += len(gold & found)
        if found - gold:
            fp_items.append(it["id"])
        if gold - found:
            misses.append(f"{it['id']} ({it['note']})")
    neg = [it for it in pii_items if not it["pii"]]

    tox_items = [it for it in items if not it["id"].startswith("pii")]
    tp = fp = fn = tn = 0
    errors = []
    for it in tox_items:
        pred = tox.score(it["text"]).score >= 0.5
        if pred and it["toxic"]:
            tp += 1
        elif pred:
            fp += 1
            errors.append(f"FP {it['id']} ({it['note']})")
        elif it["toxic"]:
            fn += 1
            errors.append(f"FN {it['id']} ({it['note']})")
        else:
            tn += 1

    report = {
        "pii": {
            "items": len(pii_items),
            "entity_recall": round(found_ents / gold_ents, 3),
            "false_positive_items": f"{len(fp_items)}/{len(neg)} negatives",
            "fp_ids": fp_items,
            "missed": misses,
        },
        "toxicity_lexicon": {
            "items": len(tox_items),
            "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
            "recall": round(tp / (tp + fn), 3) if tp + fn else 0.0,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "errors": errors,
        },
    }
    out = ROOT / "experiments/results/safety_challenge.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
