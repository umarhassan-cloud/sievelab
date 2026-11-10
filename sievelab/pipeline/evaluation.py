"""Measure every filter against ground truth injected into the synthetic corpus.

Synthetic documents carry ``meta["_gold"]`` with the true language, document
type, educational quality, duplicate group, injected PII types and toxicity.
Filters never read it; this module joins decisions with labels afterwards, so
the README can report precision/recall per stage instead of just "X% removed".
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score

from sievelab.document import Document


def _prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4), "tp": tp, "fp": fp, "fn": fn}


def evaluate(
    docs: Sequence[Document],
    status: Dict[str, str],
    stage_order: List[str],
    dedup_threshold: float,
    quality_threshold: float = 3.0,
    tox_threshold: float = 0.5,
) -> Dict[str, Any]:
    """``status[doc_id]`` is the stage that removed the document, or ``"kept"``."""
    gold_docs = [d for d in docs if d.gold]
    if not gold_docs:
        return {}
    rank = {name: i for i, name in enumerate(stage_order)}
    rank["kept"] = len(stage_order)

    def reached(d: Document, stage: str) -> bool:
        return rank.get(status.get(d.id, "kept"), 0) >= rank.get(stage, 0)

    out: Dict[str, Any] = {"n_gold_docs": len(gold_docs)}

    # Language ID
    if "lang_filter" in rank:
        per_lang: Dict[str, Counter] = defaultdict(Counter)
        correct = 0
        for d in gold_docs:
            g = d.gold.get("lang")
            ok = d.lang == g
            correct += ok
            per_lang[g]["n"] += 1
            per_lang[g]["correct"] += ok
        out["langid"] = {
            "accuracy": round(correct / len(gold_docs), 4),
            "per_language_accuracy": {k: round(v["correct"] / v["n"], 4) for k, v in sorted(per_lang.items())},
            "confusions": dict(Counter(f"{d.gold.get('lang')}->{d.lang}" for d in gold_docs if d.lang != d.gold.get("lang")).most_common(10)),
        }

    # Dedup
    if "dedup" in rank:
        at = [d for d in gold_docs if reached(d, "dedup")]
        groups: Dict[str, List[Document]] = defaultdict(list)
        for d in at:
            groups[d.gold.get("dup_of") or d.id].append(d)
        expected = sum(len(g) - 1 for g in groups.values())
        removed = [d for d in at if status.get(d.id) == "dedup"]
        caught = sum(min(sum(status.get(m.id) == "dedup" for m in g), len(g) - 1) for g in groups.values())
        in_dup_group = sum(1 for d in removed if len(groups[d.gold.get("dup_of") or d.id]) > 1)
        verified = sum(1 for d in removed if d.meta.get("dedup", {}).get("exact_jaccard", 0) >= dedup_threshold)
        out["dedup"] = {
            "docs_reaching_stage": len(at),
            "injected_duplicates_reaching_stage": expected,
            "removed": len(removed),
            "recall_on_injected": round(caught / expected, 4) if expected else None,
            "precision_vs_injected": round(in_dup_group / len(removed), 4) if removed else None,
            "precision_exact_jaccard": round(verified / len(removed), 4) if removed else None,
        }

    # Quality
    if "quality_filter" in rank:
        at = [d for d in gold_docs if reached(d, "quality_filter") and "quality" in d.meta and "quality" in d.gold]
        if at:
            g = np.array([d.gold["quality"] for d in at])
            p = np.array([d.meta["quality"]["edu_int"] for d in at])
            keep_pred = np.array([status.get(d.id) != "quality_filter" for d in at])
            keep_gold = g >= quality_threshold
            tp = int(np.sum(keep_pred & keep_gold))
            fp = int(np.sum(keep_pred & ~keep_gold))
            fn = int(np.sum(~keep_pred & keep_gold))
            out["quality"] = {
                "docs_scored": len(at),
                "qwk_vs_gold": round(float(cohen_kappa_score(g, p, weights="quadratic")), 4),
                "keep_decision_vs_gold": _prf(tp, fp, fn),
                "per_language_qwk": {
                    lang: round(float(cohen_kappa_score(
                        [d.gold["quality"] for d in at if d.lang == lang],
                        [d.meta["quality"]["edu_int"] for d in at if d.lang == lang], weights="quadratic")), 4)
                    for lang in sorted({d.lang for d in at})
                    if len({d.gold["quality"] for d in at if d.lang == lang}) > 1
                },
            }

    # PII
    if "pii" in rank:
        at = [d for d in gold_docs if reached(d, "pii") and "pii" in d.meta]
        type_tp: Counter = Counter()
        type_n: Counter = Counter()
        fp_docs = 0
        clean_docs = 0
        for d in at:
            gold_types = set(d.gold.get("pii") or [])
            found = {t for t, c in d.meta["pii"]["counts"].items() if c > 0}
            for t in gold_types:
                type_n[t] += 1
                type_tp[t] += t in found
            if not gold_types:
                clean_docs += 1
                fp_docs += bool(found - {"PERSON"})
        out["pii"] = {
            "docs_checked": len(at),
            "recall_by_type": {t: round(type_tp[t] / type_n[t], 4) for t in sorted(type_n)},
            "recall_overall": round(sum(type_tp.values()) / max(sum(type_n.values()), 1), 4),
            "false_positive_doc_rate": round(fp_docs / max(clean_docs, 1), 4),
        }

    # Toxicity
    if "toxicity" in rank:
        at = [d for d in gold_docs if reached(d, "toxicity") and "toxicity" in d.meta]
        pred = [d.meta["toxicity"]["score"] >= tox_threshold for d in at]
        gold = [bool(d.gold.get("toxic")) for d in at]
        tp = sum(p and g for p, g in zip(pred, gold))
        fp = sum(p and not g for p, g in zip(pred, gold))
        fn = sum(g and not p for p, g in zip(pred, gold))
        out["toxicity"] = {"docs_checked": len(at), "toxic_in_gold": sum(gold), **_prf(tp, fp, fn)}

    # Composition of the final output by gold document type
    kept = [d for d in gold_docs if status.get(d.id, "kept") == "kept"]
    out["final_by_gold_type"] = dict(Counter(d.gold.get("type") for d in kept).most_common())
    out["ingested_by_gold_type"] = dict(Counter(d.gold.get("type") for d in gold_docs).most_common())
    return out
