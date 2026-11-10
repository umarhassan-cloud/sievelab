"""Sentence-level language-ID benchmark: built-in heuristic vs. langdetect (if installed).

Document-level LID on the synthetic corpus is ~100% accurate (documents are
long and monolingual), so this benchmark uses single sentences / short lines —
the regime where Urdu/Persian/Arabic and Roman-Urdu/English confusions happen —
plus a few code-mixed examples.

    python scripts/eval_langid.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sievelab.lang.langid import LanguageIdentifier  # noqa: E402
from sievelab.synth import content as C  # noqa: E402
from sievelab.synth.generator import edu_sentences  # noqa: E402

CODE_MIXED = [
    ("ur", "آج کی meeting بہت productive تھی اور سب نے اپنے ideas share کیے۔"),
    ("ur", "میرا laptop خراب ہو گیا ہے، کیا آپ اسے repair کر سکتے ہیں؟"),
    ("ur_Latn", "kal office mein bohat kaam tha, meeting bhi late hui"),
    ("ur_Latn", "assignment submit kar di hai, ab exam ki tayari karni hai"),
    ("en", "We had biryani and chai after the cricket match in Lahore."),
    ("en", "The Mughal emperor Shah Jahan built the Taj Mahal in Agra."),
]


def build_benchmark():
    items = []
    for s in edu_sentences("en", "eval") + C.EN_INFO + C.EN_CASUAL:
        items.append(("en", s))
    for s in edu_sentences("ur", "eval") + C.UR_INFO + C.UR_CASUAL:
        items.append(("ur", s))
    for s in C.UR_LATN_CASUAL:
        items.append(("ur_Latn", s))
    for lang, sents in C.OTHER_LANGS.items():
        items += [(lang, s) for s in sents]
    return items + CODE_MIXED


def run(backend: str, items):
    lid = LanguageIdentifier(backend)
    per = defaultdict(Counter)
    conf = Counter()
    t0 = time.time()
    for gold, text in items:
        pred, _ = lid.predict(text)
        per[gold]["n"] += 1
        per[gold]["ok"] += pred == gold
        bucket = "short(<40 chars)" if len(text) < 40 else "long"
        per[bucket]["n"] += 1
        per[bucket]["ok"] += pred == gold
        if pred != gold:
            conf[f"{gold}->{pred}"] += 1
    n = len(items)
    ok = sum(per[g]["ok"] for g in {g for g, _ in items})
    return {
        "accuracy": round(ok / n, 4),
        "per_language": {k: round(v["ok"] / v["n"], 3) for k, v in sorted(per.items())},
        "top_confusions": dict(conf.most_common(8)),
        "ms_per_item": round(1000 * (time.time() - t0) / n, 3),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    items = build_benchmark()
    report = {"n_items": len(items), "languages": dict(Counter(g for g, _ in items)), "heuristic": run("heuristic", items)}
    try:
        report["langdetect"] = run("langdetect", items)
        report["langdetect_note"] = "langdetect has no Pashto, Sindhi or Roman-Urdu classes; those count as errors."
    except ImportError:
        report["langdetect"] = "not installed"
    out = ROOT / "experiments/results/langid_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
