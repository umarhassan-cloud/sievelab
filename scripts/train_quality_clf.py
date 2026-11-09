"""Train the educational-quality classifier (teacher -> student distillation).

1. Score data/samples/quality_train.jsonl with the heuristic teacher (0-5 rubric).
2. Train the TF-IDF + ridge student on 80% of the teacher labels.
3. Report on the held-out 20%: agreement with the teacher (what the student is
   trained to imitate) and with the generator's gold labels (what we actually
   care about), overall and per language.

    python scripts/train_quality_clf.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataforge.pipeline.io import read_jsonl  # noqa: E402
from dataforge.quality.annotator import EduHeuristicAnnotator  # noqa: E402
from dataforge.quality.classifier import EduQualityClassifier, regression_report  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data/samples/quality_train.jsonl"))
    ap.add_argument("--model-out", default=str(ROOT / "models/edu_quality.pkl"))
    ap.add_argument("--metrics-out", default=str(ROOT / "experiments/results/quality_classifier.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    docs = list(read_jsonl(args.data))
    teacher = EduHeuristicAnnotator()
    texts = [d.text for d in docs]
    t_labels = np.array([teacher.score(t) for t in texts], dtype=float)
    g_labels = np.array([d.gold["quality"] for d in docs], dtype=float)
    langs = np.array([d.gold["lang"] for d in docs])

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(docs))
    n_val = len(docs) // 5
    val, tr = idx[:n_val], idx[n_val:]

    clf = EduQualityClassifier().fit([texts[i] for i in tr], t_labels[tr])
    pred = clf.predict([texts[i] for i in val])

    metrics = {
        "n_train": int(len(tr)),
        "n_val": int(len(val)),
        "teacher_label_distribution": {str(k): int(v) for k, v in zip(*np.unique(t_labels, return_counts=True))},
        "teacher_vs_gold": regression_report(g_labels[val], t_labels[val]),
        "student_vs_teacher": regression_report(t_labels[val], pred),
        "student_vs_gold": regression_report(g_labels[val], pred),
        "student_vs_gold_by_lang": {
            lang: regression_report(g_labels[val][langs[val] == lang], pred[langs[val] == lang])
            for lang in sorted(set(langs[val]))
            if len(set(g_labels[val][langs[val] == lang])) > 1
        },
    }
    # Refit on all data for the shipped model.
    EduQualityClassifier().fit(texts, t_labels).save(args.model_out)

    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"\nmodel -> {args.model_out}\nmetrics -> {args.metrics_out}")


if __name__ == "__main__":
    main()
