"""Pipeline stages. Each stage maps ``List[Document] -> StageOutput``.

Stages annotate documents in ``doc.meta[<stage>]`` so every decision is
auditable in the exported data and in ``rejected.jsonl``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dataforge.dedup.dedup import NearDuplicateDeduplicator
from dataforge.dedup.exact import ExactDeduplicator
from dataforge.dedup.minhash import jaccard, shingles
from dataforge.document import Document
from dataforge.lang.langid import LanguageIdentifier
from dataforge.quality.annotator import EduHeuristicAnnotator
from dataforge.quality.classifier import EduQualityClassifier
from dataforge.quality.heuristics import HeuristicConfig, apply_rules
from dataforge.safety.pii import PIIDetector, count_by_type, redact
from dataforge.safety.policy import PIIPolicy, ToxicityPolicy
from dataforge.safety.toxicity import LexiconToxicityScorer, ToxicityClassifier
from dataforge.text_utils import words

log = logging.getLogger("dataforge")

Rejected = List[Tuple[Document, str]]


@dataclass
class StageOutput:
    kept: List[Document]
    rejected: Rejected
    info: Dict[str, Any] = field(default_factory=dict)


class Stage:
    name = "stage"

    def run(self, docs: List[Document]) -> StageOutput:  # pragma: no cover - interface
        raise NotImplementedError


class LanguageFilterStage(Stage):
    name = "lang_filter"

    def __init__(self, cfg: Dict[str, Any]):
        self.lid = LanguageIdentifier(cfg.get("backend", "heuristic"), cfg.get("fasttext_model"))
        self.keep = set(cfg.get("keep") or [])
        ms = cfg.get("min_score", 0.5)
        self.min_score = ms if isinstance(ms, dict) else {"default": ms}

    def run(self, docs: List[Document]) -> StageOutput:
        kept, rejected = [], []
        for d in docs:
            d.lang, d.lang_score = self.lid.predict(d.text)
            threshold = self.min_score.get(d.lang, self.min_score.get("default", 0.5))
            if self.keep and d.lang not in self.keep:
                rejected.append((d, f"lang_not_kept:{d.lang}"))
            elif d.lang_score < threshold:
                rejected.append((d, "low_lang_confidence"))
            else:
                kept.append(d)
        return StageOutput(kept, rejected)


class HeuristicFilterStage(Stage):
    name = "heuristic_filter"

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = HeuristicConfig.from_dict(cfg)

    def run(self, docs: List[Document]) -> StageOutput:
        kept, rejected = [], []
        for d in docs:
            ok, reasons, stats = apply_rules(d.text, d.lang, self.cfg)
            d.meta["heuristics"] = {"failed": reasons, "n_words": stats["n_words"]}
            if ok:
                kept.append(d)
            else:
                rejected.append((d, reasons[0]))
        return StageOutput(kept, rejected)


class DedupStage(Stage):
    name = "dedup"

    def __init__(self, cfg: Dict[str, Any]):
        self.exact = cfg.get("exact", True)
        self.near = cfg.get("near", True)
        self.threshold = cfg.get("threshold", 0.7)
        self.ngram = cfg.get("ngram", 5)
        self.char_level = cfg.get("char_level", False)
        self.near_dedup = NearDuplicateDeduplicator(
            threshold=self.threshold,
            num_perm=cfg.get("num_perm", 128),
            ngram=self.ngram,
            char_level=self.char_level,
            mode=cfg.get("mode", "streaming"),
            verify=cfg.get("verify", "minhash"),
            workers=cfg.get("workers", 1),
            parallel_backend=cfg.get("parallel_backend", "multiprocessing"),
            backend=cfg.get("backend", "python"),
        )

    def run(self, docs: List[Document]) -> StageOutput:
        rejected: Rejected = []
        info: Dict[str, Any] = {}
        if self.exact:
            kept_idx, removed = ExactDeduplicator().run([d.id for d in docs], [d.text for d in docs])
            for i, orig in removed.items():
                docs[i].meta["dedup"] = {"kind": "exact", "dup_of": orig, "similarity": 1.0, "exact_jaccard": 1.0}
                rejected.append((docs[i], "exact_dup"))
            info["exact_removed"] = len(removed)
            docs = [docs[i] for i in kept_idx]
        if self.near and docs:
            res = self.near_dedup.run([d.text for d in docs])
            for i, rep in res.removed.items():
                a = shingles(docs[i].text, self.ngram, self.char_level)
                b = shingles(docs[rep].text, self.ngram, self.char_level)
                docs[i].meta["dedup"] = {
                    "kind": "near",
                    "dup_of": docs[rep].id,
                    "similarity": round(res.similarity.get(i, 0.0), 4),
                    "exact_jaccard": round(jaccard(a, b), 4),
                }
                rejected.append((docs[i], "near_dup"))
            info.update(
                near_removed=len(res.removed),
                lsh_bands=res.bands,
                lsh_rows=res.rows,
                candidate_pairs=res.n_candidate_pairs,
                verified_pairs=res.n_verified_pairs,
                clusters_with_dups=res.n_clusters_with_dups,
            )
            docs = [docs[i] for i in res.kept]
        return StageOutput(docs, rejected, info)


def load_or_train_quality_model(model_path: Path, train_path: Optional[Path]) -> EduQualityClassifier:
    if model_path.exists():
        return EduQualityClassifier.load(str(model_path))
    if train_path is None or not train_path.exists():
        raise FileNotFoundError(f"No quality model at {model_path} and no training data at {train_path}")
    from dataforge.pipeline.io import read_jsonl

    log.info("Quality model not found; training on %s", train_path)
    docs = list(read_jsonl(train_path))
    teacher = EduHeuristicAnnotator()
    clf = EduQualityClassifier().fit([d.text for d in docs], [teacher.score(d.text) for d in docs])
    clf.save(str(model_path))
    return clf


class QualityStage(Stage):
    name = "quality_filter"

    def __init__(self, cfg: Dict[str, Any], base_dir: Path):
        from dataforge.pipeline.config import resolve

        train = cfg.get("train_data")
        self.clf = load_or_train_quality_model(
            resolve(cfg.get("model_path", "models/edu_quality.pkl"), base_dir), resolve(train, base_dir) if train else None
        )
        self.thresholds = dict(cfg.get("thresholds") or {"default": 2.5})

    def run(self, docs: List[Document]) -> StageOutput:
        if not docs:
            return StageOutput([], [])
        scores = self.clf.predict([d.text for d in docs])
        kept, rejected = [], []
        for d, s in zip(docs, scores):
            d.meta["quality"] = {"edu_score": round(float(s), 3), "edu_int": int(round(float(s)))}
            thr = self.thresholds.get(d.lang, self.thresholds.get("default", 2.5))
            if s >= thr:
                kept.append(d)
            else:
                rejected.append((d, "low_edu_score"))
        return StageOutput(kept, rejected, {"mean_edu_score_in": round(float(scores.mean()), 3)})


class PIIStage(Stage):
    name = "pii"

    def __init__(self, cfg: Dict[str, Any]):
        self.policy = PIIPolicy.from_dict(cfg)
        self.detector = PIIDetector(enable_names=self.policy.enable_names)

    def run(self, docs: List[Document]) -> StageOutput:
        kept, rejected = [], []
        totals: Dict[str, int] = {}
        n_redacted_docs = 0
        for d in docs:
            matches = [m for m in self.detector.detect(d.text) if self.policy.actions.get(m.type, "redact") != "keep"]
            counts = count_by_type(matches)
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            d.meta["pii"] = {"counts": counts}
            drop_types = [m.type for m in matches if self.policy.actions.get(m.type) == "drop"]
            n_words = max(len(words(d.text)), 1)
            hard = [m for m in matches if m.type != "PERSON"]
            density = 100.0 * len(hard) / n_words
            if drop_types:
                rejected.append((d, f"pii_drop:{drop_types[0]}"))
                continue
            if self.policy.max_pii_per_100_words is not None and density > self.policy.max_pii_per_100_words:
                rejected.append((d, "pii_density"))
                continue
            redact_types = {t for t, a in self.policy.actions.items() if a == "redact"}
            to_redact = [m for m in matches if m.type in redact_types]
            if to_redact:
                d.text = redact(d.text, to_redact)
                d.meta["pii"]["redacted"] = len(to_redact)
                n_redacted_docs += 1
            kept.append(d)
        return StageOutput(kept, rejected, {"entities_found": totals, "docs_redacted": n_redacted_docs})


class ToxicityStage(Stage):
    name = "toxicity"

    def __init__(self, cfg: Dict[str, Any], base_dir: Path):
        from dataforge.pipeline.config import resolve

        self.policy = ToxicityPolicy.from_dict(cfg)
        self.lexicon = (
            LexiconToxicityScorer.from_json(str(resolve(self.policy.lexicon_path, base_dir)))
            if self.policy.lexicon_path
            else LexiconToxicityScorer()
        )
        self.clf = (
            ToxicityClassifier.load(str(resolve(self.policy.classifier_path, base_dir)))
            if self.policy.classifier_path
            else None
        )

    def run(self, docs: List[Document]) -> StageOutput:
        kept, rejected = [], []
        clf_scores = self.clf.predict_proba([d.text for d in docs]) if (self.clf and docs) else None
        n_redacted = 0
        for i, d in enumerate(docs):
            res = self.lexicon.score(d.text, self.policy.threshold)
            score = res.score if clf_scores is None else max(res.score, float(clf_scores[i]))
            d.meta["toxicity"] = {"score": round(score, 3), "hits": sorted({h for h, _ in res.hits})}
            toxic = score >= self.policy.threshold
            if not toxic or self.policy.action in ("flag", "keep"):
                kept.append(d)
            elif self.policy.action == "redact_sentences":
                d.text = self.lexicon.redact_sentences(d.text, self.policy.threshold)
                d.meta["toxicity"]["redacted_sentences"] = len(res.toxic_sentences)
                n_redacted += 1
                if d.text.strip():
                    kept.append(d)
                else:
                    rejected.append((d, "toxic"))
            else:
                rejected.append((d, "toxic"))
        return StageOutput(kept, rejected, {"docs_sentence_redacted": n_redacted})
