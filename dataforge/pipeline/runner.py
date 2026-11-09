"""End-to-end orchestration: ingest → lang → heuristics → dedup → quality → PII → toxicity → export."""

from __future__ import annotations

import copy
import json
import logging
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dataforge.document import Document
from dataforge.pipeline.config import resolve
from dataforge.pipeline.evaluation import evaluate
from dataforge.pipeline.io import read_inputs, write_jsonl, write_parquet
from dataforge.pipeline.manifest import Manifest, StageRecord, lang_mix
from dataforge.pipeline.stages import (
    DedupStage,
    HeuristicFilterStage,
    LanguageFilterStage,
    PIIStage,
    QualityStage,
    Stage,
    ToxicityStage,
)

log = logging.getLogger("dataforge")


@dataclass
class PipelineResult:
    documents: List[Document]
    manifest: Manifest
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    ingested: List[Document] = field(default_factory=list)


class Pipeline:
    def __init__(self, config: Dict[str, Any], base_dir: Optional[Path] = None):
        self.config = config
        self.base_dir = Path(base_dir or Path.cwd())

    def build_stages(self) -> List[Stage]:
        c = self.config
        stages: List[Stage] = [LanguageFilterStage(c.get("lang", {}))]
        if c.get("heuristics", {}).get("enabled", True):
            stages.append(HeuristicFilterStage(c.get("heuristics", {})))
        d = c.get("dedup", {})
        if d.get("exact", True) or d.get("near", True):
            stages.append(DedupStage(d))
        if c.get("quality", {}).get("enabled", True):
            stages.append(QualityStage(c["quality"], self.base_dir))
        safety = c.get("safety", {})
        if safety.get("pii", {}).get("enabled", True):
            stages.append(PIIStage(safety.get("pii", {})))
        if safety.get("toxicity", {}).get("enabled", True):
            stages.append(ToxicityStage(safety.get("toxicity", {}), self.base_dir))
        return stages

    def ingest(self) -> List[Document]:
        paths = [resolve(p, self.base_dir) for p in self.config.get("inputs", [])]
        ext = self.config.get("extract", {})
        docs = list(read_inputs(paths, extract_backend=ext.get("backend", "builtin")))
        seen_ids: Counter = Counter()
        for d in docs:  # guarantee unique ids across sources
            seen_ids[d.id] += 1
            if seen_ids[d.id] > 1:
                d.id = f"{d.id}#{seen_ids[d.id]}"
        return docs

    def run(self, docs: Optional[List[Document]] = None, export: bool = True) -> PipelineResult:
        t0 = time.time()
        docs = self.ingest() if docs is None else docs
        manifest = Manifest(run_name=self.config.get("run_name", "run"), config=copy.deepcopy(self.config))
        manifest.n_ingested = len(docs)
        manifest.chars_ingested = sum(len(d.text) for d in docs)
        manifest.ingest_by_source = dict(Counter(Path(d.source or "unknown").name for d in docs))
        all_docs = list(docs)

        status: Dict[str, str] = {}
        rejected_records: List[Dict[str, Any]] = []

        min_chars = self.config.get("extract", {}).get("min_chars", 0)
        ts = time.time()
        current = [d for d in docs if len(d.text.strip()) >= min_chars]
        empty = [d for d in docs if len(d.text.strip()) < min_chars]
        for d in empty:
            status[d.id] = "extract"
            rejected_records.append({"id": d.id, "stage": "extract", "reason": "empty_or_short", "url": d.url, "text": d.text})
        manifest.stages.append(StageRecord("extract", len(docs), len(current), manifest.chars_ingested,
                                           sum(len(d.text) for d in current), round(time.time() - ts, 3),
                                           {"empty_or_short": len(empty)} if empty else {}))

        stages = self.build_stages()
        for stage in stages:
            ts = time.time()
            n_in, chars_in = len(current), sum(len(d.text) for d in current)
            out = stage.run(current)
            reasons: Dict[str, int] = defaultdict(int)
            for d, reason in out.rejected:
                status[d.id] = stage.name
                reasons[reason] += 1
                rejected_records.append({"id": d.id, "stage": stage.name, "reason": reason, "url": d.url,
                                         "lang": d.lang, "text": d.text, "meta": _public_meta(d.meta)})
            current = out.kept
            rec = StageRecord(stage.name, n_in, len(current), chars_in, sum(len(d.text) for d in current),
                              round(time.time() - ts, 3), dict(reasons), out.info)
            manifest.stages.append(rec)
            log.info("%-17s %6d -> %6d  (-%.1f%%)", stage.name, n_in, len(current), rec.removed_pct)

        for d in current:
            status[d.id] = "kept"
        manifest.lang_mix_in = lang_mix(all_docs)
        manifest.lang_mix_out = lang_mix(current)

        evaluation = evaluate(
            all_docs,
            status,
            ["extract"] + [s.name for s in stages],
            dedup_threshold=self.config.get("dedup", {}).get("threshold", 0.7),
            quality_threshold=3.0,
            tox_threshold=self.config.get("safety", {}).get("toxicity", {}).get("threshold", 0.5),
        )
        result = PipelineResult(current, manifest, rejected_records, evaluation, all_docs)
        if export:
            self.export(result)
        log.info("done in %.1fs: %d -> %d docs", time.time() - t0, manifest.n_ingested, len(current))
        return result

    def export(self, result: PipelineResult) -> None:
        exp = self.config.get("export", {})
        out_dir = resolve(self.config.get("output_dir", "outputs/run"), self.base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        docs = result.documents
        if exp.get("drop_gold", True):
            docs = [Document(d.id, d.text, d.url, d.source, d.lang, d.lang_score, _public_meta(d.meta)) for d in docs]
        formats = exp.get("formats", ["jsonl"])
        m = result.manifest
        if "jsonl" in formats:
            write_jsonl(out_dir / "clean.jsonl", docs)
            m.outputs["clean_jsonl"] = str(out_dir / "clean.jsonl")
        if "parquet" in formats and write_parquet(out_dir / "clean.parquet", docs):
            m.outputs["clean_parquet"] = str(out_dir / "clean.parquet")
        if exp.get("by_language", True):
            by_lang: Dict[str, List[Document]] = defaultdict(list)
            for d in docs:
                by_lang[d.lang or "und"].append(d)
            for lang, ld in by_lang.items():
                write_jsonl(out_dir / "by_lang" / f"{lang}.jsonl", ld)
        if exp.get("write_rejected", True):
            write_jsonl(out_dir / "rejected.jsonl", result.rejected)
            m.outputs["rejected_jsonl"] = str(out_dir / "rejected.jsonl")
        if result.evaluation:
            (out_dir / "eval.json").write_text(json.dumps(result.evaluation, indent=2, ensure_ascii=False), encoding="utf-8")
            m.outputs["eval_json"] = str(out_dir / "eval.json")
        m.save(out_dir / "manifest.json")
        (out_dir / "manifest.md").write_text(m.to_markdown(), encoding="utf-8")


def _public_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in meta.items() if not k.startswith("_")}
