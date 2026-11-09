from __future__ import annotations

import json
import platform
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from dataforge.document import Document


@dataclass
class StageRecord:
    name: str
    n_in: int
    n_out: int
    chars_in: int
    chars_out: int
    seconds: float
    reasons: Dict[str, int] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_removed(self) -> int:
        return self.n_in - self.n_out

    @property
    def removed_pct(self) -> float:
        return 100.0 * self.n_removed / self.n_in if self.n_in else 0.0


def lang_mix(docs: Sequence[Document]) -> Dict[str, int]:
    return dict(Counter(d.lang or "und" for d in docs).most_common())


@dataclass
class Manifest:
    run_name: str
    config: Dict[str, Any]
    n_ingested: int = 0
    chars_ingested: int = 0
    ingest_by_source: Dict[str, int] = field(default_factory=dict)
    stages: List[StageRecord] = field(default_factory=list)
    lang_mix_in: Dict[str, int] = field(default_factory=dict)
    lang_mix_out: Dict[str, int] = field(default_factory=dict)
    outputs: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    python: str = field(default_factory=platform.python_version)

    @property
    def n_final(self) -> int:
        return self.stages[-1].n_out if self.stages else self.n_ingested

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for s, rec in zip(d["stages"], self.stages):
            s["n_removed"] = rec.n_removed
            s["removed_pct_of_stage_input"] = round(rec.removed_pct, 2)
            s["removed_pct_of_ingested"] = round(100.0 * rec.n_removed / max(self.n_ingested, 1), 2)
            s["cumulative_retention_pct"] = round(100.0 * rec.n_out / max(self.n_ingested, 1), 2)
        d["n_final"] = self.n_final
        d["retention_pct"] = round(100.0 * self.n_final / max(self.n_ingested, 1), 2)
        return d

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    def to_markdown(self) -> str:
        rows = [
            f"# DataForge manifest — `{self.run_name}`",
            "",
            f"Ingested **{self.n_ingested:,}** documents ({self.chars_ingested:,} chars); "
            f"kept **{self.n_final:,}** ({100.0 * self.n_final / max(self.n_ingested, 1):.1f}%).",
            "",
            "| Stage | In | Out | Removed | % of stage input | % of ingested | Cum. retention | Top reasons | Time (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---|---:|",
        ]
        for s in self.stages:
            top = ", ".join(f"{k} ({v})" for k, v in Counter(s.reasons).most_common(3)) or "—"
            rows.append(
                f"| {s.name} | {s.n_in:,} | {s.n_out:,} | {s.n_removed:,} | {s.removed_pct:.1f}% | "
                f"{100.0 * s.n_removed / max(self.n_ingested, 1):.1f}% | "
                f"{100.0 * s.n_out / max(self.n_ingested, 1):.1f}% | {top} | {s.seconds:.2f} |"
            )
        rows += ["", "**Language mix (ingested → final):**", ""]
        rows.append("| Lang | Ingested | Final |")
        rows.append("|---|---:|---:|")
        for lang in sorted(set(self.lang_mix_in) | set(self.lang_mix_out), key=lambda k: -self.lang_mix_in.get(k, 0)):
            rows.append(f"| {lang} | {self.lang_mix_in.get(lang, 0):,} | {self.lang_mix_out.get(lang, 0):,} |")
        return "\n".join(rows) + "\n"
