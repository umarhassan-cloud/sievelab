from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Union

from dataforge.document import Document
from dataforge.lang.extract import extract_text
from dataforge.lang.warc import iter_documents as iter_warc_documents

PathLike = Union[str, Path]
WARC_SUFFIXES = (".warc", ".warc.gz", ".wet", ".wet.gz", ".arc.gz")


def _open_text(path: PathLike, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")
    return open(path, mode, encoding="utf-8", newline="\n" if "w" in mode else None)


def read_jsonl(path: PathLike) -> Iterator[Document]:
    with _open_text(path) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            d.setdefault("id", f"{Path(path).stem}-{i}")
            d.setdefault("source", str(path))
            yield Document.from_dict(d)


def write_jsonl(path: PathLike, records: Iterable[Union[Document, dict]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with _open_text(path, "wt") as fh:
        for r in records:
            d = r.to_dict() if isinstance(r, Document) else r
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_parquet(path: PathLike, docs: Sequence[Document]) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        return False
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": [d.id for d in docs],
            "text": [d.text for d in docs],
            "url": [d.url for d in docs],
            "source": [d.source for d in docs],
            "lang": [d.lang for d in docs],
            "lang_score": [d.lang_score for d in docs],
            "edu_score": [d.meta.get("quality", {}).get("edu_score") for d in docs],
            "meta": [json.dumps(d.meta, ensure_ascii=False) for d in docs],
        }
    )
    pq.write_table(table, str(path), compression="zstd")
    return True


def _iter_path(path: Path, extract_backend: str) -> Iterator[Document]:
    name = path.name.lower()
    if name.endswith((".jsonl", ".jsonl.gz", ".json")):
        yield from read_jsonl(path)
    elif name.endswith(WARC_SUFFIXES):
        yield from iter_warc_documents(str(path), extract_backend=extract_backend)
    elif name.endswith((".html", ".htm")):
        html = path.read_text(encoding="utf-8", errors="replace")
        text, title = extract_text(html, backend=extract_backend)
        yield Document(id=f"html-{path.stem}", text=text, url=path.as_uri(), source=str(path), meta={"title": title})
    elif name.endswith(".txt"):
        yield Document(id=f"txt-{path.stem}", text=path.read_text(encoding="utf-8", errors="replace"), source=str(path))


def read_inputs(paths: Sequence[PathLike], extract_backend: str = "builtin") -> Iterator[Document]:
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    yield from _iter_path(child, extract_backend)
        elif p.exists():
            yield from _iter_path(p, extract_backend)
        else:
            raise FileNotFoundError(f"Input not found: {p}")


def read_documents(paths: Sequence[PathLike]) -> List[Document]:
    return list(read_inputs(paths))
