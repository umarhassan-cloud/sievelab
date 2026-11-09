"""Minimal, dependency-free WARC / WET reader and writer.

Supports WARC/1.0 and 1.1, plain or gzip (including the per-record multi-member
gzip layout Common Crawl uses). ``response`` records are parsed as HTTP and the
HTML body is run through the text extractor; ``conversion`` records (WET files)
already contain plain text.
"""

from __future__ import annotations

import gzip
import io
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import BinaryIO, Dict, Iterator, Optional

from dataforge.document import Document
from dataforge.lang.extract import extract_text


@dataclass
class WarcRecord:
    headers: Dict[str, str]
    payload: bytes
    http_headers: Dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.headers.get("WARC-Type", "")

    @property
    def uri(self) -> Optional[str]:
        return self.headers.get("WARC-Target-URI")


def _open(path: str) -> BinaryIO:
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        return gzip.open(path, "rb")  # type: ignore[return-value]
    return open(path, "rb")


def _parse_headers(block: bytes) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    for line in block.decode("utf-8", errors="replace").split("\r\n"):
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers


def iter_records(path: str) -> Iterator[WarcRecord]:
    with _open(path) as fh:
        while True:
            line = fh.readline()
            if not line:
                return
            if not line.strip():
                continue
            if not line.startswith(b"WARC/"):
                raise ValueError(f"Malformed WARC: expected version line, got {line[:40]!r}")
            header_lines = []
            while True:
                hl = fh.readline()
                if not hl or hl in (b"\r\n", b"\n"):
                    break
                header_lines.append(hl.rstrip(b"\r\n"))
            headers = _parse_headers(b"\r\n".join(header_lines))
            length = int(headers.get("Content-Length", "0"))
            payload = fh.read(length)
            record = WarcRecord(headers=headers, payload=payload)
            if record.type == "response" and b"\r\n\r\n" in payload:
                head, body = payload.split(b"\r\n\r\n", 1)
                if head.startswith(b"HTTP/"):
                    record.http_headers = _parse_headers(head.split(b"\r\n", 1)[-1])
                    record.payload = body
            yield record


def _charset(http_headers: Dict[str, str]) -> str:
    m = re.search(r"charset=([\w-]+)", http_headers.get("Content-Type", ""), re.I)
    return m.group(1) if m else "utf-8"


def iter_documents(path: str, extract_backend: str = "builtin") -> Iterator[Document]:
    for rec in iter_records(path):
        if rec.type == "response":
            ctype = rec.http_headers.get("Content-Type", "text/html")
            if "html" not in ctype and "text" not in ctype:
                continue
            html = rec.payload.decode(_charset(rec.http_headers), errors="replace")
            text, title = extract_text(html, backend=extract_backend)
            meta = {"warc_type": "response", "title": title}
        elif rec.type == "conversion":
            text = rec.payload.decode("utf-8", errors="replace")
            meta = {"warc_type": "conversion"}
        else:
            continue
        rid = rec.headers.get("WARC-Record-ID", "").strip("<>").replace("urn:uuid:", "")
        yield Document(id=f"warc-{rid}", text=text.strip(), url=rec.uri, source=path, meta=meta)


class WarcWriter:
    """Writes WARC/1.0 records; gzip per-record when ``path`` ends in ``.gz``."""

    def __init__(self, path: str):
        self.path = path
        self.gz = path.endswith(".gz")
        self.fh = open(path, "wb")

    def write(self, warc_type: str, uri: str, payload: bytes, content_type: str, date: Optional[str] = None) -> None:
        date = date or datetime(2024, 1, 1, tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rid = uuid.uuid5(uuid.NAMESPACE_URL, f"{warc_type}:{uri}")
        head = (
            "WARC/1.0\r\n"
            f"WARC-Type: {warc_type}\r\n"
            f"WARC-Target-URI: {uri}\r\n"
            f"WARC-Date: {date}\r\n"
            f"WARC-Record-ID: <urn:uuid:{rid}>\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(payload)}\r\n\r\n"
        ).encode("utf-8")
        record = head + payload + b"\r\n\r\n"
        if self.gz:
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0) as gz:
                gz.write(record)
            self.fh.write(buf.getvalue())
        else:
            self.fh.write(record)

    def write_response(self, uri: str, html: str) -> None:
        body = html.encode("utf-8")
        http = (
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode("utf-8")
        self.write("response", uri, http + body, "application/http; msgtype=response")

    def write_conversion(self, uri: str, text: str) -> None:
        self.write("conversion", uri, text.encode("utf-8"), "text/plain")

    def close(self) -> None:
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
