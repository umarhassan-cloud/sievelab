"""Main-content text extraction from HTML.

The built-in extractor is a dependency-free approximation of what trafilatura /
resiliparse do at Common Crawl scale:

1. Drop non-content subtrees (``script``, ``style``, ``nav``, ``footer`` ...).
2. Emit block-level elements as separate lines.
3. Drop lines that look like navigation or boilerplate (short, no terminal
   punctuation, pipe-separated menus, cookie banners).

``backend="trafilatura"`` is used when the package is installed and requested.
"""

from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import List, Optional, Tuple

SKIP_TAGS = {
    "script", "style", "noscript", "nav", "footer", "header", "aside", "form",
    "svg", "iframe", "button", "select", "template", "head", "menu",
}
BLOCK_TAGS = {
    "p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6",
    "tr", "td", "th", "section", "article", "main", "blockquote", "pre", "table",
    "dd", "dt", "figcaption", "title", "body",
}
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "source", "wbr"}

BOILERPLATE_PATTERNS = re.compile(
    r"(cookie|privacy policy|terms of (use|service)|all rights reserved|subscribe to|"
    r"sign up|log ?in|register|newsletter|share (this|on)|follow us|"
    r"\u062c\u0645\u0644\u06c1 \u062d\u0642\u0648\u0642|"  # جملہ حقوق (all rights)
    r"\u0634\u06cc\u0626\u0631 \u06a9\u0631\u06cc\u06ba)",  # شیئر کریں (share)
    re.IGNORECASE,
)
TERMINAL_PUNCT = tuple(".!?\u06d4\u061f\u0964:\"')")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # Only the tag that opened a skipped region is counted, so unclosed
        # <li>/<p> inside navigation cannot desynchronise the parser.
        self.skip_tag: Optional[str] = None
        self.skip_count = 0
        self.parts: List[str] = []
        self.title: Optional[str] = None
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag == "title":
            self._in_title = True
            return
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_count += 1
            return
        attr_text = " ".join(f"{k}={v}" for k, v in attrs if k in ("class", "id", "role") and v).lower()
        is_chrome = any(t in attr_text for t in ("cookie", "banner", "navbar", "menu", "sidebar", "footer", "advert"))
        if (tag in SKIP_TAGS or is_chrome) and tag not in VOID_TAGS:
            self.skip_tag, self.skip_count = tag, 1
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
            return
        if self.skip_tag is not None:
            if tag == self.skip_tag:
                self.skip_count -= 1
                if self.skip_count == 0:
                    self.skip_tag = None
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title = (self.title or "") + data.strip()
            return
        if self.skip_tag is not None:
            return
        self.parts.append(data)


def _is_boilerplate_line(line: str, min_words: int) -> bool:
    if "|" in line and line.count("|") >= 2:
        return True
    n_words = len(line.split())
    if BOILERPLATE_PATTERNS.search(line) and n_words < 25:
        return True
    if n_words < min_words and not line.endswith(TERMINAL_PUNCT):
        return True
    return False


def extract_text_builtin(html: str, min_line_words: int = 4) -> Tuple[str, Optional[str]]:
    parser = _Extractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:  # malformed markup: fall back to tag stripping
        return re.sub(r"<[^>]+>", " ", unescape(html)), None
    raw = "".join(parser.parts)
    out_lines = []
    for line in raw.splitlines():
        line = re.sub(r"[ \t\u00a0]+", " ", line).strip()
        if line and not _is_boilerplate_line(line, min_line_words):
            out_lines.append(line)
    return "\n".join(out_lines), parser.title


def extract_text(html: str, backend: str = "builtin") -> Tuple[str, Optional[str]]:
    """Return ``(main_text, title)``."""
    if backend == "trafilatura":
        try:
            import trafilatura  # type: ignore

            text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
            return text, None
        except ImportError:
            pass
    return extract_text_builtin(html)
