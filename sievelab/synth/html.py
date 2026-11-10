from __future__ import annotations

from html import escape

NAV = {
    "en": ["Home", "News", "Science", "Education", "About", "Contact"],
    "ur": ["صفحہ اول", "خبریں", "سائنس", "تعلیم", "ہمارے بارے میں", "رابطہ"],
}
COOKIE = {"en": "We use cookies to improve your experience.", "ur": "ہم آپ کے تجربے کو بہتر بنانے کے لیے کوکیز استعمال کرتے ہیں۔"}
FOOTER = {"en": "\u00a9 2024 Example Media. All rights reserved.", "ur": "\u00a9 2024 مثال میڈیا۔ جملہ حقوق محفوظ ہیں۔"}


def render_page(title: str, text: str, lang: str) -> str:
    """Wrap article text in realistic page chrome (nav, cookie banner, sidebar, footer, scripts)."""
    ui = "ur" if lang == "ur" else "en"
    direction = "rtl" if lang in ("ur", "ar", "fa", "ps", "sd") else "ltr"
    nav = "".join(f'<li><a href="/{i}">{escape(x)}</a></li>' for i, x in enumerate(NAV[ui]))
    paras = "\n".join(f"<p>{escape(p)}</p>" for p in text.split("\n") if p.strip())
    return f"""<!DOCTYPE html>
<html lang="{lang}" dir="{direction}">
<head><meta charset="utf-8"><title>{escape(title)}</title>
<style>body {{ font-family: sans-serif; }}</style>
<script>window.dataLayer = window.dataLayer || []; function track(e) {{ dataLayer.push(e); }}</script>
</head>
<body>
<header><nav class="navbar"><ul>{nav}</ul></nav></header>
<div class="cookie-banner">{escape(COOKIE[ui])} <button>OK</button></div>
<main><article>
<h1>{escape(title)}</h1>
{paras}
</article></main>
<aside class="sidebar"><h3>Related</h3><ul><li><a href="/r1">{escape(NAV[ui][2])}</a></li><li><a href="/r2">{escape(NAV[ui][3])}</a></li></ul></aside>
<footer><p>{escape(FOOTER[ui])}</p><p>{" | ".join(NAV[ui])}</p></footer>
<script>track({{"page": "article"}});</script>
</body>
</html>
"""
