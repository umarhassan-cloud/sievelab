import gzip

from dataforge.lang import LanguageIdentifier, WarcWriter, extract_text, iter_documents, iter_records

LID = LanguageIdentifier()


def test_urdu_vs_persian_vs_arabic():
    assert LID.predict("پاکستان ایک خوبصورت ملک ہے اور اس کے پہاڑ بہت اونچے ہیں۔")[0] == "ur"
    assert LID.predict("تهران پایتخت ایران است و جمعیت زیادی دارد.")[0] == "fa"
    assert LID.predict("يقع نهر النيل في شمال شرق قارة أفريقيا وهو أطول نهر في العالم.")[0] == "ar"


def test_pashto_and_sindhi():
    assert LID.predict("پېښور د پښتونخوا یو لرغونی ښار دی او ډېر مشهور دی.")[0] == "ps"
    assert LID.predict("ڪراچي سنڌ جو سڀ کان وڏو شهر آهي ۽ هتي گهڻا ماڻهو رهن ٿا.")[0] == "sd"


def test_roman_urdu_vs_english():
    assert LID.predict("yaar kal ka match bohat zabardast tha, main to hairan reh gaya")[0] == "ur_Latn"
    assert LID.predict("The weather service expects light rain in the northern districts this week.")[0] == "en"


def test_other_scripts_and_short_text():
    assert LID.predict("पानी सभी जीवों के लिए आवश्यक है।")[0] == "hi"
    assert LID.predict("ok")[0] == "und"


def test_html_extraction_drops_chrome():
    html = """<html><head><title>T</title><script>var x=1;</script></head><body>
    <nav><ul><li>Home</li><li>About</li></ul></nav>
    <div class="cookie-banner">We use cookies.</div>
    <article><h1>Title</h1><p>Plants need water and light in order to grow well.</p>
    <p>For example, a bean seed sprouts within a week when kept moist.</p></article>
    <footer>All rights reserved | Privacy | Terms</footer></body></html>"""
    text, title = extract_text(html)
    assert title == "T"
    assert "Plants need water" in text and "bean seed" in text
    for junk in ("var x", "cookies", "Home", "rights reserved"):
        assert junk not in text


def test_warc_roundtrip_plain_and_gzip(tmp_path):
    for name in ("s.warc", "s.warc.gz"):
        p = tmp_path / name
        with WarcWriter(str(p)) as w:
            w.write_response("https://a.example/1", "<html><body><p>Hello from a WARC response record today.</p></body></html>")
            w.write_conversion("https://b.example/2", "Plain text from a WET conversion record.")
        if name.endswith(".gz"):
            with open(p, "rb") as fh:
                assert fh.read(2) == b"\x1f\x8b"
            assert gzip.open(p).read().startswith(b"WARC/1.0")
        recs = list(iter_records(str(p)))
        assert [r.type for r in recs] == ["response", "conversion"]
        docs = list(iter_documents(str(p)))
        assert docs[0].text == "Hello from a WARC response record today."
        assert docs[1].url == "https://b.example/2"
