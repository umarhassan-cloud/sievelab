"""Regenerate everything under data/samples/ deterministically.

    python scripts/make_sample_corpus.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataforge.lang.warc import WarcWriter  # noqa: E402
from dataforge.pipeline.io import write_jsonl  # noqa: E402
from dataforge.synth.generator import CorpusGenerator  # noqa: E402
from dataforge.synth.html import render_page  # noqa: E402

OUT = ROOT / "data" / "samples"

HAND_PAGES = {
    "en_photosynthesis.html": ("en", "How Photosynthesis Powers Life on Earth",
        "Photosynthesis is the process by which green plants, algae and some bacteria convert light energy into chemical energy.\n"
        "Inside the chloroplasts, the pigment chlorophyll absorbs sunlight. This energy is used to split water molecules and release oxygen.\n"
        "The plant then combines carbon dioxide from the air with hydrogen to build glucose, a sugar that stores energy.\n"
        "For example, a single large tree can release enough oxygen in a season to support several people. "
        "Because nearly every food chain begins with plants, photosynthesis is the foundation of almost all life on Earth."),
    "ur_indus_river.html": ("ur", "دریائے سندھ اور پاکستان کی زراعت",
        "دریائے سندھ تبت کے پہاڑوں سے نکلتا ہے اور پاکستان سے گزرتا ہوا بحیرہ عرب میں جا گرتا ہے۔\n"
        "یہ دریا ہزاروں سال سے اس خطے کی زراعت اور تہذیب کی بنیاد رہا ہے۔ وادی سندھ کی قدیم تہذیب اسی دریا کے کنارے پروان چڑھی۔\n"
        "آج بھی پنجاب اور سندھ کے کھیتوں کو نہروں کے ذریعے اسی دریا کا پانی پہنچایا جاتا ہے۔\n"
        "مثال کے طور پر، گندم، کپاس اور چاول کی فصلیں زیادہ تر انہی نہروں پر انحصار کرتی ہیں۔ "
        "اس لیے پانی کا درست استعمال اور ڈیموں کی دیکھ بھال ملک کی غذائی سلامتی کے لیے بہت ضروری ہے۔"),
    "en_spam_deals.html": ("en", "MEGA DEALS",
        "BUY CHEAP WATCHES ONLINE!!! Best price guaranteed, click here FREE shipping!!!\n"
        "watches watches watches discount sale 70% off limited offer, order now\n"
        "BUY CHEAP WATCHES ONLINE!!! Best price guaranteed, click here FREE shipping!!!"),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    corpus = CorpusGenerator(seed=13, split="train").crawl_corpus(n_docs=2200)
    write_jsonl(OUT / "corpus.jsonl", corpus)

    qtrain = CorpusGenerator(seed=7, split="train").quality_train_set(n_docs=1600)
    write_jsonl(OUT / "quality_train.jsonl", qtrain)

    heldout = CorpusGenerator(seed=99, split="eval").heldout_eval_set(n_per_lang=120)
    write_jsonl(OUT / "heldout_eval.jsonl", heldout)

    # Crawl-format inputs. Some WARC pages re-host corpus articles, so dedup has
    # to work across sources (JSONL vs extracted HTML).
    rng = random.Random(5)
    rehosted = rng.sample([d for d in corpus if d.gold.get("type") in ("edu", "info") and not d.gold.get("dup_of")], 40)
    fresh = CorpusGenerator(seed=21, split="train")
    fresh_docs = [fresh.base_doc(rng.choice(["en", "ur"]), rng.choice(["edu", "edu", "info", "casual", "spam"]), prefix="warc")
                  for _ in range(30)]
    with WarcWriter(str(OUT / "cc_sample.warc.gz")) as w:
        for i, d in enumerate(rehosted + fresh_docs):
            title = d.text.split("\n")[0][:60]
            w.write_response(f"https://news{i % 7}.example.com/article/{i}", render_page(title, d.text, d.gold["lang"]))

    wet_gen = CorpusGenerator(seed=33, split="train")
    with WarcWriter(str(OUT / "cc_sample.wet")) as w:
        for i in range(40):
            lang = rng.choice(["en", "ur", "ur", "ar", "fa", "hi", "sw", "ps", "sd"])
            dtype = "other" if lang not in ("en", "ur") else rng.choice(["edu", "info", "casual"])
            d = wet_gen.base_doc(lang, dtype, prefix="wet")
            w.write_conversion(f"https://site{i}.example.net/page", d.text)

    html_dir = OUT / "html"
    html_dir.mkdir(exist_ok=True)
    for name, (lang, title, text) in HAND_PAGES.items():
        (html_dir / name).write_text(render_page(title, text, lang), encoding="utf-8")

    print(f"corpus.jsonl        {len(corpus):5d} docs")
    print(f"quality_train.jsonl {len(qtrain):5d} docs")
    print(f"heldout_eval.jsonl  {len(heldout):5d} docs")
    print(f"cc_sample.warc.gz   {len(rehosted) + len(fresh_docs):5d} response records")
    print("cc_sample.wet          40 conversion records")
    print(f"html/               {len(HAND_PAGES):5d} pages")


if __name__ == "__main__":
    main()
