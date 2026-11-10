"""Deterministic generator for a crawl-like multilingual sample corpus with ground truth.

Every document carries ``meta["_gold"]``:
``lang``, ``type`` (edu / edu_short / info / casual / spam / boilerplate / other),
``quality`` (0-5), ``dup_of`` (id of the original for injected duplicates),
``dup_kind``, ``jaccard`` (true word-5-gram Jaccard to the original),
``pii`` (list of injected entity types) and ``toxic``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sievelab.dedup.minhash import jaccard, shingles
from sievelab.document import Document
from sievelab.synth import content as C
from sievelab.synth.procedural import procedural_sentences

URDU_DIGITS = str.maketrans("0123456789", "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9")


def split_items(items: Sequence, split: str, k: int = 5) -> List:
    """Every k-th item goes to the eval split; the rest to train."""
    if split == "all":
        return list(items)
    want_eval = split == "eval"
    return [x for i, x in enumerate(items) if (i % k == k - 1) == want_eval]


def _render(rows, templates, fmt: Callable) -> List[str]:
    return [t.format(*fmt(r)[0], **fmt(r)[1]) for r in rows for t in templates]


def edu_sentences(lang: str, split: str = "train") -> List[str]:
    if lang == "en":
        out = _render(split_items(C.EN_ELEMENTS, split), C.EN_ELEMENT_T, lambda r: (r, {}))
        out += _render(split_items(C.EN_ORGANS, split), C.EN_ORGAN_T, lambda r: (r, {}))
        out += _render(split_items(C.EN_CONCEPTS, split), C.EN_CONCEPT_T,
                       lambda r: (r, {"0l": r[0][0].lower() + r[0][1:]}))
        out += _render(split_items(C.EN_HISTORY, split), C.EN_HISTORY_T,
                       lambda r: (r, {"0u": r[0][0].upper() + r[0][1:]}))
        return out + split_items(C.EN_EDU_SENTENCES, split)
    if lang == "ur":
        out = _render(split_items(C.UR_ELEMENTS, split), C.UR_ELEMENT_T, lambda r: (r, {}))
        out += _render(split_items(C.UR_ORGANS, split), C.UR_ORGAN_T, lambda r: (r, {}))
        out += _render(split_items(C.UR_CONCEPTS, split), C.UR_CONCEPT_T, lambda r: (r, {}))
        out += _render(split_items(C.UR_HISTORY, split), C.UR_HISTORY_T, lambda r: (r, {}))
        return out + split_items(C.UR_EDU_SENTENCES, split)
    raise ValueError(lang)


@dataclass
class _LangSpec:
    intros: List[str]
    topics: List[str]
    info: List[str]
    info_t: List[str]
    cities: List[str]
    events: List[str]
    days: List[str]
    casual: List[str]
    spam_t: List[str]
    products: List[str]
    boiler: List[str]


SPECS = {
    "en": _LangSpec(C.EN_EDU_INTROS, C.EN_TOPICS, C.EN_INFO, C.EN_INFO_T, C.EN_CITIES, C.EN_EVENTS, C.EN_DAYS,
                    C.EN_CASUAL, C.EN_SPAM_T, C.EN_PRODUCTS, C.BOILERPLATE_EN),
    "ur": _LangSpec(C.UR_EDU_INTROS, C.UR_TOPICS, C.UR_INFO, C.UR_INFO_T, C.UR_CITIES, C.UR_EVENTS, C.UR_DAYS,
                    C.UR_CASUAL, C.UR_SPAM_T, C.UR_PRODUCTS, C.BOILERPLATE_UR),
}


def _paragraphs(rng: random.Random, sents: List[str], sep: str = " ") -> str:
    paras, i = [], 0
    while i < len(sents):
        k = rng.randint(2, 4)
        paras.append(sep.join(sents[i : i + k]))
        i += k
    return "\n".join(paras)


def _url(rng: random.Random) -> str:
    return f"http://www.{rng.choice(['best', 'super', 'mega', 'top'])}-{rng.choice(['deals', 'offers', 'shop'])}{rng.randint(1, 99)}.example/buy"


class CorpusGenerator:
    PROCEDURAL_POOL = {"train": 15000, "eval": 2500, "all": 17500}

    def __init__(self, seed: int = 0, split: str = "train", procedural_frac: float = 0.85):
        self.rng = random.Random(seed)
        self.split = split
        self.procedural_frac = procedural_frac
        self.edu = {lang: edu_sentences(lang, split) for lang in ("en", "ur")}
        self.proc = {lang: procedural_sentences(lang, split, self.PROCEDURAL_POOL[split]) for lang in ("en", "ur")}
        self._n = 0

    def _edu_sentences(self, lang: str, k: int) -> List[str]:
        n_proc = sum(self.rng.random() < self.procedural_frac for _ in range(k))
        base = self.rng.sample(self.edu[lang], min(k - n_proc, len(self.edu[lang])))
        sents = base + self.rng.sample(self.proc[lang], n_proc)
        self.rng.shuffle(sents)
        return sents

    # ---------------------------------------------------------------- doc types
    def edu_doc(self, lang: str) -> Tuple[str, int]:
        r, spec = self.rng, SPECS[lang]
        n = r.randint(6, 12)
        sents = [r.choice(spec.intros).format(t=r.choice(spec.topics))] + self._edu_sentences(lang, n)
        text = _paragraphs(r, sents)
        return text, 5 if n >= 9 else 4

    def edu_short_doc(self, lang: str) -> Tuple[str, int]:
        r, spec = self.rng, SPECS[lang]
        sents = self._edu_sentences(lang, r.randint(3, 4)) + r.sample(spec.info, r.randint(1, 2))
        r.shuffle(sents)
        return _paragraphs(r, sents), 3

    def info_doc(self, lang: str) -> Tuple[str, int]:
        r, spec = self.rng, SPECS[lang]
        sents = r.sample(spec.info, r.randint(3, 6))
        for _ in range(r.randint(1, 3)):
            sents.insert(r.randrange(len(sents) + 1),
                         r.choice(spec.info_t).format(c=r.choice(spec.cities), e=r.choice(spec.events), d=r.choice(spec.days)))
        return _paragraphs(r, sents), 2

    def casual_doc(self, lang: str) -> Tuple[str, int]:
        r = self.rng
        pool = C.UR_LATN_CASUAL if lang == "ur_Latn" else SPECS[lang].casual
        lines = r.sample(pool, min(r.randint(5, 9), len(pool)))
        if lang == "en":
            lines = [ln + (" " + r.choice(C.EN_EMOJI) if r.random() < 0.3 else "") for ln in lines]
        return "\n".join(lines), 1

    def spam_doc(self, lang: str) -> Tuple[str, int]:
        r, spec = self.rng, SPECS[lang]
        p = r.choice(spec.products)
        lines = [r.choice(spec.spam_t).format(p=p, u=_url(r)) for _ in range(r.randint(3, 7))]
        return "\n".join(lines), 0

    def boilerplate_doc(self, lang: str) -> Tuple[str, int]:
        r, spec = self.rng, SPECS[lang]
        lines = r.sample(spec.boiler, min(r.randint(3, 6), len(spec.boiler)))
        if r.random() < 0.5:
            lines.insert(r.randrange(len(lines) + 1), r.choice(spec.info))
        return "\n".join(lines), 0

    def other_doc(self, lang: str) -> Tuple[str, int]:
        r = self.rng
        pool = C.OTHER_LANGS[lang]
        return _paragraphs(r, r.sample(pool, r.randint(4, min(7, len(pool))))), 3

    # ---------------------------------------------------------------- injections
    def _digits(self, n: int) -> str:
        return "".join(str(self.rng.randint(0, 9)) for _ in range(n))

    def pii_value(self, ptype: str, lang: str) -> str:
        r = self.rng
        if ptype == "EMAIL":
            name = r.choice(C.EN_NAMES).lower().replace(" ", r.choice([".", "_", ""]))
            return f"{name}{r.randint(1, 99)}@{r.choice(C.EMAIL_DOMAINS)}"
        if ptype == "PHONE":
            v = r.choice([f"+92 3{self._digits(2)} {self._digits(7)}", f"03{self._digits(2)}-{self._digits(7)}",
                          f"+1 (415) 555-{self._digits(4)}" if lang == "en" else f"03{self._digits(2)}{self._digits(7)}"])
        elif ptype == "CNIC":
            v = f"{r.choice(['35202', '42101', '61101', '17301'])}-{self._digits(7)}-{r.randint(1, 9)}"
        elif ptype == "CREDIT_CARD":
            body = [4] + [r.randint(0, 9) for _ in range(14)]
            total = 0
            for i, d in enumerate(reversed(body)):
                d = d * 2 if i % 2 == 0 else d
                total += d - 9 if d > 9 else d
            digits = "".join(map(str, body + [(10 - total % 10) % 10]))
            v = " ".join(digits[i : i + 4] for i in range(0, 16, 4))
        elif ptype == "IP_ADDRESS":
            v = ".".join(str(r.randint(1, 254)) for _ in range(4))
        else:
            raise ValueError(ptype)
        if lang == "ur" and ptype in ("PHONE", "CNIC") and r.random() < 0.3:
            v = v.translate(URDU_DIGITS)
        return v

    def inject_pii(self, text: str, lang: str) -> Tuple[str, List[str]]:
        templates = C.PII_TEMPLATES_UR if lang == "ur" else C.PII_TEMPLATES_EN
        types = self.rng.sample(sorted(templates), self.rng.randint(1, 2))
        lines = text.split("\n")
        for t in types:
            key = {"EMAIL": "email", "PHONE": "phone", "CNIC": "cnic", "CREDIT_CARD": "card", "IP_ADDRESS": "ip"}[t]
            sentence = templates[t].format(**{key: self.pii_value(t, lang)})
            i = self.rng.randrange(len(lines))
            lines[i] = lines[i] + " " + sentence
        return "\n".join(lines), types

    def inject_toxic(self, text: str, lang: str) -> str:
        pool = C.TOXIC.get(lang, C.TOXIC["en"])
        comment = " ".join(self.rng.sample(pool, self.rng.randint(1, 2)))
        prefix = "تبصرہ:" if lang == "ur" else "Comment:"
        return text + "\n" + (comment if lang == "ur_Latn" else f"{prefix} {comment}")

    def near_duplicate(self, text: str, min_jaccard: float = 0.75) -> Optional[Tuple[str, float]]:
        r = self.rng
        base = shingles(text)
        for attempt in range(6):
            lines = text.split("\n")
            toks_total = sum(len(ln.split()) for ln in lines)
            n_edits = max(1, int(toks_total * r.uniform(0.005, 0.02))) if attempt < 3 else 1
            for _ in range(n_edits):
                li = r.randrange(len(lines))
                toks = lines[li].split()
                if len(toks) < 2:
                    continue
                op = r.random()
                j = r.randrange(len(toks))
                if op < 0.4:
                    toks[j] = r.choice(toks)
                elif op < 0.7:
                    del toks[j]
                else:
                    toks.insert(j, r.choice(toks))
                lines[li] = " ".join(toks)
            if r.random() < 0.5:
                lines.append(r.choice(C.BOILERPLATE_EN[:3]))
            new = "\n".join(lines)
            jac = jaccard(base, shingles(new))
            if new != text and jac >= min_jaccard:
                return new, round(jac, 4)
        return None

    # ---------------------------------------------------------------- corpora
    def _doc(self, prefix: str, text: str, lang: str, dtype: str, quality: int, **gold) -> Document:
        self._n += 1
        return Document(
            id=f"{prefix}-{self._n:05d}",
            text=text,
            url=f"https://{lang.replace('_', '-').lower()}.example.org/{dtype}/{self._n}",
            source="synthetic",
            meta={"_gold": {"lang": lang, "type": dtype, "quality": quality, "dup_of": None, "pii": [], "toxic": False, **gold}},
        )

    def base_doc(self, lang: str, dtype: str, prefix: str = "doc") -> Document:
        if dtype == "other":
            text, q = self.other_doc(lang)
        else:
            text, q = getattr(self, f"{dtype}_doc")(lang)
        return self._doc(prefix, text, lang, dtype, q)

    def crawl_corpus(
        self,
        n_docs: int = 2200,
        exact_dup_rate: float = 0.10,
        near_dup_rate: float = 0.08,
        pii_rate: float = 0.15,
        toxic_rate: float = 0.06,
    ) -> List[Document]:
        r = self.rng
        lang_mix = [("en", 0.38), ("ur", 0.36), ("ur_Latn", 0.04)] + [(l, 0.22 / len(C.OTHER_LANGS)) for l in C.OTHER_LANGS]
        type_mix = [("edu", 0.30), ("edu_short", 0.12), ("info", 0.22), ("casual", 0.16), ("spam", 0.12), ("boilerplate", 0.08)]
        docs: List[Document] = []
        for _ in range(n_docs):
            lang = r.choices([l for l, _ in lang_mix], [w for _, w in lang_mix])[0]
            if lang in ("en", "ur"):
                dtype = r.choices([t for t, _ in type_mix], [w for _, w in type_mix])[0]
            elif lang == "ur_Latn":
                dtype = "casual"
            else:
                dtype = "other"
            d = self.base_doc(lang, dtype)
            g = d.meta["_gold"]
            if lang in ("en", "ur") and dtype in ("edu", "edu_short", "info") and r.random() < pii_rate:
                d.text, g["pii"] = self.inject_pii(d.text, lang)
            if lang in ("en", "ur", "ur_Latn") and dtype in ("edu", "edu_short", "info", "casual") and r.random() < toxic_rate * (2 if dtype == "casual" else 1):
                d.text = self.inject_toxic(d.text, lang)
                g["toxic"] = True
            docs.append(d)

        originals = list(docs)
        for _ in range(int(n_docs * exact_dup_rate)):
            o = r.choice(originals)
            text = o.text if r.random() < 0.5 else "  " + o.text.replace("\n", "\n\n") + "\n"
            docs.append(self._dup(o, text, "exact", 1.0))
        for _ in range(int(n_docs * near_dup_rate)):
            o = r.choice(originals)
            nd = self.near_duplicate(o.text)
            if nd:
                docs.append(self._dup(o, nd[0], "near", nd[1]))
        r.shuffle(docs)
        return docs

    def _dup(self, orig: Document, text: str, kind: str, jac: float) -> Document:
        g = dict(orig.meta["_gold"])
        g.update(dup_of=orig.id, dup_kind=kind, jaccard=jac)
        self._n += 1
        return Document(id=f"doc-{self._n:05d}", text=text, url=f"{orig.url}?mirror={self._n}", source="synthetic",
                        meta={"_gold": g})

    def quality_train_set(self, n_docs: int = 1600) -> List[Document]:
        r = self.rng
        type_mix = [("edu", 0.25), ("edu_short", 0.15), ("info", 0.22), ("casual", 0.16), ("spam", 0.12), ("boilerplate", 0.10)]
        docs = []
        for _ in range(n_docs):
            lang = r.choices(["en", "ur", "ur_Latn"], [0.48, 0.46, 0.06])[0]
            dtype = "casual" if lang == "ur_Latn" else r.choices([t for t, _ in type_mix], [w for _, w in type_mix])[0]
            docs.append(self.base_doc(lang, dtype, prefix="qtrain"))
        return docs

    def heldout_eval_set(self, n_per_lang: int = 120) -> List[Document]:
        docs = []
        for lang in ("en", "ur"):
            for _ in range(n_per_lang):
                docs.append(self.base_doc(lang, "edu", prefix=f"eval-{lang}"))
        return docs
