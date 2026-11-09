# -*- coding: utf-8 -*-
"""Procedurally generated, factually correct educational sentences (English + Urdu).

Hand-written pools alone give only a few hundred unique sentences per
language, which makes any language model saturate after ~100k characters:
"more data" becomes pure repetition. These generators provide thousands of
distinct, true statements (arithmetic, units, geometry, primes, geography,
astronomy, zoology, history of technology). Train and eval splits use
disjoint parameters/rows, so held-out evaluation text is genuinely novel.
"""

from __future__ import annotations

import random
from typing import Callable, List, Tuple

COUNTRIES = [
    ("Pakistan", "Islamabad", "پاکستان", "اسلام آباد"),
    ("China", "Beijing", "چین", "بیجنگ"),
    ("Turkey", "Ankara", "ترکی", "انقرہ"),
    ("Iran", "Tehran", "ایران", "تہران"),
    ("Egypt", "Cairo", "مصر", "قاہرہ"),
    ("Japan", "Tokyo", "جاپان", "ٹوکیو"),
    ("the United Kingdom", "London", "برطانیہ", "لندن"),
    ("France", "Paris", "فرانس", "پیرس"),
    ("Saudi Arabia", "Riyadh", "سعودی عرب", "ریاض"),
    ("Indonesia", "Jakarta", "انڈونیشیا", "جکارتہ"),
    ("Malaysia", "Kuala Lumpur", "ملائیشیا", "کوالالمپور"),
    ("Canada", "Ottawa", "کینیڈا", "اوٹاوا"),
    ("Russia", "Moscow", "روس", "ماسکو"),
    ("Germany", "Berlin", "جرمنی", "برلن"),
    ("Bangladesh", "Dhaka", "بنگلہ دیش", "ڈھاکہ"),
    ("Afghanistan", "Kabul", "افغانستان", "کابل"),
    ("Nepal", "Kathmandu", "نیپال", "کٹھمنڈو"),
    ("Sri Lanka", "Colombo", "سری لنکا", "کولمبو"),
    ("Kenya", "Nairobi", "کینیا", "نیروبی"),
    ("Australia", "Canberra", "آسٹریلیا", "کینبرا"),
]
PLANETS = [
    ("Mercury", "first", "is the smallest planet in the solar system", "عطارد", "پہلا", "نظام شمسی کا سب سے چھوٹا سیارہ ہے"),
    ("Venus", "second", "is the hottest planet because its thick atmosphere traps heat", "زہرہ", "دوسرا", "سب سے گرم سیارہ ہے کیونکہ اس کا گھنا کرۂ ہوائی حرارت کو روک لیتا ہے"),
    ("Earth", "third", "is the only planet known to support life", "زمین", "تیسرا", "واحد سیارہ ہے جس پر زندگی پائی جاتی ہے"),
    ("Mars", "fourth", "looks red because its soil contains iron oxide", "مریخ", "چوتھا", "سرخ دکھائی دیتا ہے کیونکہ اس کی مٹی میں آئرن آکسائیڈ ہے"),
    ("Jupiter", "fifth", "is the largest planet and has a giant storm called the Great Red Spot", "مشتری", "پانچواں", "سب سے بڑا سیارہ ہے اور اس پر ایک بہت بڑا طوفان ہے"),
    ("Saturn", "sixth", "is famous for its bright rings made of ice and rock", "زحل", "چھٹا", "اپنے برف اور چٹانوں سے بنے حلقوں کی وجہ سے مشہور ہے"),
    ("Uranus", "seventh", "rotates on its side with an extreme tilt", "یورینس", "ساتواں", "اپنے محور پر بہت زیادہ جھکا ہوا گھومتا ہے"),
    ("Neptune", "eighth", "has the strongest winds measured in the solar system", "نیپچون", "آٹھواں", "نظام شمسی کی تیز ترین ہواؤں والا سیارہ ہے"),
]
ANIMALS = [
    ("lion", "mammal", "grasslands", "meat", "شیر", "ممالیہ", "گھاس کے میدانوں", "گوشت"),
    ("elephant", "mammal", "forests and savannas", "plants", "ہاتھی", "ممالیہ", "جنگلوں", "پودے"),
    ("eagle", "bird", "mountains", "small animals", "عقاب", "پرندہ", "پہاڑوں", "چھوٹے جانور"),
    ("crocodile", "reptile", "rivers", "fish and other animals", "مگرمچھ", "رینگنے والا جانور", "دریاؤں", "مچھلیاں"),
    ("dolphin", "mammal", "oceans", "fish", "ڈولفن", "ممالیہ", "سمندروں", "مچھلیاں"),
    ("frog", "amphibian", "ponds and wetlands", "insects", "مینڈک", "جل تھلیا", "تالابوں", "کیڑے مکوڑے"),
    ("shark", "fish", "oceans", "fish and seals", "شارک", "مچھلی", "سمندروں", "مچھلیاں"),
    ("camel", "mammal", "deserts", "desert plants", "اونٹ", "ممالیہ", "صحراؤں", "صحرائی پودے"),
    ("owl", "bird", "forests", "mice and insects", "الو", "پرندہ", "جنگلوں", "چوہے"),
    ("snow leopard", "mammal", "the high mountains of Central Asia", "wild goats", "برفانی چیتا", "ممالیہ", "وسطی ایشیا کے اونچے پہاڑوں", "جنگلی بکریاں"),
    ("penguin", "bird", "cold southern seas", "fish", "پینگوئن", "پرندہ", "جنوبی ٹھنڈے سمندروں", "مچھلیاں"),
    ("tortoise", "reptile", "dry grasslands", "plants", "کچھوا", "رینگنے والا جانور", "خشک میدانوں", "پودے"),
    ("honeybee", "insect", "gardens and meadows", "nectar", "شہد کی مکھی", "کیڑا", "باغوں", "پھولوں کا رس"),
    ("salmon", "fish", "rivers and seas", "small fish", "سالمن", "مچھلی", "دریاؤں", "چھوٹی مچھلیاں"),
    ("bat", "mammal", "caves", "insects and fruit", "چمگادڑ", "ممالیہ", "غاروں", "کیڑے اور پھل"),
]
INVENTIONS = [
    ("telephone", "Alexander Graham Bell", 1876, "ٹیلی فون", "الیگزینڈر گراہم بیل"),
    ("light bulb", "Thomas Edison", 1879, "بلب", "تھامس ایڈیسن"),
    ("radio", "Guglielmo Marconi", 1895, "ریڈیو", "مارکونی"),
    ("aeroplane", "the Wright brothers", 1903, "ہوائی جہاز", "رائٹ برادران"),
    ("penicillin", "Alexander Fleming", 1928, "پینسلین", "الیگزینڈر فلیمنگ"),
    ("television", "John Logie Baird", 1926, "ٹیلی وژن", "جان لوگی بیئرڈ"),
    ("World Wide Web", "Tim Berners-Lee", 1989, "ورلڈ وائڈ ویب", "ٹم برنرز لی"),
    ("steam engine", "James Watt", 1769, "بھاپ کا انجن", "جیمز واٹ"),
    ("printing press", "Johannes Gutenberg", 1440, "چھاپہ خانہ", "گوٹن برگ"),
    ("telescope", "Hans Lippershey", 1608, "دوربین", "ہانس لپرشے"),
]


def _is_prime(n: int) -> bool:
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


def _smallest_divisor(n: int) -> int:
    return next(d for d in range(2, n) if n % d == 0)


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


# Each generator: rng -> (key tuple for the train/eval split, english sentence, urdu sentence)
Gen = Callable[[random.Random], Tuple[tuple, str, str]]


def _add(r):
    a, b = r.randint(10, 999), r.randint(10, 999)
    t = r.randrange(3)
    en = [f"The sum of {a} and {b} is {a + b}.", f"If you add {a} to {b}, you get {a + b}.",
          f"Adding {a} and {b} gives a total of {a + b}."][t]
    ur = [f"{a} اور {b} کا مجموعہ {a + b} ہے۔", f"اگر {a} میں {b} جمع کیا جائے تو جواب {a + b} آتا ہے۔",
          f"{a} اور {b} کو جمع کرنے سے کل {a + b} بنتا ہے۔"][t]
    return ("add", a, b), en, ur


def _mul(r):
    a, b = r.randint(2, 99), r.randint(2, 30)
    t = r.randrange(2)
    en = [f"{a} multiplied by {b} equals {a * b}.", f"The product of {a} and {b} is {a * b}."][t]
    ur = [f"{a} کو {b} سے ضرب دینے پر {a * b} حاصل ہوتا ہے۔", f"{a} اور {b} کا حاصل ضرب {a * b} ہے۔"][t]
    return ("mul", a, b), en, ur


def _rect(r):
    l, w = r.randint(2, 60), r.randint(2, 40)
    en = (f"A rectangle that is {l} metres long and {w} metres wide has an area of {l * w} square metres "
          f"and a perimeter of {2 * (l + w)} metres.")
    ur = (f"ایک مستطیل جس کی لمبائی {l} میٹر اور چوڑائی {w} میٹر ہو، اس کا رقبہ {l * w} مربع میٹر "
          f"اور احاطہ {2 * (l + w)} میٹر ہوتا ہے۔")
    return ("rect", l, w), en, ur


def _percent(r):
    p = r.choice([5, 10, 20, 25, 30, 40, 50, 60, 75, 80])
    n = r.randint(1, 60) * 20
    v = _fmt(p * n / 100)
    return ("pct", p, n), f"{p} percent of {n} is {v}.", f"{n} کا {p} فیصد {v} ہے۔"


def _units(r):
    x = r.randint(2, 500)
    t = r.randrange(4)
    en = [f"{x} kilometres is equal to {x * 1000} metres.", f"{x} kilograms equals {x * 1000} grams.",
          f"{x} hours contain {x * 60} minutes.", f"{x} litres is the same as {x * 1000} millilitres."][t]
    ur = [f"{x} کلومیٹر {x * 1000} میٹر کے برابر ہیں۔", f"{x} کلوگرام میں {x * 1000} گرام ہوتے ہیں۔",
          f"{x} گھنٹوں میں {x * 60} منٹ ہوتے ہیں۔", f"{x} لیٹر {x * 1000} ملی لیٹر کے برابر ہیں۔"][t]
    return ("unit", t, x), en, ur


def _temp(r):
    c = r.randint(-8, 40) * 5
    f = _fmt(c * 9 / 5 + 32)
    return (("temp", c), f"A temperature of {c} degrees Celsius is equal to {f} degrees Fahrenheit.",
            f"{c} ڈگری سینٹی گریڈ درجہ حرارت {f} ڈگری فارن ہائیٹ کے برابر ہے۔")


def _prime(r):
    n = r.randint(11, 2000)
    if _is_prime(n):
        return (("prime", n), f"{n} is a prime number because its only divisors are one and {n}.",
                f"{n} ایک مفرد عدد ہے کیونکہ یہ صرف ایک اور {n} پر تقسیم ہوتا ہے۔")
    d = _smallest_divisor(n)
    return (("prime", n), f"{n} is not a prime number, since it can be divided by {d} to give {n // d}.",
            f"{n} مفرد عدد نہیں ہے کیونکہ یہ {d} پر تقسیم ہو کر {n // d} دیتا ہے۔")


def _speed(r):
    v, t = r.randint(3, 24) * 5, r.randint(2, 9)
    return (("speed", v, t), f"A car travelling at {v} kilometres per hour covers {v * t} kilometres in {t} hours.",
            f"{v} کلومیٹر فی گھنٹہ کی رفتار سے چلنے والی گاڑی {t} گھنٹوں میں {v * t} کلومیٹر فاصلہ طے کرتی ہے۔")


def _country(r):
    i, t = r.randrange(len(COUNTRIES)), r.randrange(3)
    c, cap, cu, capu = COUNTRIES[i]
    cc = c[0].upper() + c[1:]
    en = [f"The capital of {c} is {cap}.", f"{cap} is the capital city of {c}.",
          f"{cc} has its capital at {cap}, which is the centre of its national government."][t]
    ur = [f"{cu} کا دارالحکومت {capu} ہے۔", f"{capu} {cu} کا دارالحکومت ہے۔",
          f"{cu} کی قومی حکومت کا مرکز {capu} ہے جو اس کا دارالحکومت ہے۔"][t]
    return ("country", i), en, ur


def _planet(r):
    i = r.randrange(len(PLANETS))
    n, o, f, nu, ou, fu = PLANETS[i]
    return ("planet", i), f"{n} is the {o} planet from the Sun and {f}.", f"{nu} سورج سے {ou} سیارہ ہے اور {fu}۔"


def _animal(r):
    i, t = r.randrange(len(ANIMALS)), r.randrange(2)
    n, c, h, d, nu, cu, hu, du = ANIMALS[i]
    art = "an" if c[0] in "aeiou" else "a"
    en = [f"The {n} is {art} {c} that lives in {h} and feeds mainly on {d}.",
          f"Biologists classify the {n} as {art} {c}; it is usually found in {h}."][t]
    ur = [f"{nu} ایک {cu} ہے جو {hu} میں رہتا ہے اور زیادہ تر {du} کھاتا ہے۔",
          f"ماہرین حیاتیات {nu} کو {cu} قرار دیتے ہیں، اور یہ عموماً {hu} میں پایا جاتا ہے۔"][t]
    return ("animal", i), en, ur


def _invention(r):
    i, t = r.randrange(len(INVENTIONS)), r.randrange(2)
    th, p, y, thu, pu = INVENTIONS[i]
    en = [f"The {th} was invented by {p} in {y}.", f"In {y}, {p} developed the {th}."][t]
    ur = [f"{thu} {y} میں {pu} نے ایجاد کیا۔", f"سن {y} میں {pu} نے {thu} تیار کیا۔"][t]
    return ("invention", i), en, ur


NUMERIC_GENS: List[Gen] = [_add, _mul, _rect, _percent, _units, _temp, _prime, _speed]
TABLE_GENS: List[Gen] = [_country, _planet, _animal, _invention]


def _split_of(key: tuple) -> str:
    if key[0] in ("country", "planet", "animal", "invention"):
        return "eval" if key[1] % 5 == 4 else "train"
    h = 0
    for k in key[1:]:
        h = (h * 1_000_003 + int(k) * 7919) % 2_147_483_647
    return "eval" if h % 5 == 4 else "train"


def procedural_sentences(lang: str, split: str, n: int, seed: int = 0) -> List[str]:
    """``n`` distinct sentences whose parameters belong to ``split``."""
    r = random.Random(f"{seed}-{lang}-{split}")
    gens = NUMERIC_GENS + TABLE_GENS
    out, seen = [], set()
    attempts = 0
    while len(out) < n and attempts < n * 50:
        attempts += 1
        key, en, ur = r.choice(gens)(r)
        if split != "all" and _split_of(key) != split:
            continue
        s = en if lang == "en" else ur
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out
