# -*- coding: utf-8 -*-
"""Seed content for the synthetic multilingual web corpus.

Educational content is built from (a) hand-written sentences and (b) small
fact tables rendered through several templates. Rows/sentences are split
deterministically into a *train* pool (used for the crawl-like corpus and the
quality-classifier training set) and a disjoint *eval* pool (used only for the
held-out LM evaluation set), so the raw-vs-clean LM experiment never evaluates
on sentences it trained on.
"""

from __future__ import annotations

from typing import Dict, List

# --------------------------------------------------------------------------- English
EN_ELEMENTS = [
    ("Hydrogen", "H", 1, "is the lightest and most abundant element in the universe"),
    ("Helium", "He", 2, "is a noble gas that rarely reacts with other elements"),
    ("Carbon", "C", 6, "forms the backbone of all known organic molecules"),
    ("Nitrogen", "N", 7, "makes up about seventy-eight percent of the atmosphere of Earth"),
    ("Oxygen", "O", 8, "is essential for cellular respiration in most living organisms"),
    ("Sodium", "Na", 11, "is a soft metal that reacts strongly with water"),
    ("Silicon", "Si", 14, "is the key material used to make computer chips"),
    ("Calcium", "Ca", 20, "is an important mineral for healthy bones and teeth"),
    ("Iron", "Fe", 26, "is the main component of steel and is found in the hemoglobin of our blood"),
    ("Copper", "Cu", 29, "is widely used in electrical wiring because it conducts electricity well"),
    ("Gold", "Au", 79, "is a dense metal that resists corrosion and has been used as money for centuries"),
    ("Uranium", "U", 92, "is a heavy radioactive element used as fuel in nuclear power plants"),
    ("Neon", "Ne", 10, "glows with a bright orange-red light when electricity passes through it"),
    ("Aluminium", "Al", 13, "is a light metal used to build aircraft and drink cans"),
    ("Chlorine", "Cl", 17, "is used to disinfect drinking water and swimming pools"),
]
EN_ELEMENT_T = [
    "{0}, with the chemical symbol {1} and atomic number {2}, {3}.",
    "The element {0} ({1}) {3}.",
    "Element number {2} in the periodic table is {0}, which {3}.",
]

EN_ORGANS = [
    ("heart", "pumps blood through the circulatory system"),
    ("lungs", "exchange oxygen and carbon dioxide with the blood"),
    ("liver", "filters toxins from the blood and produces bile for digestion"),
    ("kidneys", "remove waste from the blood and produce urine"),
    ("brain", "controls thought, memory and the coordination of movement"),
    ("stomach", "breaks down food using acid and digestive enzymes"),
    ("skin", "protects the body from infection and helps regulate temperature"),
    ("small intestine", "absorbs most of the nutrients from digested food"),
    ("pancreas", "produces insulin, a hormone that controls the level of sugar in the blood"),
    ("eyes", "focus light onto the retina so that the brain can form images"),
]
EN_ORGAN_T = [
    "The {0} {1}.",
    "In the human body, the {0} {1}.",
    "Doctors explain that the {0} {1}.",
]

EN_CONCEPTS = [
    ("A prime number", "a whole number greater than one that has no divisors other than one and itself"),
    ("An algorithm", "a finite sequence of well-defined steps used to solve a problem"),
    ("Photosynthesis", "the process by which green plants use sunlight to turn water and carbon dioxide into glucose and oxygen"),
    ("Inflation", "a general increase in prices that reduces the purchasing power of money"),
    ("Gravity", "the force that attracts objects with mass toward each other"),
    ("Democracy", "a system of government in which citizens choose their leaders through elections"),
    ("Erosion", "the gradual wearing away of rock and soil by water, wind or ice"),
    ("An ecosystem", "a community of living organisms interacting with their physical environment"),
    ("Evaporation", "the process in which a liquid changes into a gas"),
    ("A fraction", "a number that represents a part of a whole"),
    ("Supply and demand", "an economic model that explains how prices are determined in a market"),
    ("Machine learning", "a field of computer science in which systems learn patterns from data"),
    ("The water cycle", "the continuous movement of water between the land, the oceans and the atmosphere"),
    ("A hypothesis", "a testable explanation that scientists propose before running an experiment"),
    ("Plate tectonics", "the theory that the outer shell of Earth is divided into large plates that slowly move"),
    ("Osmosis", "the movement of water across a membrane from a dilute solution to a concentrated one"),
    ("A catalyst", "a substance that speeds up a chemical reaction without being used up"),
    ("Kinetic energy", "the energy an object has because of its motion"),
    ("A noun", "a word that names a person, place, thing or idea"),
    ("The equator", "an imaginary line that divides Earth into the northern and southern hemispheres"),
]
EN_CONCEPT_T = [
    "{0} is {1}.",
    "In simple terms, {0l} is {1}.",
    "Students should remember that {0l} is {1}.",
    "By definition, {0l} is {1}.",
]

EN_HISTORY = [
    ("the Indus Valley Civilization", "flourished around 2500 BCE in what is now Pakistan and northwest India"),
    ("the printing press", "was developed by Johannes Gutenberg around 1440 and made books far cheaper"),
    ("the Industrial Revolution", "began in Britain in the eighteenth century and transformed manufacturing"),
    ("the Mughal Empire", "ruled much of the Indian subcontinent from the sixteenth to the nineteenth century"),
    ("the Silk Road", "connected China with the Middle East and Europe through a network of trade routes"),
    ("the first powered flight", "was achieved by the Wright brothers in 1903"),
    ("the Badshahi Mosque in Lahore", "was completed in 1673 during the reign of Aurangzeb"),
    ("Mohenjo-daro", "was one of the largest cities of the ancient Indus Valley"),
    ("the House of Wisdom in Baghdad", "was a major centre of translation and scholarship in the ninth century"),
    ("the Renaissance", "was a period of renewed interest in art and science that began in Italy"),
]
EN_HISTORY_T = [
    "Historians note that {0} {1}.",
    "{0u} {1}.",
    "It is worth knowing that {0} {1}.",
]

EN_EDU_SENTENCES = [
    "Science aims to understand the rules of nature through observation and experiment.",
    "For example, when water is heated to one hundred degrees Celsius at sea level, it boils and turns into steam.",
    "This is why teachers encourage students to ask questions and to test their ideas with simple experiments.",
    "In mathematics, an equation shows that two quantities are equal.",
    "Earth completes one orbit around the Sun in about three hundred and sixty-five days.",
    "The Moon does not produce its own light; it reflects the light of the Sun.",
    "Plants need water, light and nutrients from the soil in order to grow.",
    "Because air contains oxygen, a fire cannot keep burning without it.",
    "Scientists record the results of an experiment carefully so that others can repeat it.",
    "A computer stores information as long sequences of zeros and ones.",
    "The Indus is the longest river in Pakistan and is vital for agriculture.",
    "A healthy diet includes fruit, vegetables, whole grains and a source of protein.",
    "Sound travels through the air as a wave of pressure.",
    "Light travels at roughly three hundred thousand kilometres per second.",
    "Studying history helps us understand how societies change over time.",
    "Geography is the study of the surface of Earth, its climates and its people.",
    "In other words, every action has an equal and opposite reaction.",
    "The laws of motion described by Newton form the foundation of classical physics.",
    "The cell is the basic unit of life, and all living things are made of cells.",
    "Vaccines train the immune system to recognise and fight specific diseases.",
    "Similarly, the roots of trees hold soil in place and reduce erosion.",
    "As a result, regular practice is the most reliable way to master a new skill.",
    "A magnet has two poles, called the north pole and the south pole.",
    "Earthquakes occur when the plates of the crust of Earth move suddenly.",
    "To find the area of a rectangle, multiply its length by its width.",
    "The heart of an adult beats roughly seventy times every minute at rest.",
    "Clouds form when water vapour cools and condenses into tiny droplets.",
    "A sentence usually contains a subject and a verb.",
    "The scientific method begins with a question and ends with a conclusion based on evidence.",
    "Fossils provide evidence of plants and animals that lived millions of years ago.",
    "Therefore, the experiment shows that heat makes most materials expand.",
    "Percentages are a convenient way to compare parts of different wholes.",
    "Volcanoes form where molten rock from deep inside the planet reaches the surface.",
    "Bacteria are single-celled organisms, and many of them are helpful to humans.",
    "The Sun is a star, and it is the main source of energy for life on Earth.",
]
EN_EDU_INTROS = [
    "In this lesson, we will learn about {t}.",
    "This chapter explains the basic ideas of {t}.",
    "The following notes introduce key concepts in {t} for students.",
    "Let us study some important facts about {t}.",
]
EN_TOPICS = ["chemistry", "the human body", "history", "mathematics", "physics", "biology", "geography", "economics", "earth science"]

EN_INFO = [
    "The city council announced on Tuesday that a new bus route will open next month.",
    "Local shops reported higher sales during the long holiday weekend.",
    "The weather service expects light rain in the northern districts this week.",
    "The museum will extend its opening hours during the summer season.",
    "Officials said the repair work on the main bridge is almost complete.",
    "The new smartphone has a larger battery and a brighter screen than last year's model.",
    "Ticket prices for the national cricket final were announced this morning.",
    "The library is hosting a reading club for children every Saturday.",
    "Traffic was slow on the motorway because of road maintenance.",
    "A new restaurant serving traditional food has opened near the university.",
    "The airline added two extra flights a week between the two cities.",
    "Residents were asked to save water during the dry season.",
    "The hospital opened a new emergency ward with twenty beds.",
    "The festival attracted thousands of visitors from across the country.",
]
EN_INFO_T = [
    "{c} will host the {e} on {d}, the organisers said.",
    "The {e} in {c} has been moved to {d} because of the weather.",
    "Registration for the {e} in {c} closes on {d}.",
]
EN_CITIES = ["Lahore", "Karachi", "Islamabad", "Peshawar", "Quetta", "Multan", "London", "Toronto", "Dubai"]
EN_EVENTS = ["book fair", "food festival", "science exhibition", "cricket match", "job fair", "music concert", "marathon"]
EN_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

EN_CASUAL = [
    "lol that match yesterday was crazy",
    "omg I can't believe the ending of that show haha",
    "anyone know a good pizza place near the mall?",
    "my phone battery died again, so annoying",
    "brb gonna grab some chai",
    "honestly this weekend was the best!!",
    "idk why everyone is talking about this movie",
    "cant wait for the holidays tbh",
    "who else is still awake lol",
    "that new song is stuck in my head haha",
    "bro the traffic today was insane",
    "just finished my exams, finally free!!",
    "wanna go for a drive later?",
    "this coffee shop has the best wifi ever",
    "my cat knocked over my plant again",
]
EN_EMOJI = ["\U0001F602", "\U0001F525", "\U0001F60D", "\U0001F44D", "\U0001F64F"]

EN_SPAM_T = [
    "BUY CHEAP {p} ONLINE!!! Best price guaranteed, click here {u} FREE shipping!!!",
    "{p} {p} {p} discount sale 70% off limited offer, order now at {u}",
    "Earn $5000 per week from home, no experience needed, click now {u}!!!",
    "Best {p} deals, cheap {p}, discount {p}, buy {p} today with free bonus!!!",
    "WIN a FREE {p} now!!! Limited offer, click the link {u} and claim your prize!!!",
]
EN_PRODUCTS = ["watches", "sunglasses", "phones", "shoes", "laptops", "perfume", "supplements"]

BOILERPLATE_EN = [
    "Home | About Us | Contact | Privacy Policy | Terms of Service",
    "We use cookies to improve your experience. Accept all cookies.",
    "\u00a9 2024 All rights reserved.",
    "Login | Register | Forgot password?",
    "Subscribe to our newsletter",
    "Share on Facebook | Twitter | WhatsApp",
    "Skip to main content",
    "Related posts",
    "Previous | Next",
]

# --------------------------------------------------------------------------- Urdu
UR_ELEMENTS = [
    ("ہائیڈروجن", "H", 1, "کائنات کا سب سے ہلکا اور سب سے زیادہ پایا جانے والا عنصر ہے"),
    ("آکسیجن", "O", 8, "زیادہ تر جانداروں میں سانس لینے کے عمل کے لیے ضروری ہے"),
    ("کاربن", "C", 6, "تمام نامیاتی مالیکیولز کی بنیاد بناتا ہے"),
    ("لوہا", "Fe", 26, "فولاد کا بنیادی جزو ہے اور ہمارے خون کی ہیموگلوبن میں بھی پایا جاتا ہے"),
    ("تانبا", "Cu", 29, "بجلی کی تاروں میں استعمال ہوتا ہے کیونکہ یہ بجلی کو اچھی طرح گزارتا ہے"),
    ("سونا", "Au", 79, "ایک بھاری دھات ہے جسے زنگ نہیں لگتا"),
    ("کیلشیم", "Ca", 20, "ہڈیوں اور دانتوں کی مضبوطی کے لیے اہم معدنیات ہے"),
    ("نائٹروجن", "N", 7, "زمین کے کرۂ ہوائی کا تقریباً اٹھہتر فیصد حصہ ہے"),
    ("سلیکان", "Si", 14, "کمپیوٹر چپس بنانے میں استعمال ہونے والا اہم مادہ ہے"),
    ("سوڈیم", "Na", 11, "ایک نرم دھات ہے جو پانی کے ساتھ شدید ردعمل کرتی ہے"),
    ("ہیلیم", "He", 2, "ایک ہلکی گیس ہے جو دوسرے عناصر کے ساتھ بہت کم ردعمل کرتی ہے"),
    ("یورینیم", "U", 92, "ایک بھاری تابکار عنصر ہے جو جوہری بجلی گھروں میں ایندھن کے طور پر استعمال ہوتا ہے"),
]
UR_ELEMENT_T = [
    "{0} کا کیمیائی نشان {1} اور جوہری عدد {2} ہے، اور یہ {3}۔",
    "عنصر {0} ({1}) {3}۔",
    "دوری جدول میں عنصر نمبر {2} {0} ہے، جو {3}۔",
]

UR_ORGANS = [
    ("دل", "خون کو پورے جسم میں پمپ کرتا ہے"),
    ("پھیپھڑے", "خون اور ہوا کے درمیان آکسیجن اور کاربن ڈائی آکسائیڈ کا تبادلہ کرتے ہیں"),
    ("جگر", "خون سے زہریلے مادوں کو صاف کرتا ہے اور ہاضمے کے لیے صفرا بناتا ہے"),
    ("گردے", "خون سے فاضل مادے نکالتے ہیں اور پیشاب بناتے ہیں"),
    ("دماغ", "سوچ، یادداشت اور جسم کی حرکات کو کنٹرول کرتا ہے"),
    ("معدہ", "تیزاب اور خامروں کی مدد سے خوراک کو توڑتا ہے"),
    ("جلد", "جسم کو انفیکشن سے بچاتی ہے اور درجہ حرارت کو متوازن رکھتی ہے"),
    ("لبلبہ", "انسولین بناتا ہے جو خون میں شکر کی مقدار کو کنٹرول کرتی ہے"),
    ("آنکھیں", "روشنی کو پردۂ چشم پر مرکوز کرتی ہیں تاکہ دماغ تصویر بنا سکے"),
]
UR_ORGAN_T = [
    "انسانی جسم میں {0} {1}۔",
    "{0} {1}۔",
    "ڈاکٹروں کے مطابق {0} {1}۔",
]

UR_CONCEPTS = [
    ("ضیائی تالیف", "وہ عمل ہے جس میں سبز پودے سورج کی روشنی سے پانی اور کاربن ڈائی آکسائیڈ کو گلوکوز اور آکسیجن میں بدلتے ہیں"),
    ("مفرد عدد", "ایک سے بڑا وہ مکمل عدد ہے جو صرف ایک اور اپنے آپ پر تقسیم ہوتا ہے"),
    ("الگورتھم", "کسی مسئلے کو حل کرنے کے لیے واضح اقدامات کا ایک محدود سلسلہ ہے"),
    ("کشش ثقل", "وہ قوت ہے جو کمیت رکھنے والی اشیاء کو ایک دوسرے کی طرف کھینچتی ہے"),
    ("افراط زر", "قیمتوں میں عمومی اضافہ ہے جو پیسے کی قوت خرید کو کم کر دیتا ہے"),
    ("جمہوریت", "حکومت کا وہ نظام ہے جس میں شہری انتخابات کے ذریعے اپنے رہنما چنتے ہیں"),
    ("کٹاؤ", "پانی، ہوا یا برف کے ذریعے چٹانوں اور مٹی کا آہستہ آہستہ گھس جانا ہے"),
    ("ماحولیاتی نظام", "جانداروں اور ان کے طبعی ماحول کے باہمی تعلق کا نام ہے"),
    ("تبخیر", "وہ عمل ہے جس میں مائع گیس میں تبدیل ہو جاتا ہے"),
    ("کسر", "ایسا عدد ہے جو کسی مکمل چیز کے ایک حصے کو ظاہر کرتا ہے"),
    ("مشین لرننگ", "کمپیوٹر سائنس کا وہ شعبہ ہے جس میں نظام ڈیٹا سے نمونے سیکھتے ہیں"),
    ("آبی چکر", "پانی کی زمین، سمندروں اور فضا کے درمیان مسلسل گردش کا نام ہے"),
    ("مفروضہ", "ایک قابل جانچ وضاحت ہے جو سائنس دان تجربے سے پہلے پیش کرتے ہیں"),
    ("عمل انگیز", "ایسا مادہ ہے جو کیمیائی ردعمل کو تیز کرتا ہے لیکن خود استعمال نہیں ہوتا"),
    ("حرکی توانائی", "وہ توانائی ہے جو کسی جسم میں اس کی حرکت کی وجہ سے ہوتی ہے"),
    ("خط استوا", "ایک فرضی لکیر ہے جو زمین کو شمالی اور جنوبی نصف کرے میں تقسیم کرتی ہے"),
]
UR_CONCEPT_T = [
    "{0} {1}۔",
    "آسان الفاظ میں، {0} {1}۔",
    "طلبہ کو یاد رکھنا چاہیے کہ {0} {1}۔",
    "تعریف کے مطابق {0} {1}۔",
]

UR_HISTORY = [
    ("وادی سندھ کی تہذیب", "تقریباً ڈھائی ہزار سال قبل مسیح میں موجودہ پاکستان کے علاقے میں پروان چڑھی"),
    ("مغلیہ سلطنت", "سولہویں سے انیسویں صدی تک برصغیر کے بڑے حصے پر حکمران رہی"),
    ("لاہور کی بادشاہی مسجد", "سن سولہ سو تہتر میں اورنگزیب کے دور میں مکمل ہوئی"),
    ("موہنجو داڑو", "قدیم وادی سندھ کے سب سے بڑے شہروں میں سے ایک تھا"),
    ("شاہراہ ریشم", "چین کو مشرق وسطیٰ اور یورپ سے تجارتی راستوں کے ذریعے جوڑتی تھی"),
    ("صنعتی انقلاب", "اٹھارویں صدی میں برطانیہ سے شروع ہوا اور اس نے صنعت کو بدل دیا"),
    ("چھاپہ خانہ", "پندرہویں صدی میں ایجاد ہوا اور اس سے کتابیں بہت سستی ہو گئیں"),
    ("پاکستان", "چودہ اگست انیس سو سینتالیس کو ایک آزاد ملک کے طور پر وجود میں آیا"),
    ("بغداد کا بیت الحکمت", "نویں صدی میں ترجمے اور علم کا ایک بڑا مرکز تھا"),
]
UR_HISTORY_T = [
    "مورخین کے مطابق {0} {1}۔",
    "{0} {1}۔",
    "یہ جاننا ضروری ہے کہ {0} {1}۔",
]

UR_EDU_SENTENCES = [
    "سائنس کا مقصد فطرت کے اصولوں کو تجربے اور مشاہدے کے ذریعے سمجھنا ہے۔",
    "مثال کے طور پر، جب پانی کو سو ڈگری سینٹی گریڈ تک گرم کیا جاتا ہے تو وہ بھاپ بن جاتا ہے۔",
    "اس لیے اساتذہ طلبہ کو سوال پوچھنے اور خود تجربات کرنے کی ترغیب دیتے ہیں۔",
    "ریاضی میں مساوات دو برابر مقداروں کے درمیان تعلق کو ظاہر کرتی ہے۔",
    "زمین سورج کے گرد ایک چکر تقریباً تین سو پینسٹھ دن میں مکمل کرتی ہے۔",
    "چاند کی اپنی روشنی نہیں ہوتی بلکہ وہ سورج کی روشنی منعکس کرتا ہے۔",
    "اردو زبان کا رسم الخط نستعلیق کہلاتا ہے اور یہ دائیں سے بائیں لکھی جاتی ہے۔",
    "علامہ اقبال نے اپنی شاعری کے ذریعے نوجوانوں کو خودی کا پیغام دیا۔",
    "پودوں کو بڑھنے کے لیے پانی، روشنی اور غذائی اجزاء کی ضرورت ہوتی ہے۔",
    "کیونکہ ہوا میں آکسیجن موجود ہے، اس لیے آگ جلنے کے لیے ہوا ضروری ہے۔",
    "اس سبق میں ہم یہ سیکھیں گے کہ بارش کیسے بنتی ہے۔",
    "تجربے کے نتائج کو ہمیشہ احتیاط سے لکھنا چاہیے تاکہ دوسرے انہیں دہرا سکیں۔",
    "کمپیوٹر معلومات کو صفر اور ایک کی صورت میں محفوظ کرتا ہے۔",
    "دریائے سندھ پاکستان کا سب سے بڑا دریا ہے اور زراعت کے لیے بہت اہم ہے۔",
    "صحت مند غذا میں پھل، سبزیاں، اناج اور پروٹین شامل ہونے چاہییں۔",
    "آواز ہوا میں لہروں کی صورت میں سفر کرتی ہے۔",
    "روشنی کی رفتار تقریباً تین لاکھ کلومیٹر فی سیکنڈ ہے۔",
    "تاریخ کا مطالعہ ہمیں ماضی کی غلطیوں سے سیکھنے میں مدد دیتا ہے۔",
    "جغرافیہ زمین کی سطح، آب و ہوا اور انسانی آبادی کا مطالعہ ہے۔",
    "اس کا مطلب یہ ہے کہ ہر عمل کا ایک برابر اور مخالف ردعمل ہوتا ہے۔",
    "نیوٹن کے حرکت کے قوانین طبیعیات کی بنیاد سمجھے جاتے ہیں۔",
    "خلیہ زندگی کی بنیادی اکائی ہے اور تمام جاندار خلیوں سے مل کر بنے ہیں۔",
    "ویکسین جسم کے مدافعتی نظام کو بیماریوں سے لڑنے کے لیے تیار کرتی ہے۔",
    "اسی طرح، درخت مٹی کے کٹاؤ کو روکنے میں مدد کرتے ہیں۔",
    "نتیجہ یہ نکلتا ہے کہ مسلسل مشق سے ہی کوئی مہارت حاصل ہوتی ہے۔",
    "قطب شمالی پر سردیوں میں کئی مہینوں تک سورج نظر نہیں آتا۔",
    "مقناطیس کے دو قطب ہوتے ہیں جنہیں شمالی اور جنوبی قطب کہا جاتا ہے۔",
    "زلزلے زمین کی پلیٹوں کی اچانک حرکت کی وجہ سے آتے ہیں۔",
    "مستطیل کا رقبہ معلوم کرنے کے لیے اس کی لمبائی کو چوڑائی سے ضرب دیں۔",
    "بادل اس وقت بنتے ہیں جب آبی بخارات ٹھنڈے ہو کر ننھے قطروں میں بدل جاتے ہیں۔",
    "سائنسی طریقہ کار ایک سوال سے شروع ہوتا ہے اور شواہد پر مبنی نتیجے پر ختم ہوتا ہے۔",
    "فوسلز ان پودوں اور جانوروں کا ثبوت ہیں جو لاکھوں سال پہلے زندہ تھے۔",
    "لہذا تجربہ ظاہر کرتا ہے کہ حرارت سے زیادہ تر اشیاء پھیل جاتی ہیں۔",
    "آتش فشاں وہاں بنتے ہیں جہاں زمین کے اندر سے پگھلی ہوئی چٹان سطح تک پہنچتی ہے۔",
    "سورج ایک ستارہ ہے اور زمین پر زندگی کے لیے توانائی کا بنیادی ذریعہ ہے۔",
]
UR_EDU_INTROS = [
    "اس سبق میں ہم {t} کے بارے میں سیکھیں گے۔",
    "یہ باب {t} کے بنیادی تصورات کی وضاحت کرتا ہے۔",
    "درج ذیل نوٹس طلبہ کے لیے {t} کے اہم تصورات بیان کرتے ہیں۔",
    "آئیے {t} کے بارے میں چند اہم حقائق کا مطالعہ کریں۔",
]
UR_TOPICS = ["کیمیا", "انسانی جسم", "تاریخ", "ریاضی", "طبیعیات", "حیاتیات", "جغرافیہ", "معیشت"]

UR_INFO = [
    "محکمہ موسمیات نے اگلے ہفتے بارش کی پیش گوئی کی ہے۔",
    "شہر میں نئی بس سروس کا آغاز اگلے مہینے ہوگا۔",
    "مقامی دکانداروں کے مطابق عید سے پہلے خریداری میں اضافہ ہوا ہے۔",
    "حکومت نے سڑکوں کی مرمت کے لیے نئے منصوبے کا اعلان کیا ہے۔",
    "ٹیم نے کل رات کا میچ تین وکٹوں سے جیت لیا۔",
    "پٹرول کی قیمتوں میں معمولی کمی کی گئی ہے۔",
    "ہسپتال میں نیا ایمرجنسی وارڈ کھول دیا گیا ہے۔",
    "یونیورسٹی میں داخلوں کی آخری تاریخ پندرہ ستمبر ہے۔",
    "موٹروے پر مرمت کے کام کی وجہ سے ٹریفک سست رہی۔",
    "شہریوں سے پانی بچانے کی اپیل کی گئی ہے۔",
    "میلے میں ملک بھر سے ہزاروں لوگوں نے شرکت کی۔",
    "نئے موبائل فون میں بڑی بیٹری اور روشن اسکرین ہے۔",
]
UR_INFO_T = [
    "{c} میں {e} {d} کو منعقد ہوگا، منتظمین نے بتایا۔",
    "{c} میں ہونے والا {e} موسم کی وجہ سے {d} تک ملتوی کر دیا گیا ہے۔",
    "{c} کے {e} کے لیے رجسٹریشن {d} کو بند ہو جائے گی۔",
]
UR_CITIES = ["لاہور", "کراچی", "اسلام آباد", "پشاور", "کوئٹہ", "ملتان", "فیصل آباد", "حیدرآباد"]
UR_EVENTS = ["کتاب میلہ", "کرکٹ میچ", "تعلیمی کانفرنس", "کھانوں کا میلہ", "سائنس نمائش", "مشاعرہ"]
UR_DAYS = ["پیر", "منگل", "بدھ", "جمعرات", "جمعہ", "ہفتہ", "اتوار"]

UR_CASUAL = [
    "یار کل کا میچ تو کمال تھا ہاہاہا",
    "کوئی بتائے گا اچھی بریانی کہاں ملتی ہے؟",
    "آج پھر بجلی چلی گئی، بہت تنگ ہوں",
    "واہ کیا زبردست ڈرامہ تھا!!",
    "میں تو بس چائے پینے جا رہا ہوں",
    "سب کو عید مبارک",
    "یہ گانا سن کر مزہ آ گیا",
    "امتحان سر پر ہیں اور میں فون پر ہوں لول",
    "یار آج موسم بہت اچھا ہے، باہر چلیں؟",
    "میری بلی نے پھر گلدان توڑ دیا",
    "کل رات دیر تک فلم دیکھتے رہے",
    "یار یہ ٹریفک کبھی ٹھیک نہیں ہوگی",
]
UR_SPAM_T = [
    "سستے {p} خریدیں!!! بہترین قیمت، ابھی کلک کریں {u} مفت ڈیلیوری!!!",
    "{p} {p} {p} رعایت سیل ستر فیصد چھوٹ محدود پیشکش، ابھی آرڈر کریں {u}",
    "گھر بیٹھے ہزاروں روپے کمائیں، ابھی کلک کریں {u}!!!",
    "بہترین {p} سستے {p} رعایت {p} آج ہی خریدیں مفت تحفہ!!!",
]
UR_PRODUCTS = ["موبائل فون", "گھڑیاں", "جوتے", "کپڑے", "لیپ ٹاپ", "عطر"]
BOILERPLATE_UR = [
    "صفحہ اول | ہمارے بارے میں | رابطہ | رازداری کی پالیسی",
    "جملہ حقوق محفوظ ہیں۔",
    "لاگ ان | رجسٹر کریں",
    "ہمارا نیوز لیٹر سبسکرائب کریں",
    "شیئر کریں: فیس بک | ٹویٹر | واٹس ایپ",
    "متعلقہ خبریں",
]

# --------------------------------------------------------------------------- Roman Urdu
UR_LATN_CASUAL = [
    "yaar kal ka match bohat zabardast tha",
    "koi batayega acha biryani kahan milti hai?",
    "aaj phir light chali gayi yaar, bohat tang hoon",
    "main to bas chai peene ja raha hoon",
    "ye gaana sun kar maza aa gaya",
    "exam sar pe hain aur main phone pe hoon lol",
    "aaj mausam bohat acha hai, bahar chalein?",
    "kal raat der tak movie dekhte rahe",
    "tum kahan ho, sab intezar kar rahe hain",
    "ye traffic kabhi theek nahi hogi yaar",
]

# --------------------------------------------------------------------------- Other languages
OTHER_LANGS: Dict[str, List[str]] = {
    "ar": [
        "الماء ضروري لجميع الكائنات الحية على الأرض.",
        "تدور الأرض حول الشمس مرة واحدة كل عام.",
        "تعتبر القراءة من أهم الوسائل لاكتساب المعرفة.",
        "يتكون جسم الإنسان من ملايين الخلايا.",
        "كانت بغداد مركزا مهما للعلم في العصر العباسي.",
        "تساعد الأشجار في تنقية الهواء من الملوثات.",
        "الرياضيات هي لغة العلوم الطبيعية.",
        "يقع نهر النيل في شمال شرق قارة أفريقيا.",
        "هذا الكتاب مفيد جدا للطلاب في المدرسة.",
    ],
    "fa": [
        "آب برای همه موجودات زنده ضروری است.",
        "زمین هر سال یک بار به دور خورشید می‌چرخد.",
        "خواندن کتاب یکی از بهترین راه‌های یادگیری است.",
        "شیراز شهر شاعران بزرگی مانند حافظ و سعدی است.",
        "درختان هوا را از آلودگی پاک می‌کنند.",
        "ریاضیات زبان علوم طبیعی است.",
        "این کتاب برای دانش‌آموزان دبیرستان نوشته شده است.",
        "تهران پایتخت ایران است و جمعیت زیادی دارد.",
    ],
    "hi": [
        "पानी सभी जीवों के लिए आवश्यक है।",
        "पृथ्वी हर साल सूर्य का एक चक्कर लगाती है।",
        "किताबें पढ़ना ज्ञान प्राप्त करने का सबसे अच्छा तरीका है।",
        "मानव शरीर लाखों कोशिकाओं से बना है।",
        "गंगा भारत की सबसे लंबी नदी है।",
        "पेड़ हवा को साफ करने में मदद करते हैं।",
        "गणित विज्ञान की भाषा है।",
        "यह किताब स्कूल के छात्रों के लिए बहुत उपयोगी है।",
    ],
    "fr": [
        "L'eau est essentielle à tous les êtres vivants sur la Terre.",
        "La Terre tourne autour du Soleil en une année environ.",
        "La lecture est l'un des meilleurs moyens d'apprendre.",
        "Le corps humain est composé de milliards de cellules.",
        "Paris est la capitale de la France et une ville très visitée.",
        "Les arbres aident à purifier l'air que nous respirons.",
        "Les mathématiques sont le langage des sciences naturelles.",
        "Ce livre est très utile pour les élèves du lycée.",
    ],
    "es": [
        "El agua es esencial para todos los seres vivos de la Tierra.",
        "La Tierra gira alrededor del Sol una vez al año.",
        "La lectura es una de las mejores formas de aprender.",
        "El cuerpo humano está formado por millones de células.",
        "Madrid es la capital de España y tiene muchos museos.",
        "Los árboles ayudan a limpiar el aire que respiramos.",
        "Las matemáticas son el lenguaje de las ciencias naturales.",
        "Este libro es muy útil para los estudiantes de la escuela.",
    ],
    "sw": [
        "Maji ni muhimu kwa viumbe vyote vilivyo hai duniani.",
        "Dunia inazunguka jua mara moja kila mwaka.",
        "Kusoma vitabu ni njia bora ya kupata maarifa.",
        "Mwili wa binadamu umeundwa na seli nyingi sana.",
        "Mlima Kilimanjaro ni mlima mrefu zaidi barani Afrika.",
        "Miti husaidia kusafisha hewa tunayovuta.",
        "Hisabati ni lugha ya sayansi ya asili.",
        "Kitabu hiki ni muhimu sana kwa wanafunzi wa shule.",
    ],
    "ps": [
        "اوبه د ټولو ژوندیو موجوداتو لپاره اړینه ده.",
        "ځمکه هر کال یو ځل د لمر شاوخوا ګرځي.",
        "د کتابونو لوستل د زده کړې غوره لاره ده.",
        "پېښور د پښتونخوا یو لرغونی ښار دی.",
        "ونې هوا له ککړتیا څخه پاکوي.",
        "د انسان بدن له ډېرو حجرو څخه جوړ دی.",
        "دا کتاب د ښوونځي زده کوونکو لپاره ګټور دی.",
    ],
    "sd": [
        "پاڻي سڀني جاندارن لاءِ ضروري آهي.",
        "ڌرتي هر سال سج جي چوڌاري هڪ چڪر لڳائي ٿي.",
        "ڪتاب پڙهڻ علم حاصل ڪرڻ جو بهترين طريقو آهي.",
        "ڪراچي سنڌ جو سڀ کان وڏو شهر آهي.",
        "وڻ هوا کي صاف ڪرڻ ۾ مدد ڪن ٿا.",
        "هي ڪتاب اسڪول جي شاگردن لاءِ تمام ڪارآمد آهي.",
    ],
}

# --------------------------------------------------------------------------- Toxic & PII
TOXIC = {
    "en": [
        "You are a stupid idiot and nobody wants you here.",
        "Shut up, you pathetic loser.",
        "People like you are worthless trash.",
        "I will kill you if you come here again.",
        "Those people are vermin and should be driven out.",
        "What a dumb moron, get lost.",
    ],
    "ur": [
        "تم بیوقوف اور جاہل ہو۔",
        "چپ کرو گدھے، تمہیں کچھ نہیں آتا۔",
        "یہ لوگ کمینے اور بے غیرت ہیں۔",
        "میں تمہیں جان سے مار دوں گا۔",
    ],
    "ur_Latn": ["tum bewakoof aur jahil ho", "chup kar gadhay, tujhe kuch nahi aata", "main tumhe maar dunga"],
}

EN_NAMES = ["Ayesha Khan", "Bilal Ahmed", "Sara Malik", "Usman Tariq", "Fatima Raza", "John Carter", "Maria Lopez"]
UR_NAMES = ["علی رضا", "فاطمہ بی بی", "احمد خان", "زینب ملک", "عمر فاروق"]
EMAIL_DOMAINS = ["example.com", "mail.example.pk", "school.edu.pk", "inbox.example.org"]

PII_TEMPLATES_EN = {
    "EMAIL": "For more details, contact the office at {email}.",
    "PHONE": "You can call the help desk on {phone} during working hours.",
    "CNIC": "Applicants must provide a CNIC number such as {cnic} at registration.",
    "CREDIT_CARD": "Payment was made with card {card} last month.",
    "IP_ADDRESS": "The server log shows a request from {ip} at midnight.",
}
PII_TEMPLATES_UR = {
    "EMAIL": "مزید معلومات کے لیے {email} پر ای میل کریں۔",
    "PHONE": "مزید معلومات کے لیے {phone} پر رابطہ کریں۔",
    "CNIC": "داخلے کے لیے شناختی کارڈ نمبر {cnic} درج کریں۔",
    "CREDIT_CARD": "ادائیگی کارڈ نمبر {card} سے کی گئی۔",
}
