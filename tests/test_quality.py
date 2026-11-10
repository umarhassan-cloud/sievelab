from sievelab.quality import EduHeuristicAnnotator, EduQualityClassifier, HeuristicConfig, apply_rules, regression_report
from sievelab.synth import CorpusGenerator

EDU = ("In this lesson, we will learn about biology. The cell is the basic unit of life, and all living things are "
       "made of cells. For example, the human body contains trillions of cells that work together. Because cells "
       "need energy, they break down glucose in a process called respiration. Scientists study cells with "
       "microscopes, and this research helps doctors understand diseases. Therefore, the study of cells is an "
       "important part of science education. In other words, biology explains how organisms grow, reproduce and "
       "respond to their environment. Students should remember that every organism is built from cells, and that "
       "the structure of a cell is closely related to its function in the body.")
SPAM = "BUY CHEAP watches ONLINE!!! Best price, click here http://deals.example FREE shipping!!! discount sale offer"


def test_heuristics_pass_edu_reject_spam():
    cfg = HeuristicConfig()
    assert apply_rules(EDU, "en", cfg)[0]
    ok, reasons, _ = apply_rules(SPAM, "en", cfg)
    assert not ok and "too_short" in reasons


def test_heuristics_language_aware():
    cfg = HeuristicConfig(min_words=5)
    urdu = "پانی زندگی کے لیے ضروری ہے اور پودوں کو بڑھنے کے لیے روشنی کی ضرورت ہوتی ہے۔"
    ok, reasons, _ = apply_rules(urdu, "ur", cfg)
    assert ok, reasons
    assert "uppercase" not in apply_rules(urdu.upper(), "ur", cfg)[1]


def test_alpha_word_rule_penalises_math():
    math_text = "The sum of 345 and 712 is 1057. The product of 12 and 9 is 108. Multiplication is repeated addition. " * 3
    assert "non_alpha_words" in apply_rules(math_text, "en", HeuristicConfig(min_words=5))[1]
    assert "non_alpha_words" not in apply_rules(math_text, "en", HeuristicConfig(min_words=5, min_alpha_word_frac=0.6))[1]


def test_annotator_orders_documents():
    ann = EduHeuristicAnnotator()
    assert ann.score(EDU) >= 4
    assert ann.score(SPAM) == 0
    assert ann.score("lol that match was crazy haha\nomg the ending\nbrb gonna grab chai") <= 1


def test_classifier_learns_teacher_and_generalises():
    gen = CorpusGenerator(seed=3)
    docs = gen.quality_train_set(500)
    ann = EduHeuristicAnnotator()
    labels = [ann.score(d.text) for d in docs]
    clf = EduQualityClassifier().fit([d.text for d in docs[:400]], labels[:400])
    rep = regression_report(labels[400:], clf.predict([d.text for d in docs[400:]]))
    assert rep["qwk"] > 0.8
    assert clf.predict([EDU])[0] > clf.predict([SPAM])[0]


def test_classifier_save_load(tmp_path):
    clf = EduQualityClassifier().fit([EDU, SPAM] * 5, [5, 0] * 5)
    p = tmp_path / "m.pkl"
    clf.save(str(p))
    assert abs(EduQualityClassifier.load(str(p)).predict([EDU])[0] - clf.predict([EDU])[0]) < 1e-9
