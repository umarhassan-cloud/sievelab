from dataforge.safety import LexiconToxicityScorer, PIIDetector, ToxicityClassifier, luhn_valid, redact

DET = PIIDetector()


def types(text):
    return [m.type for m in DET.detect(text)]


def test_email_phone_cnic_card_ip():
    text = ("Mail ali.khan@example.pk or call +92 300 1234567. CNIC 35202-1234567-1. "
            "Card 4111 1111 1111 1111. Server 192.168.1.20.")
    assert sorted(types(text)) == ["CNIC", "CREDIT_CARD", "EMAIL", "IP_ADDRESS", "PHONE"]


def test_urdu_digits_are_detected():
    assert types("مزید معلومات کے لیے ۰۳۰۰-۱۲۳۴۵۶۷ پر رابطہ کریں۔") == ["PHONE"]
    assert types("شناختی کارڈ نمبر ۳۵۲۰۲-۱۲۳۴۵۶۷-۱ درج کریں۔") == ["CNIC"]


def test_luhn_rejects_random_numbers():
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")
    assert "CREDIT_CARD" not in types("Order number 4111 1111 1111 1112 has shipped.")


def test_invalid_ip_and_plain_numbers_not_flagged():
    assert types("Version 1.2.3.400 was released in 2024 with 300 fixes.") == []


def test_names_need_honorific():
    t = types("Dr. Ayesha Khan presented the results. Newton described gravity.")
    assert t == ["PERSON"]
    assert types("جناب علی رضا نے تقریب سے خطاب کیا۔") == ["PERSON"]


def test_redaction():
    text = "Write to sara@example.com today."
    assert redact(text, DET.detect(text)) == "Write to <EMAIL> today."


def test_toxicity_sentence_level():
    tox = LexiconToxicityScorer()
    clean = "Photosynthesis converts light into chemical energy. Plants release oxygen."
    assert tox.score(clean).score == 0.0
    doc = clean + " Comment: you are a stupid idiot."
    res = tox.score(doc)
    assert res.score >= 0.5 and res.toxic_sentences == [2]
    assert "idiot" not in tox.redact_sentences(doc)
    assert "Photosynthesis" in tox.redact_sentences(doc)


def test_toxicity_threats_and_urdu():
    tox = LexiconToxicityScorer()
    assert tox.score("I will kill you tomorrow.").score == 1.0
    assert tox.score("میں تمہیں جان سے مار دوں گا۔").score == 1.0
    assert tox.score("tum bewakoof aur jahil ho").score >= 0.5
    assert tox.score("St00pid id1ot").score >= 0.5  # leetspeak normalisation


def test_toxicity_classifier_trains():
    texts = ["you are a stupid idiot", "shut up loser", "worthless trash person", "go away moron"] * 3 + \
            ["plants need water", "the sun is a star", "read a book today", "the river is long"] * 3
    clf = ToxicityClassifier().fit(texts, [1] * 12 + [0] * 12)
    p = clf.predict_proba(["what an idiot", "the moon reflects light"])
    assert p[0] > p[1]
