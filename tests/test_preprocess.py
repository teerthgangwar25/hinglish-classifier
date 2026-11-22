from preprocess import clean_text, dedupe_key


def test_urls_and_mentions_become_tokens():
    assert clean_text("dekho http://t.co/abc @modi ji") == "dekho [URL] [USER] ji"


def test_repeated_letters_are_squeezed():
    assert clean_text("sooooo bura") == "soo bura"


def test_emoji_and_hashtag_are_kept():
    assert clean_text("bakwas 😂 #IndvsPak") == "bakwas 😂 #IndvsPak"


def test_devanagari_is_untouched():
    assert clean_text("यह बहुत बुरा है") == "यह बहुत बुरा है"


def test_dedupe_key_ignores_case_punctuation_and_links():
    a = dedupe_key("Tera BAAP bambani h!! http://t.co/x")
    b = dedupe_key("tera baap bambani h")
    assert a == b


def test_dedupe_key_keeps_devanagari_vowel_signs():
    # these two differ only by a vowel sign; they must NOT collapse into one key
    assert dedupe_key("बुरा") != dedupe_key("बरा")


def test_normalise_roman_unifies_variants_but_leaves_other_words():
    from preprocess import normalise_roman
    assert normalise_roman("Nhi yaar, ye bht bura h") == "nahi yaar, yeh bahut bura hai"
    assert normalise_roman("to the movie") == "to the movie"     # English 'to' is untouched


def test_prepare_text_defaults_equal_clean_text():
    from preprocess import clean_text, prepare_text
    t = "dekho http://t.co/x @modi sooooo bura"
    assert prepare_text(t) == clean_text(t)


def test_transliteration_gives_colloquial_roman():
    import pytest
    pytest.importorskip("indic_transliteration")
    from preprocess import prepare_text
    assert prepare_text("यह बहुत बुरा है", script="translit") == "yah bahut bura hai"
    assert prepare_text("तुम लोग पागल हो क्या", script="translit", norm=True) == "tum log pagal ho kya"
