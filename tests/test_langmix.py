import math

from langmix import mix_summary, script_class, words_of


def test_script_classes():
    assert script_class("yeh bahut bura hai") == "roman"
    assert script_class("यह बहुत बुरा है") == "devanagari"
    assert script_class("yeh बहुत bura") == "mixed"
    assert script_class("😂😂 123") == "neither"


def test_words_drop_urls_numbers_and_edge_punctuation():
    assert words_of("dekho [URL] 123 #IndvsPak, wow!!") == ["dekho", "IndvsPak", "wow"]


def test_mix_buckets():
    assert mix_summary(["a"] * 4, ["hi", "hi", "hi", "en"])["mix_bucket"] == "mostly-Hindi"
    assert mix_summary(["a"] * 4, ["en", "en", "en", "hi"])["mix_bucket"] == "mostly-English"
    assert mix_summary(["a"] * 4, ["hi", "hi", "en", "en"])["mix_bucket"] == "balanced"
    r = mix_summary(["a"], ["hi"])
    assert r["mix_bucket"] == "too-few-words" and r["n_hi"] == 1
    assert math.isnan(mix_summary(["a"], ["other"])["hi_share"])


class FakeTagger:
    """Stands in for HingBERT-LID: words in EN are English, everything else Hindi."""
    EN = {"movie", "bad", "the"}

    def tag_batch(self, batch_words):
        return [["en" if w.lower() in self.EN else "hi" for w in ws] for ws in batch_words]


def test_tag_texts_uses_script_rule_for_devanagari_and_tagger_for_roman():
    from langmix import tag_texts
    rows = tag_texts(["yeh movie bad hai", "यह movie बहुत", "[URL] 123"], FakeTagger())
    assert (rows[0]["n_hi"], rows[0]["n_en"]) == (2, 2) and rows[0]["script"] == "roman"
    assert (rows[1]["n_hi"], rows[1]["n_en"]) == (2, 1) and rows[1]["script"] == "mixed"
    assert rows[2]["mix_bucket"] == "too-few-words"
