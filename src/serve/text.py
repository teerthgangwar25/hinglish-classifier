"""Text cleaning and cheap input features for serving.

This is a copy of the functions in src/preprocess.py and src/langmix.py that the API needs, written
without pandas so the Docker image stays small. tests/test_serve.py checks that the copies give the
same answers as the originals, so they cannot drift apart silently.
"""
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\S+")
REPEAT_RE = re.compile(r"(.)\1{2,}")           # same character 3+ times in a row
SPACE_RE = re.compile(r"\s+")
DEVA = re.compile(r"[\u0900-\u097F]")
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2B50\u2764]")


def clean_text(text: str) -> str:
    """Exactly what the model saw in training: URLs -> [URL], @mentions -> [USER], 'sooooo' -> 'soo'."""
    text = URL_RE.sub("[URL]", text)
    text = MENTION_RE.sub("[USER]", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
    return SPACE_RE.sub(" ", text).strip()


def script_class(text: str) -> str:
    d = len(DEVA.findall(text))
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if d + latin == 0:
        return "neither"
    share = d / (d + latin)
    return "roman" if share < 0.05 else ("devanagari" if share > 0.95 else "mixed")


def _body(text: str) -> str:
    """Comment body: no links, mentions or hashtags (their Latin letters would hide Devanagari text)."""
    for rx in (URL_RE, MENTION_RE, HASHTAG_RE):
        text = rx.sub(" ", text)
    return clean_text(text)


def features(text: str) -> dict:
    """Cheap per-request features that go into the log (never the text itself)."""
    body = _body(text)
    d = len(DEVA.findall(body))
    latin = sum(1 for ch in body if ch.isascii() and ch.isalpha())
    return {
        "n_chars": len(text),
        "n_words": len(text.split()),
        "script": script_class(body),
        "deva_share": round(d / (d + latin), 3) if d + latin else 0.0,
        "has_hashtag": int("#" in text),
        "has_mention": int(bool(MENTION_RE.search(text))),
        "has_url": int(bool(URL_RE.search(text))),
        "has_emoji": int(bool(EMOJI.search(text))),
    }
