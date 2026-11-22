"""Loading and light cleaning for the TRAC-1 Hindi data.

Nothing here is "learned" from data, so it is safe to apply before or after splitting.
Script handling (D3) is deliberately NOT touched: Roman and Devanagari text pass through as-is.
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

# Files have NO header row: id, text, 3-class label (OAG / CAG / NAG)
COLUMNS = ["id", "text", "label3"]
FILES = {
    "train": "agr_hi_train.csv",
    "dev": "agr_hi_dev.csv",
    "fb_test": "agr_hi_fb_gold.csv",   # in-domain test (Facebook)
    "tw_test": "agr_hi_tw_gold.csv",   # surprise test (Twitter)
}
EXPECTED_ROWS = {"train": 12000, "dev": 3001, "fb_test": 970, "tw_test": 1194}


def load_split(name: str) -> pd.DataFrame:
    """Read one raw file and add the binary label (D2b: NAG=0, OAG+CAG=1)."""
    df = pd.read_csv(
        RAW / FILES[name],
        header=None,
        names=COLUMNS,
        encoding="utf-8",          # NEVER open/save these in Excel; it garbles Devanagari
        dtype=str,
        keep_default_na=False,     # keep literal text like "nan" or "NA" as text
    )
    df["label"] = (df["label3"] != "NAG").astype(int)
    return df


# ---------- light cleaning (Phase 1, step 2) ----------
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
REPEAT_RE = re.compile(r"(.)\1{2,}")           # same char 3+ times in a row
SPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """URLs -> [URL], @mentions -> [USER], 'sooooo' -> 'soo', strip control chars.
    Emojis, hashtags and script are kept."""
    text = URL_RE.sub("[URL]", text)
    text = MENTION_RE.sub("[USER]", text)
    text = REPEAT_RE.sub(r"\1\1", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\n\t ")
    return SPACE_RE.sub(" ", text).strip()


def dedupe_key(text: str) -> str:
    """Aggressive normalisation used ONLY to detect duplicates (never for training).
    Drops URLs, mentions, punctuation, symbols and emojis; lowercases; keeps letters AND
    combining marks (matras), which a plain regex like [^\\w\\s] would wrongly delete
    from Devanagari."""
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = REPEAT_RE.sub(r"\1\1", text.casefold())
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] in "LMNZ")
    key = SPACE_RE.sub(" ", text).strip()
    return key if key else text.strip()      # empty after cleaning (e.g. only an emoji): fall back


def load_frozen_split(name: str) -> pd.DataFrame:
    """Load a split using the frozen id lists written by split.py (so every model sees
    exactly the same rows, in the same order, with duplicates and leaks already removed)."""
    ids = (ROOT / "splits" / f"{name}_ids.txt").read_text(encoding="utf-8").split()
    df = load_split(name)
    df = df[df["id"].isin(set(ids))].reset_index(drop=True)
    assert len(df) == len(ids), f"{name}: frozen ids do not match the raw file"
    return df


# ---------- ablation options (Layer A) ----------
# Hand-written, deliberately conservative groups of common Roman-Hindi spelling variants.
# Coverage is incomplete by design: only function words where the variants are unambiguous.
NORM_GROUPS = {
    "nahi": ["nahi", "nhi", "nai", "nahin", "nahee", "nahe"],
    "hai": ["hai", "hay", "hae", "hy", "h"],
    "kya": ["kya", "kyaa"],
    "kyun": ["kyun", "kyu", "kyon", "kyoon"],
    "bahut": ["bahut", "bohot", "bhot", "bohut", "bahot", "bhut", "bht", "bohat"],
    "bhi": ["bhi", "bi", "bhee"],
    "aap": ["aap", "ap"],
    "tum": ["tum", "tm"],
    "kuch": ["kuch", "kch", "kuchh"],
    "accha": ["accha", "acha", "achha", "achchha", "achcha"],
    "yeh": ["yeh", "ye", "yah", "yh"],
    "woh": ["woh", "wo", "vo", "voh"],
    "kaise": ["kaise", "kese", "kaisey"],
    "sab": ["sab", "sb"],
}
_NORM = {variant: canon for canon, variants in NORM_GROUPS.items() for variant in variants}
WORD_RE = re.compile(r"[A-Za-z]+")
DEVA_RUN = re.compile(r"[\u0900-\u097F]+")


def normalise_roman(text: str) -> str:
    """Map known Roman spelling variants to one form ('nhi' / 'nahin' -> 'nahi'). Case-insensitive."""
    return WORD_RE.sub(lambda m: _NORM.get(m.group(0).lower(), m.group(0)), text)


def transliterate_devanagari(text: str) -> str:
    """Devanagari -> Roman, approximately like people type it ('यह बहुत बुरा है' -> 'yah bahut bura hai').
    Uses ITRANS via the indic-transliteration package, then simplifies: drop the final 'a' after a
    consonant (schwa), turn nasal marks into 'n', lower-case. Internal schwas are left alone."""
    from indic_transliteration import sanscript

    def convert(m):
        s = sanscript.transliterate(m.group(0), sanscript.DEVANAGARI, sanscript.ITRANS)
        for old, new in (("RRi", "ri"), (".N", "n"), (".n", "n"), ("~N", "n"), ("M", "n"), ("H", "h")):
            s = s.replace(old, new)
        s = re.sub(r"(?<=[b-df-hj-np-tv-zB-DF-HJ-NP-TV-Z])a$", "", s)
        return s.lower()

    return DEVA_RUN.sub(convert, text)


def prepare_text(text: str, norm: bool = False, script: str = "asis") -> str:
    """The text a model sees. Defaults reproduce the main experiments exactly (clean_text only)."""
    t = clean_text(text)
    if script == "translit":
        t = transliterate_devanagari(t)
    if norm:
        t = normalise_roman(t)
    return t
