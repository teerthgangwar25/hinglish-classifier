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
