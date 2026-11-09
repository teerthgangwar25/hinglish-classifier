"""Per-comment language-mix features (decision D12a): script class plus Hindi share of the words.

Rule: a word with any Devanagari letter is Hindi. Roman words are tagged by HingBERT-LID
(l3cube-pune/hing-bert-lid, a token-level Hindi/English tagger). The output csv holds ids and
counts only (no text), so it is safe to commit.

    python src/langmix.py --inspect          # FIRST: print the model's labels + 3 toy examples
    python src/langmix.py --split dev        # writes splits/langmix_dev.csv
Needs the internet on first use (model download, ~0.4 GB).
"""
import argparse
import re

import pandas as pd

from preprocess import ROOT, clean_text, load_frozen_split

DEVA = re.compile(r"[\u0900-\u097F]")
EDGE = re.compile(r"^[\W_]+|[\W_]+$")      # punctuation / hashtag marks at the edges of a token


def script_class(text: str) -> str:
    d = len(DEVA.findall(text))
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    if d + latin == 0:
        return "neither"
    share = d / (d + latin)
    return "roman" if share < 0.05 else ("devanagari" if share > 0.95 else "mixed")


def words_of(text: str) -> list[str]:
    """Whitespace tokens that contain a letter; [URL]/[USER] tokens and bare numbers/emojis are dropped."""
    out = []
    for tok in text.split():
        if tok in ("[URL]", "[USER]"):
            continue
        tok = EDGE.sub("", tok)
        if tok and any(ch.isalpha() for ch in tok):
            out.append(tok)
    return out


def mix_summary(words: list[str], tags: list[str]) -> dict:
    n_hi = sum(1 for t in tags if t == "hi")
    n_en = sum(1 for t in tags if t == "en")
    total = n_hi + n_en
    share = n_hi / total if total else float("nan")
    if total < 3:
        bucket = "too-few-words"
    elif share >= 0.7:
        bucket = "mostly-Hindi"
    elif share <= 0.3:
        bucket = "mostly-English"
    else:
        bucket = "balanced"
    return {"n_words": len(words), "n_hi": n_hi, "n_en": n_en, "hi_share": share, "mix_bucket": bucket}


class LIDTagger:
    """Word-level tagger. Label names are read from the model config and mapped to hi / en / other."""

    def __init__(self, en_labels=None, hi_labels=None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        mid = "l3cube-pune/hing-bert-lid"
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForTokenClassification.from_pretrained(mid).to(self.dev).eval()
        self.id2label = self.model.config.id2label
        en = {s.strip().lower() for s in (en_labels or "en,eng,english,e,lang1").split(",")}
        hi = {s.strip().lower() for s in (hi_labels or "hi,hin,hindi,h,lang2").split(",")}
        self.map = {i: ("en" if str(l).lower() in en else "hi" if str(l).lower() in hi else "other")
                    for i, l in self.id2label.items()}

    def tag_batch(self, batch_words: list[list[str]]) -> list[list[str]]:
        torch = self.torch
        enc = self.tok(batch_words, is_split_into_words=True, truncation=True, max_length=128,
                       padding=True, return_tensors="pt")
        with torch.no_grad():
            pred = self.model(**{k: v.to(self.dev) for k, v in enc.items()}).logits.argmax(-1).cpu().tolist()
        result = []
        for b, words in enumerate(batch_words):
            tags, seen = ["other"] * len(words), set()
            for pos, wid in enumerate(enc.word_ids(b)):
                if wid is not None and wid not in seen:      # label of the first sub-token of each word
                    seen.add(wid)
                    tags[wid] = self.map[pred[b][pos]]
            result.append(tags)
        return result


def tag_texts(texts: list[str], tagger: LIDTagger, batch: int = 64) -> list[dict]:
    rows = []
    for i in range(0, len(texts), batch):
        chunk = [clean_text(t) for t in texts[i:i + batch]]
        word_lists = [words_of(t) for t in chunk]
        roman_idx = [(j, k) for j, ws in enumerate(word_lists) for k, w in enumerate(ws) if not DEVA.search(w)]
        # tag only the Roman words, in one pass per comment
        tags = [["hi" if DEVA.search(w) else "other" for w in ws] for ws in word_lists]
        todo = [j for j in sorted({j for j, _ in roman_idx})]
        if todo:
            lid = tagger.tag_batch([word_lists[j] for j in todo])
            for j, lid_tags in zip(todo, lid):
                for k, w in enumerate(word_lists[j]):
                    if not DEVA.search(w):
                        tags[j][k] = lid_tags[k]
        for ws, tg, raw in zip(word_lists, tags, chunk):
            rows.append({**mix_summary(ws, tg), "script": script_class(raw)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "fb_test", "tw_test"])
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--en_labels"); ap.add_argument("--hi_labels")
    a = ap.parse_args()
    tagger = LIDTagger(a.en_labels, a.hi_labels)
    print("model labels:", tagger.id2label, "-> mapped:", tagger.map)
    if a.inspect:
        for s in ["yeh movie bahut achhi hai but the ending was bad", "this is a normal english sentence",
                  "tum log bilkul pagal ho kya"]:
            ws = words_of(s)
            print(list(zip(ws, tagger.tag_batch([ws])[0])))
        return
    df = load_frozen_split(a.split)
    rows = tag_texts(df["text"].tolist(), tagger)
    out = pd.concat([df[["id"]].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    path = ROOT / "splits" / f"langmix_{a.split}.csv"
    out.to_csv(path, index=False)
    print(out["mix_bucket"].value_counts().to_string(), "\nwrote", path)


if __name__ == "__main__":
    main()
