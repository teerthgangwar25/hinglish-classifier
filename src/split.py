"""Phase 1: build leakage-safe splits from the OFFICIAL TRAC-1 files and freeze them.

Policy (decision D2/D4):
  1. Test sets (FB gold, Twitter gold) are never modified. We only measure them.
  2. Inside train and inside dev: drop empty texts; drop whole groups of duplicates whose
     binary labels disagree (label noise); keep one copy of the agreeing duplicates.
  3. Remove from train AND dev anything that also appears in either test set.
  4. Remove from dev anything that also appears in train.
Outputs (counts and ids only, no text):
  splits/{train,dev,fb_test,tw_test}_ids.txt, splits/test_hashes.json, reports/dedupe_report.json
Run from the repo root:  python src/split.py
"""
import hashlib
import json
from pathlib import Path

import pandas as pd

from preprocess import EXPECTED_ROWS, FILES, RAW, ROOT, dedupe_key, load_split

SPLITS = ROOT / "splits"
REPORTS = ROOT / "reports"


def dedupe_within(df: pd.DataFrame, report: dict) -> pd.DataFrame:
    n0 = len(df)
    df = df[df["text"].str.strip() != ""].copy()
    report["empty_removed"] = n0 - len(df)

    df["key"] = df["text"].map(dedupe_key)
    # groups of identical keys whose binary labels differ -> ambiguous, drop all of them
    n_labels = df.groupby("key")["label"].transform("nunique")
    conflict = n_labels > 1
    report["conflict_groups"] = int(df.loc[conflict, "key"].nunique())
    report["conflict_rows_removed"] = int(conflict.sum())
    df = df[~conflict]

    n1 = len(df)
    df = df.drop_duplicates("key", keep="first")
    report["duplicate_rows_removed"] = n1 - len(df)
    return df


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    SPLITS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    raw = {name: load_split(name) for name in FILES}
    report = {"raw_rows": {k: len(v) for k, v in raw.items()}}
    for name, expected in EXPECTED_ROWS.items():
        if len(raw[name]) != expected:
            print(f"WARNING: {name} has {len(raw[name])} rows, expected {expected}. Different file version?")

    # keys of the untouched test sets
    test_keys = {}
    for name in ("fb_test", "tw_test"):
        df = raw[name]
        keys = df["text"].map(dedupe_key)
        report[f"{name}_internal_duplicate_rows"] = int(keys.duplicated().sum())  # measured, not removed
        test_keys[name] = set(keys)
    all_test = test_keys["fb_test"] | test_keys["tw_test"]

    clean = {}
    for name in ("train", "dev"):
        r = {}
        df = dedupe_within(raw[name], r)
        df["in_fb_test"] = df["key"].isin(test_keys["fb_test"])
        df["in_tw_test"] = df["key"].isin(test_keys["tw_test"])
        r["overlap_with_fb_test_removed"] = int(df["in_fb_test"].sum())
        r["overlap_with_tw_test_removed"] = int(df["in_tw_test"].sum())
        df = df[~df["key"].isin(all_test)]
        clean[name] = df
        report[name] = r

    in_train = clean["dev"]["key"].isin(set(clean["train"]["key"]))
    report["dev"]["overlap_with_train_removed"] = int(in_train.sum())
    clean["dev"] = clean["dev"][~in_train]

    clean["fb_test"] = raw["fb_test"]
    clean["tw_test"] = raw["tw_test"]

    # sanity: no id appears in two splits, no key crosses train/dev/test
    ids = {k: set(v["id"]) for k, v in clean.items()}
    names = list(ids)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (ids[a] & ids[b]), f"id overlap between {a} and {b}"
    tr_keys = set(clean["train"]["key"])
    assert not (tr_keys & set(clean["dev"]["key"])), "train/dev text leak"
    assert not (tr_keys & all_test), "train/test text leak"
    assert not (set(clean["dev"]["key"]) & all_test), "dev/test text leak"

    report["final_rows"] = {k: len(v) for k, v in clean.items()}
    report["final_offensive_share"] = {k: round(float(v["label"].mean()), 3) for k, v in clean.items()}

    for name, df in clean.items():
        (SPLITS / f"{name}_ids.txt").write_text("\n".join(df["id"]) + "\n", encoding="utf-8")
    hashes = {name: sha256(RAW / FILES[name]) for name in ("fb_test", "tw_test")}
    (SPLITS / "test_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (REPORTS / "dedupe_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
