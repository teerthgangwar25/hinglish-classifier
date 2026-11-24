"""Layer B: replay frozen DEV comments through the serving model and fill a prediction log.
The data is REPLAYED, not live traffic: timestamps are random, spread over 28 days.
Only hashes and features are stored, never the comment text."""
import hashlib
import random
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so `serve.*` and `preprocess` import

from preprocess import ROOT, load_frozen_split
from serve.app import SCHEMA, Predictor
from serve.text import clean_text, features

DB = ROOT / "logs" / "replay.db"
SALT = "replay-salt"
DAYS = 28
INSERT = ("INSERT INTO predictions (ts_utc, endpoint, batch_size, text_hash, n_chars, n_words, script, "
          "deva_share, has_hashtag, has_mention, has_url, has_emoji, label, score, threshold, "
          "model_version, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)")


def main() -> None:
    dev = load_frozen_split("dev")
    pred = Predictor(ROOT / "serving" / "model", threads=4, bucket=8)

    # random timestamps over the last DAYS days, sorted so id order follows time order
    rng = random.Random(1)
    end = datetime.now(timezone.utc)
    secs = sorted(rng.uniform(0, DAYS * 86400) for _ in range(len(dev)))
    stamps = [(end - timedelta(days=DAYS) + timedelta(seconds=s)).isoformat(timespec="seconds") for s in secs]

    rows = []
    for ts, text in zip(stamps, dev["text"]):
        t0 = time.perf_counter()
        prob = float(pred.predict_proba([clean_text(text)])[0])       # one comment per call, like /predict
        ms = (time.perf_counter() - t0) * 1000
        f = features(text)
        digest = hashlib.sha256((SALT + clean_text(text)).encode("utf-8")).hexdigest()[:16]
        rows.append((ts, "/predict", 1, digest, f["n_chars"], f["n_words"], f["script"], f["deva_share"],
                     f["has_hashtag"], f["has_mention"], f["has_url"], f["has_emoji"],
                     int(prob >= pred.threshold), prob, pred.threshold, pred.meta["model_version"], round(ms, 2)))

    DB.parent.mkdir(exist_ok=True)
    DB.unlink(missing_ok=True)                                         # start from a clean file
    with sqlite3.connect(DB) as con:
        con.executescript(SCHEMA)
        con.executemany(INSERT, rows)
    print(f"wrote {len(rows)} rows to {DB}")


if __name__ == "__main__":
    main()