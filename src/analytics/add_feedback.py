"""Layer B: add a feedback table (true dev labels) next to the replayed predictions.
It is keyed by the same salted hash the replay used, so no text is stored."""
import hashlib
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from preprocess import ROOT, load_frozen_split
from serve.text import clean_text

DB = ROOT / "logs" / "replay.db"
SALT = "replay-salt"          # must match replay.py, or the join finds nothing


def main() -> None:
    dev = load_frozen_split("dev")
    rows = [(hashlib.sha256((SALT + clean_text(t)).encode("utf-8")).hexdigest()[:16], int(y))
            for t, y in zip(dev["text"], dev["label"])]
    with sqlite3.connect(DB) as con:
        con.execute("CREATE TABLE IF NOT EXISTS feedback (text_hash TEXT PRIMARY KEY, true_label INTEGER NOT NULL)")
        con.execute("DELETE FROM feedback")
        con.executemany("INSERT OR IGNORE INTO feedback VALUES (?, ?)", rows)
        matched = con.execute("SELECT COUNT(*) FROM predictions p JOIN feedback f USING (text_hash)").fetchone()[0]
    print(f"feedback rows: {len(rows)}, predictions matched to a true label: {matched}")


if __name__ == "__main__":
    main()