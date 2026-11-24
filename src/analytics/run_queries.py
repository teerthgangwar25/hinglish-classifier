"""Run every query in sql/ against logs/replay.db and save each result to reports/sql/ (counts only)."""
import sqlite3
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "logs" / "replay.db"
SQL = ROOT / "sql"
OUT = ROOT / "reports" / "sql"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB) as con:
        for f in sorted(SQL.glob("*.sql")):
            df = pd.read_sql_query(f.read_text(encoding="utf-8"), con)
            md = f"# {f.stem}\n\n" + df.to_markdown(index=False) + "\n"
            (OUT / f"{f.stem}.md").write_text(md, encoding="utf-8")
            print(md)


if __name__ == "__main__":
    main()