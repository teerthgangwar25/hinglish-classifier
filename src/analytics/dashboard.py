"""Layer B dashboard. Every panel is one saved query from sql/.
Run:  streamlit run src/analytics/dashboard.py"""
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "logs" / "replay.db"
SQL = ROOT / "sql"

st.set_page_config(page_title="Hinglish classifier: prediction log", layout="wide")


@st.cache_data
def run(name: str) -> pd.DataFrame:
    """Read sql/<name>.sql and run it. Cached, so a re-run of the page does not hit the database again."""
    with sqlite3.connect(DB) as con:
        return pd.read_sql_query((SQL / f"{name}.sql").read_text(encoding="utf-8"), con)


st.title("Hinglish aggression classifier: prediction log")
st.caption("Replayed dev comments with random timestamps, not live traffic. "
           "Hashes and features only, no comment text. Latency comes from a laptop and is not a benchmark.")

lat = run("06_latency_percentiles").iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Predictions logged", f"{int(lat['n']):,}")
c2.metric("p50 latency (ms)", lat["p50_ms"])
c3.metric("p95 latency (ms)", lat["p95_ms"])
c4.metric("p99 latency (ms)", lat["p99_ms"])

left, right = st.columns(2)
with left:
    st.subheader("Daily flag rate")
    daily = run("01_daily_flag_rate").set_index("day")
    st.line_chart(daily[["flag_rate", "flag_rate_7d"]])
with right:
    st.subheader("Error rate by model score")
    st.bar_chart(run("03_confidence_vs_error").set_index("score_bucket")["error_rate"])

left, right = st.columns(2)
with left:
    st.subheader("Error by comment length")
    st.dataframe(run("04_error_by_length"), hide_index=True)
with right:
    st.subheader("Confusion counts by script")
    st.dataframe(run("08_confusion_by_script"), hide_index=True)

st.subheader("Flag rate and latency by script")
st.dataframe(run("02_by_script"), hide_index=True)

with st.expander("10 slowest requests (features only)"):
    st.dataframe(run("05_slowest_requests"), hide_index=True)
with st.expander("Biggest day-over-day changes in flag rate"):
    st.dataframe(run("07_biggest_daily_jumps"), hide_index=True)