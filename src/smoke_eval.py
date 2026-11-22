"""D9 smoke gate. Runs in CI (and locally) against the frozen 300-comment dev sample.

Three gates, all must pass:
  1. label agreement with the reference labels saved from the registered model (catches broken
     preprocessing or a swapped model with no sampling noise)
  2. macro-F1 floor (a coarse safety net, set well below the registered 0.79)
  3. p95 latency of single predictions (model call only: tokenise + inference, no HTTP)

Never prints comment text. This repo is public, so CI logs are public too.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from serve.app import Predictor
from serve.text import clean_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--smoke-csv", required=True)
    ap.add_argument("--min-agreement", type=float, default=0.99)
    ap.add_argument("--min-macro-f1", type=float, default=0.72)
    ap.add_argument("--max-p95-ms", type=float, default=250.0)
    ap.add_argument("--latency-n", type=int, default=200)
    a = ap.parse_args()

    df = pd.read_csv(a.smoke_csv, encoding="utf-8", dtype={"id": str}, keep_default_na=False)
    texts = [clean_text(t) for t in df["text"]]
    pred = Predictor(Path(a.model_dir), threads=int(os.environ.get("ORT_THREADS", "4")), bucket=8)

    probs = pred.predict_proba(texts)
    labels = (probs >= pred.threshold).astype(int)
    agreement = float((labels == df["ref_label"].to_numpy()).mean())
    macro_f1 = float(f1_score(df["label"], labels, average="macro"))
    mean_prob_diff = float(np.abs(probs - df["ref_prob"].to_numpy()).mean())

    for t in texts[:5]:                       # warm-up: the first calls are slower
        pred.predict_proba([t])
    times = []
    for t in texts[:a.latency_n]:             # one text per call, like a /predict request
        t0 = time.perf_counter()
        pred.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    p95 = float(np.percentile(times, 95))

    gates = [
        ("label agreement", agreement, f">= {a.min_agreement}", agreement >= a.min_agreement),
        ("macro-F1 vs true labels", macro_f1, f">= {a.min_macro_f1}", macro_f1 >= a.min_macro_f1),
        ("p95 latency (ms)", p95, f"<= {a.max_p95_ms}", p95 <= a.max_p95_ms),
    ]
    lines = ["| Gate | Value | Limit | Result |", "|:--|--:|:--|:--|"]
    for name, value, limit, ok in gates:
        lines.append(f"| {name} | {value:.4f} | {limit} | {'PASS' if ok else 'FAIL'} |")
    lines.append(f"\nrows {len(df)}, mean abs prob diff vs reference {mean_prob_diff:.4f}, "
                 f"threshold {pred.threshold}, model {pred.meta['model_version']}")
    report = "\n".join(lines)
    print(report)

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:                               # shows the table on the run's summary page
        with open(summary, "a", encoding="utf-8") as f:
            f.write("## Smoke gate\n\n" + report + "\n")

    if not all(ok for *_, ok in gates):
        print("\nSMOKE GATE FAILED")
        sys.exit(1)
    print("\nsmoke gate passed")


if __name__ == "__main__":
    main()