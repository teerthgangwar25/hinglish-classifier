"""Phase 3: where does the model break? DEV only (the test sets stay sealed until the final evaluation).

    python src/error_analysis.py --run hingbert_lr1e-05_seed1

Needs models/<run>/dev_probs.npy (saved by train_transformer.py) and the frozen D7 threshold from MLflow.
Optional: splits/langmix_dev.csv (python src/langmix.py --split dev) adds the language-mix slices.
Writes
  reports/error_analysis_<run>.md      slice tables, counts only, safe to commit
  data/error_review/<run>.csv          raw text of ~100 errors to read by hand. GIT-IGNORED: abusive text, licensed data.
"""
import argparse

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from langmix import script_class
from preprocess import ROOT, clean_text, load_frozen_split
from tracking import setup_mlflow


def slice_metrics(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(col, dropna=False):
        pos, neg = g[g.label == 1], g[g.label == 0]
        rows.append({
            col: key, "n": len(g), "offensive_share": round(g.label.mean(), 3),
            "macro_f1": round(f1_score(g.label, g.pred, average="macro"), 3) if min(len(pos), len(neg)) >= 20 else np.nan,
            "recall_offensive": round((pos.pred == 1).mean(), 3) if len(pos) else np.nan,
            "recall_not_offensive": round((neg.pred == 0).mean(), 3) if len(neg) else np.nan,
            "error_rate": round((g.pred != g.label).mean(), 3),
        })
    return pd.DataFrame(rows)


def length_bucket(text: str) -> str:
    n = len(text.split())
    return "short (<=10 words)" if n <= 10 else ("medium (11-40)" if n <= 40 else "long (>40)")


def get_threshold(run_name: str) -> float:
    setup_mlflow("transformers")
    runs = mlflow.search_runs(experiment_names=["transformers"],
                              filter_string=f"tags.mlflow.runName = '{run_name}' and attributes.status = 'FINISHED'")
    if runs.empty:
        raise SystemExit(f"no finished MLflow run named {run_name}")
    return float(runs.sort_values("start_time").iloc[-1]["metrics.dev_threshold"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--k", type=int, default=25, help="errors per type: k most confident + k random")
    a = ap.parse_args()

    dev = load_frozen_split("dev")
    probs = np.load(ROOT / "models" / a.run / "dev_probs.npy")
    assert len(probs) == len(dev), "dev_probs.npy does not match the frozen dev split"
    thr = get_threshold(a.run)
    dev["prob"], dev["pred"] = probs, (probs >= thr).astype(int)
    dev["clean"] = dev["text"].map(clean_text)
    dev["script"] = dev["clean"].map(script_class)
    dev["length"] = dev["text"].map(length_bucket)
    lm_path = ROOT / "splits" / "langmix_dev.csv"
    have_mix = lm_path.exists()
    if have_mix:
        dev = dev.merge(pd.read_csv(lm_path)[["id", "mix_bucket"]], on="id", how="left")

    overall = {"threshold": thr, "macro_f1": round(float(f1_score(dev.label, dev.pred, average="macro")), 3),
               "recall_offensive": round(float((dev[dev.label == 1].pred == 1).mean()), 3),
               "recall_not_offensive": round(float((dev[dev.label == 0].pred == 0).mean()), 3)}
    flagged = (dev.groupby("label3")["pred"].agg(n="size", flagged_offensive="mean").round(3).reset_index())

    parts = [f"# Error analysis on dev: {a.run}", "", f"Overall: {overall}", "",
             "## By original 3-class label (share flagged offensive; high is good for OAG/CAG, bad for NAG)", "",
             flagged.to_markdown(index=False), "", "## By script", "", slice_metrics(dev, "script").to_markdown(index=False),
             "", "## By length", "", slice_metrics(dev, "length").to_markdown(index=False)]
    if have_mix:
        parts += ["", "## By language mix (HingBERT-LID)", "", slice_metrics(dev, "mix_bucket").to_markdown(index=False)]
    else:
        parts += ["", "(no language-mix slice yet: run `python src/langmix.py --split dev`)"]
    text = "\n".join(parts)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / f"error_analysis_{a.run}.md").write_text(text, encoding="utf-8")
    print(text)

    # errors to read by hand: per error type, the k most confident mistakes plus k random others
    rng = np.random.default_rng(1)
    picks = []
    for etype, mask, dist in [("missed_aggressive (FN)", (dev.label == 1) & (dev.pred == 0), thr - dev.prob),
                              ("false_alarm (FP)", (dev.label == 0) & (dev.pred == 1), dev.prob - thr)]:
        errs = dev[mask].assign(error_type=etype, dist=dist[mask]).sort_values("dist", ascending=False)
        top, rest = errs.head(a.k), errs.iloc[a.k:]
        rand = rest.sample(min(a.k, len(rest)), random_state=int(rng.integers(1e6))) if len(rest) else rest
        picks += [top.assign(sample="most_confident"), rand.assign(sample="random")]
        print(f"{etype}: {int(mask.sum())} errors on dev")
    cols = ["id", "error_type", "sample", "label3", "prob", "script", "text"] + (["mix_bucket"] if have_mix else [])
    review = pd.concat(picks)[cols].assign(category="", notes="")
    out = ROOT / "data" / "error_review"
    out.mkdir(parents=True, exist_ok=True)
    review.to_csv(out / f"{a.run}.csv", index=False, encoding="utf-8-sig")   # BOM so Excel shows Devanagari correctly
    print(f"wrote {out / (a.run + '.csv')}  ({len(review)} rows to review)")


if __name__ == "__main__":
    main()
