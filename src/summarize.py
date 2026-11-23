"""Aggregate dev results from MLflow into a mean +/- std table (one row per model and learning rate).
Reads the local mlflow.db. Writes reports/dev_summary.md (numbers only, safe to commit).
Run from repo root:  python src/summarize.py
Test sets are NOT involved: everything here is train/dev."""
import mlflow
import pandas as pd

from preprocess import ROOT
from tracking import setup_mlflow

METRICS = ["dev_macro_f1_at_0.5", "dev_macro_f1_tuned", "dev_recall_offensive_tuned", "dev_recall_not_offensive_tuned"]


def main() -> None:
    setup_mlflow("ablations")        # creates the experiment if it does not exist yet
    setup_mlflow("transformers")
    runs = mlflow.search_runs(experiment_names=["transformers", "ablations"], filter_string="attributes.status = 'FINISHED'")
    if runs.empty:
        print("no finished runs yet")
        return
    for col, default in (("params.max_len", "128"), ("params.norm", "off"), ("params.script", "asis")):
        runs[col] = runs[col].fillna(default) if col in runs else default      # older runs predate these params
    keys = ["params.model_key", "params.lr", "params.max_len", "params.norm", "params.script"]
    runs = runs.sort_values("start_time").drop_duplicates(keys + ["params.seed"], keep="last")   # a re-run replaces the older one
    rows = []
    for (model, lr, max_len, norm, script), g in runs.groupby(keys):
        row = {"model": model, "lr": lr, "max_len": max_len, "norm": norm, "script": script,
               "seeds": ",".join(sorted(g["params.seed"])), "n": len(g)}
        for m in METRICS:
            v = g[f"metrics.{m}"].astype(float)
            row[m] = f"{v.mean():.3f}" + (f" ± {v.std(ddof=1):.3f}" if len(v) > 1 else "")
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["model", "lr", "max_len", "norm", "script"])

    base = mlflow.search_runs(experiment_names=["tfidf-baseline"], filter_string="tags.role = 'registered-baseline'")
    lines = ["# Dev results (mean ± std over seeds; std shown only when n > 1)", "",
             "Dev numbers at the tuned threshold are slightly optimistic: dev picked the threshold.", "",
             table.to_markdown(index=False), ""]
    if not base.empty:
        b = base.sort_values("start_time").iloc[-1]
        lines += [f"TF-IDF baseline: macro-F1 @0.5 = {b['metrics.dev_macro_f1']:.3f}, "
                  f"tuned = {b['metrics.dev_macro_f1_tuned']:.3f}"]
    text = "\n".join(lines)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "dev_summary.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
