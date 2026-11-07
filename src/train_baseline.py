"""Phase 2 baseline: TF-IDF character n-grams + logistic regression, tracked in MLflow.
Uses train + dev ONLY. The two test sets are not touched here (they are evaluated once, at the end).
Run from repo root:  python src/train_baseline.py
View runs:           mlflow ui --backend-store-uri sqlite:///mlflow.db     then open http://127.0.0.1:5000
"""
import hashlib
import subprocess
import tempfile
from pathlib import Path

import joblib
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from evaluate import compute_metrics
from preprocess import ROOT, clean_text, load_frozen_split

C_GRID = [0.3, 1, 3, 10]


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def data_hash() -> str:
    f = ROOT / "data_checksums.json"
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else "missing"


def main() -> None:
    train, dev = load_frozen_split("train"), load_frozen_split("dev")
    x_train, y_train = train["text"].map(clean_text), train["label"]
    x_dev, y_dev = dev["text"].map(clean_text), dev["label"]

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    exp_name = "tfidf-baseline"
    if mlflow.get_experiment_by_name(exp_name) is None:
        # as_uri() percent-encodes spaces, so a folder like "Hinglish Offensive-Content Classifier" is safe
        mlflow.create_experiment(exp_name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(exp_name)

    best = (-1.0, None, None)
    for C in C_GRID:
        # char_wb = character n-grams inside word boundaries: copes with spelling variants
        # ("nahi" / "nhi" / "nahin" share many 2-5 grams), and works for Devanagari too.
        model = make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
            LogisticRegression(C=C, class_weight="balanced", max_iter=2000),
        )
        with mlflow.start_run(run_name=f"tfidf_char2-5_LR_C{C}"):
            model.fit(x_train, y_train)
            m = compute_metrics(y_dev, model.predict(x_dev))
            mlflow.log_params({"ngram_range": "2-5", "analyzer": "char_wb", "C": C,
                               "class_weight": "balanced", "train_rows": len(train), "dev_rows": len(dev)})
            mlflow.set_tags({"git_commit": git_commit(), "data_hash": data_hash(), "split": "official-dedup"})
            mlflow.log_metrics({f"dev_{k}": float(v) for k, v in m.items()})
            print(f"C={C:<4} dev macro-F1={m['macro_f1']:.4f}  acc={m['accuracy']:.4f}  "
                  f"R(not-off)={m['recall_not_offensive']:.3f}  R(off)={m['recall_offensive']:.3f}")
            if m["macro_f1"] > best[0]:
                best = (m["macro_f1"], C, model)

    score, C, model = best
    print(f"\nBest: C={C}, dev macro-F1={score:.4f}")
    out = ROOT / "models"
    out.mkdir(exist_ok=True)
    joblib.dump(model, out / "tfidf_baseline.joblib")      # models/ is git-ignored
    with mlflow.start_run(run_name="tfidf_baseline_BEST"):
        mlflow.log_params({"C": C, "ngram_range": "2-5"})
        mlflow.log_metric("dev_macro_f1", score)
        mlflow.set_tags({"git_commit": git_commit(), "data_hash": data_hash(), "role": "registered-baseline"})
        mlflow.log_artifact(str(out / "tfidf_baseline.joblib"))


if __name__ == "__main__":
    main()
