"""
The only file that knows how YOUR trained models are stored.
sealed_test.py uses just two things from here: RUNS and load_run().

Check that it reproduces training BEFORE any test run:
    python src/test_adapter.py <run_name>
"""
import gc
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Default-config runs have no suffix (train_transformer.py only adds one for ablations).
RUNS = {
    "hingbert": [f"hingbert_lr1e-05_seed{s}" for s in (1, 2, 3)],
    "muril":    [f"muril_lr2e-05_seed{s}" for s in (1, 2, 3)],
    "xlmr":     [f"xlmr_lr1e-05_seed{s}" for s in (1, 2, 3)],
    "tfidf":    ["tfidf_baseline"],
}
HF_NAME = {  # tokenizer fallback if it was not saved next to the weights
    "hingbert": "l3cube-pune/hing-mbert-mixed",
    "muril": "google/muril-base-cased",
    "xlmr": "FacebookAI/xlm-roberta-base",
}


def _threshold(run):
    """Frozen dev threshold: the 'dev_threshold' metric of the finished MLflow run."""
    import mlflow
    from tracking import setup_mlflow
    setup_mlflow("transformers")                      # points MLflow at the local database
    ids = [e.experiment_id for e in mlflow.search_experiments()]
    name = "tfidf_baseline_BEST" if run == "tfidf_baseline" else run   # name used by train_baseline.py
    runs = mlflow.search_runs(experiment_ids=ids,
                              filter_string=f"tags.mlflow.runName = '{name}' and attributes.status = 'FINISHED'")
    if runs.empty:
        raise SystemExit(f"no finished MLflow run named {run}")
    row = runs.sort_values("start_time").iloc[-1]
    if "metrics.dev_threshold" not in row or np.isnan(row["metrics.dev_threshold"]):
        raise SystemExit(f"run {run} has no dev_threshold metric")
    return float(row["metrics.dev_threshold"])


def _model_dir(run):
    base = ROOT / "models" / run
    hits = sorted(base.rglob("config.json"), key=lambda p: len(p.parts))
    if not hits:
        raise SystemExit(f"no saved model weights found under {base}. "
                         "Only dev_probs.npy was saved? Then these runs must be re-trained with weights saved.")
    return hits[0].parent


def _transformer(run):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from preprocess import clean_text
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = _model_dir(run)
    try:
        tok = AutoTokenizer.from_pretrained(d)
    except Exception:
        tok = AutoTokenizer.from_pretrained(HF_NAME[run.split("_")[0]])
    model = AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval()
    m = re.search(r"len(\d+)", run)
    max_len = int(m.group(1)) if m else 128

    def predict(texts):
        cleaned = [clean_text(t) for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(cleaned), 64):
                enc = tok(cleaned[i:i + 64], truncation=True, max_length=max_len,
                          padding=True, return_tensors="pt").to(dev)
                logits = model(**enc).logits.float()
                p = torch.softmax(logits, -1)[:, 1] if logits.shape[-1] == 2 else torch.sigmoid(logits[:, 0])
                out.append(p.cpu().numpy())
        return np.concatenate(out)
    return predict


def _tfidf(run):
    import joblib
    from preprocess import clean_text
    cands = list((ROOT / "models" / run).rglob("*.joblib")) + list((ROOT / "models").glob(f"{run}*.joblib"))
    if not cands:
        raise SystemExit(f"no .joblib baseline model found for {run} under models/")
    pipe = joblib.load(cands[0])
    return lambda texts: pipe.predict_proba([clean_text(t) for t in texts])[:, 1]


def load_run(run):
    """-> (frozen dev threshold, predict(list[str]) -> P(aggressive))"""
    predict = _tfidf(run) if run.startswith("tfidf") else _transformer(run)
    return _threshold(run), predict


if __name__ == "__main__":      # sanity check against the dev probabilities saved at training time
    from preprocess import load_frozen_split
    run = sys.argv[1]
    thr, predict = load_run(run)
    dev = load_frozen_split("dev")
    p = predict(dev["text"].tolist())
    ref_path = ROOT / "models" / run / "dev_probs.npy"
    print(f"{run}: threshold {thr:.3f}, scored {len(p)} dev rows")
    if not ref_path.exists():
        from sklearn.metrics import f1_score
        f = f1_score(dev["label"], (p >= thr).astype(int), average="macro")
        sys.exit(f"no dev_probs.npy saved for this run. Dev macro-F1 at the frozen threshold = {f:.3f} "
                 "(baseline should be 0.751)")
    ref = np.load(ref_path)
    diff = np.abs(p - ref)
    agree = ((p >= thr) == (ref >= thr)).mean()
    print(f"max |diff| {diff.max():.4f}, mean {diff.mean():.5f}, same label at threshold {agree:.4f}")
    print("PASS" if (agree >= 0.995 and diff.mean() < 0.01) else "FAIL: preprocessing or model file does not match training")
