"""Build serving/model/: everything the API needs, in one folder (no MLflow needed at run time).

    python src/serve/prepare_bundle.py                      # champion from the MLflow registry
    python src/serve/prepare_bundle.py --run R --threshold T

Needs models/onnx/<run>/model_fp32.onnx (python src/export_onnx.py) and the tokenizer in models/<run>/best.
Ends with a parity check: the API's own tokenizer and runtime must reproduce the transformers + ONNX scores.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))
ROOT = SRC.parent
REGISTRY_NAME = "hinglish-offensive-classifier"


def champion():
    from mlflow import MlflowClient
    from tracking import setup_mlflow
    setup_mlflow("registry")
    mv = MlflowClient().get_model_version_by_alias(REGISTRY_NAME, "champion")
    return mv.tags["source_run"], float(mv.tags["dev_threshold"]), f"{REGISTRY_NAME}:v{mv.version}-onnx-fp32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--out", default=str(ROOT / "serving" / "model"))
    a = ap.parse_args()
    if a.run and a.threshold is not None:
        run, thr, version = a.run, a.threshold, f"{a.run}-onnx-fp32"
    else:
        run, thr, version = champion()

    onnx_path = ROOT / "models" / "onnx" / run / "model_fp32.onnx"
    tok_dir = ROOT / "models" / run / "best"
    if not onnx_path.exists():
        sys.exit(f"missing {onnx_path}: run  python src/export_onnx.py  first")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx_path, out / "model.onnx")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if (tok_dir / name).exists():
            shutil.copy(tok_dir / name, out / name)
    if not (out / "tokenizer.json").exists():
        sys.exit(f"no tokenizer.json in {tok_dir}")

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    meta = {"model_run": run, "model_version": version, "format": "onnx-fp32", "threshold": thr, "max_len": 128,
            "model_sha256": hashlib.sha256((out / "model.onnx").read_bytes()).hexdigest()[:16], "git_commit": commit}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"bundle written to {out}\n{json.dumps(meta, indent=2)}")

    # parity check against the transformers + ONNX path used in export_onnx.py
    try:
        import numpy as np
        from export_onnx import make_ort_fn
        from preprocess import clean_text, load_frozen_split
        from serve.app import Predictor
        texts = load_frozen_split("dev")["text"].map(clean_text).tolist()[:200]
        ref = make_ort_fn(out / "model.onnx", tok_dir, 4)(texts)
        got = Predictor(out, 4).predict_proba(texts)
        diff = float(np.abs(ref - got).max())
        flips = int(((ref >= thr) != (got >= thr)).sum())
        print(f"parity on 200 dev comments: max |diff| {diff:.2e}, label flips {flips}  ->",
              "PASS" if diff < 1e-3 and flips == 0 else "FAIL: tokenizers/runtime disagree with transformers")
    except ImportError as e:
        print("parity check skipped:", e)


if __name__ == "__main__":
    main()
