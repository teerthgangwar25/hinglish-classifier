"""
Phase 4 (decision D8): compare PyTorch, ONNX fp32 and ONNX int8 for the registered champion model.

    python src/export_onnx.py                      # champion read from the MLflow registry
    python src/export_onnx.py --run <run> --threshold 0.57     # or name it yourself

What it does
  1. exports models/onnx/<run>/model_fp32.onnx and quantises it to model_int8.onnx (dynamic int8)
  2. scores the frozen DEV split with all three formats, on CPU (the serving target)
  3. applies the D8 rule: adopt a faster format only if macro-F1 at the frozen threshold
     drops by no more than 0.010 compared with PyTorch
  4. writes reports/optimization.md (numbers only)
The test sets are not touched.
"""
import argparse
import os
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from preprocess import ROOT, clean_text, load_frozen_split

REGISTRY_NAME = "hinglish-offensive-classifier"
MAX_DROP = 0.010            # D8: largest acceptable macro-F1 drop
MAX_LEN = 128


def champion():
    """Source run and frozen dev threshold of the registered champion (set by register_best.py)."""
    from mlflow import MlflowClient
    from tracking import setup_mlflow
    setup_mlflow("registry")
    mv = MlflowClient().get_model_version_by_alias(REGISTRY_NAME, "champion")
    return mv.tags["source_run"], float(mv.tags["dev_threshold"])


# ---------------------------------------------------------------- export
def export_fp32(model_dir, out_path):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir)
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, attn_implementation="eager")
    except (TypeError, ValueError):
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.float().cpu().eval()

    class Logits(torch.nn.Module):           # the exporter wants a plain tensor, not a ModelOutput
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    enc = tok(["yeh ek test hai", "short"], padding=True, truncation=True, max_length=MAX_LEN,
              return_tensors="pt", return_token_type_ids=False)
    kw = dict(input_names=["input_ids", "attention_mask"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                            "logits": {0: "batch"}})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = (enc["input_ids"], enc["attention_mask"])
    try:
        torch.onnx.export(Logits(model), args, str(out_path), dynamo=False, **kw)   # classic exporter
    except TypeError:                                                                # older torch: no 'dynamo' flag
        torch.onnx.export(Logits(model), args, str(out_path), **kw)


def quantise_int8(fp32_path, int8_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)


# ---------------------------------------------------------------- scoring
def softmax_pos(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def make_torch_fn(model_dir, threads):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().cpu().eval()

    def fn(texts, bs=32):
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), bs):
                enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
                out.append(model(**enc).logits.float().numpy())
        return softmax_pos(np.concatenate(out))
    return fn


def make_ort_fn(onnx_path, model_dir, threads):
    import onnxruntime as ort
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def fn(texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=MAX_LEN,
                      return_tensors="np", return_token_type_ids=False)
            out.append(sess.run(["logits"], {"input_ids": enc["input_ids"].astype(np.int64),
                                             "attention_mask": enc["attention_mask"].astype(np.int64)})[0])
        return softmax_pos(np.concatenate(out))
    return fn


def latency(fn, texts, n=200, warm=10):
    """End-to-end (tokenise + model) per-request latency at batch size 1, then throughput at batch 32."""
    for t in texts[:warm]:
        fn([t])
    ms = []
    for t in texts[warm:warm + n]:
        t0 = time.perf_counter()
        fn([t])
        ms.append((time.perf_counter() - t0) * 1000)
    big = texts[:256]
    t0 = time.perf_counter()
    fn(big, bs=32)
    thr = len(big) / (time.perf_counter() - t0)
    return float(np.percentile(ms, 50)), float(np.percentile(ms, 95)), float(thr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for every format (kept equal)")
    ap.add_argument("--force", action="store_true", help="re-export even if the .onnx files exist")
    a = ap.parse_args()

    run, thr = (a.run, a.threshold) if a.run and a.threshold is not None else champion()
    model_dir = ROOT / "models" / run / "best"
    out_dir = ROOT / "models" / "onnx" / run
    fp32, int8 = out_dir / "model_fp32.onnx", out_dir / "model_int8.onnx"
    print(f"champion run: {run} | frozen dev threshold {thr:.2f}")

    if a.force or not fp32.exists():
        print("exporting ONNX fp32 ...")
        export_fp32(model_dir, fp32)
    if a.force or not int8.exists():
        print("quantising to int8 ...")
        quantise_int8(fp32, int8)

    dev = load_frozen_split("dev")
    texts = dev["text"].map(clean_text).tolist()
    y = dev["label"].to_numpy()
    fns = {"PyTorch fp32": make_torch_fn(model_dir, a.threads),
           "ONNX fp32": make_ort_fn(fp32, model_dir, a.threads),
           "ONNX int8": make_ort_fn(int8, model_dir, a.threads)}
    sizes = {"PyTorch fp32": next(iter(model_dir.glob("*.safetensors")), None) or next(iter(model_dir.glob("*.bin"))),
             "ONNX fp32": fp32, "ONNX int8": int8}

    rng = np.random.default_rng(1)
    sample = [texts[i] for i in rng.permutation(len(texts))[:300]]
    rows, ref_p = [], None
    for name, fn in fns.items():
        print(f"scoring dev with {name} ...")
        p = fn(texts)
        ref_p = p if ref_p is None else ref_p
        pred = (p >= thr).astype(int)
        p50, p95, tput = latency(fn, sample)
        rows.append({"format": name, "macro_f1": f1_score(y, pred, average="macro"),
                     "recall_aggr": float(((pred == 1) & (y == 1)).sum() / (y == 1).sum()),
                     "recall_not": float(((pred == 0) & (y == 0)).sum() / (y == 0).sum()),
                     "mean_abs_prob_diff": float(np.abs(p - ref_p).mean()),
                     "same_label_as_torch": float((pred == (ref_p >= thr)).mean()),
                     "size_mb": Path(sizes[name]).stat().st_size / 1e6, "p50_ms": p50, "p95_ms": p95, "per_s": tput})

    base = rows[0]["macro_f1"]
    for r in rows:
        r["drop"] = base - r["macro_f1"]
        r["passes"] = r["drop"] <= MAX_DROP + 1e-9
    ok = [r for r in rows[1:] if r["passes"]]
    chosen = min(ok, key=lambda r: r["p95_ms"]) if ok and min(r["p95_ms"] for r in ok) < rows[0]["p95_ms"] else rows[0]

    lines = [f"# Optimisation comparison (D8): {run}", "",
             f"Dev split, {len(texts)} comments, CPU, {a.threads} threads, frozen threshold {thr:.2f}. "
             f"CPU: {platform.processor() or platform.machine()} ({os.cpu_count()} logical cores). "
             f"Latency is end-to-end (tokenise + model) over 200 single requests; throughput at batch 32.", "",
             "| Format | Macro-F1 | Drop vs PyTorch | Recall aggr. | Recall not-aggr. | Same label as PyTorch | Mean abs prob diff | Size (MB) | p50 ms | p95 ms | Comments/s |",
             "|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|"]
    for r in rows:
        lines.append(f"| {r['format']} | {r['macro_f1']:.4f} | {r['drop']:+.4f} | {r['recall_aggr']:.3f} | {r['recall_not']:.3f} | "
                     f"{r['same_label_as_torch']:.4f} | {r['mean_abs_prob_diff']:.4f} | {r['size_mb']:.0f} | "
                     f"{r['p50_ms']:.1f} | {r['p95_ms']:.1f} | {r['per_s']:.1f} |")
    lines += ["", f"**Rule (D8):** adopt a faster format only if the macro-F1 drop is at most {MAX_DROP:.3f}.",
              f"**Decision:** {chosen['format']}."]
    out = ROOT / "reports" / "optimization.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n" + "\n".join(lines) + f"\n\nwrote {out}")


if __name__ == "__main__":
    main()
