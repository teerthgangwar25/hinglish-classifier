"""Phase 2: fine-tune one transformer with the Hugging Face Trainer (decision D5, step 1).

Uses train + dev ONLY. The two test sets are not touched here.
Examples (repo root, venv active):
    python src/train_transformer.py --model hingbert --smoke      # 1-minute wiring check, run this first
    python src/train_transformer.py --model hingbert --seed 1
    python src/train_transformer.py --model xlmr --seed 1         # defaults to batch 8 x accumulation 2
Out-of-memory? Lower --batch and raise --grad_accum (effective batch = batch x grad_accum),
lower --max_len, or add --grad_ckpt.
"""
import argparse
import inspect

import mlflow
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)
import transformers

from evaluate import compute_metrics
from preprocess import ROOT, clean_text, load_frozen_split
from thresholds import pick_threshold
from tracking import data_hash, git_commit, setup_mlflow

# D6 / D6b. batch x accum = effective batch 16 for all three.
MODELS = {
    "hingbert": dict(id="l3cube-pune/hing-mbert-mixed", batch=16, accum=1),
    "muril": dict(id="google/muril-base-cased", batch=16, accum=1),
    "xlmr": dict(id="FacebookAI/xlm-roberta-base", batch=8, accum=2),
}
RECALL_FLOOR = 0.90  # D7


class TextDataset(Dataset):
    """Pre-tokenised texts. Padding happens per batch in the collator (faster than padding everything)."""

    def __init__(self, enc, labels):
        self.enc, self.labels = enc, list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


class WeightedTrainer(Trainer):
    """Same as Trainer, but the loss is class-weighted (the data is ~82% offensive).
    This mirrors class_weight='balanced' in the TF-IDF baseline, so the comparison stays fair."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits.float()
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=self.class_weights.to(logits.device))
        return (loss, outputs) if return_outputs else loss


def hf_metrics(eval_pred):
    preds = eval_pred.predictions
    if isinstance(preds, tuple):
        preds = preds[0]
    m = compute_metrics(eval_pred.label_ids, np.argmax(preds, axis=-1))
    return {k: float(v) for k, v in m.items()}


def softmax_pos(logits) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(axis=1, keepdims=True))[:, 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--grad_accum", type=int)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subset, 1 epoch, logged to a separate 'smoke' experiment")
    a = ap.parse_args()

    cfg = MODELS[a.model]
    batch = a.batch or cfg["batch"]
    accum = a.grad_accum or cfg["accum"]
    set_seed(a.seed)

    train, dev = load_frozen_split("train"), load_frozen_split("dev")
    if a.smoke:
        train, dev, a.epochs = train.sample(256, random_state=a.seed), dev.sample(128, random_state=a.seed), 1
    x_train, y_train = train["text"].map(clean_text).tolist(), train["label"].to_numpy()
    x_dev, y_dev = dev["text"].map(clean_text).tolist(), dev["label"].to_numpy()

    tok = AutoTokenizer.from_pretrained(cfg["id"])
    lens = np.array([len(i) for i in tok(x_train, truncation=False)["input_ids"]])
    print(f"token length p50/p90/p99: {int(np.percentile(lens, 50))}/{int(np.percentile(lens, 90))}/"
          f"{int(np.percentile(lens, 99))} | share cut at max_len={a.max_len}: {(lens > a.max_len).mean():.1%}")
    ds_train = TextDataset(tok(x_train, truncation=True, max_length=a.max_len), y_train)
    ds_dev = TextDataset(tok(x_dev, truncation=True, max_length=a.max_len), y_dev)

    model = AutoModelForSequenceClassification.from_pretrained(cfg["id"], num_labels=2)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg['id']}: {n_params / 1e6:.0f}M params, vocab {model.config.vocab_size}")
    # (mBERT's vocab is 119547: a quick check that hing-mbert-mixed really is mBERT-based)

    weights = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train), dtype=torch.float32)
    on_gpu = torch.cuda.is_available()
    use_bf16 = on_gpu and torch.cuda.is_bf16_supported()
    name = f"{a.model}_seed{a.seed}" + ("_smoke" if a.smoke else "")
    out = ROOT / "models" / name  # git-ignored

    ta = inspect.signature(TrainingArguments.__init__).parameters  # tolerate renamed arguments across versions
    kw = {("eval_strategy" if "eval_strategy" in ta else "evaluation_strategy"): "epoch"}
    kw.update({"warmup_ratio": 0.1} if "warmup_ratio" in ta else {"warmup_steps": 0.1})
    args = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=batch, per_device_eval_batch_size=64,
        gradient_accumulation_steps=accum, learning_rate=a.lr, num_train_epochs=a.epochs,
        weight_decay=0.01, save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="macro_f1", greater_is_better=True, bf16=use_bf16,
        fp16=(on_gpu and not use_bf16), gradient_checkpointing=a.grad_ckpt, seed=a.seed, data_seed=a.seed,
        logging_steps=50, report_to="none", dataloader_num_workers=0, **kw,
    )
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=ds_train, eval_dataset=ds_dev,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=hf_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], class_weights=weights,
    )

    setup_mlflow("smoke" if a.smoke else "transformers")
    with mlflow.start_run(run_name=name):
        mlflow.log_params({
            "model_key": a.model, "model_id": cfg["id"], "seed": a.seed, "epochs_max": a.epochs, "lr": a.lr,
            "max_len": a.max_len, "batch": batch, "grad_accum": accum, "effective_batch": batch * accum,
            "class_weights": [round(float(w), 3) for w in weights], "bf16": use_bf16, "params_m": round(n_params / 1e6),
            "train_rows": len(train), "dev_rows": len(dev), "recall_floor": RECALL_FLOOR,
        })
        mlflow.set_tags({
            "git_commit": git_commit(), "data_hash": data_hash(), "split": "official-dedup",
            "gpu": torch.cuda.get_device_name(0) if on_gpu else "cpu", "torch": torch.__version__,
            "transformers": transformers.__version__,
        })
        trainer.train()

        # per-epoch dev metrics from the Trainer's log
        for rec in trainer.state.log_history:
            if "eval_macro_f1" in rec:
                mlflow.log_metrics({"dev_macro_f1": rec["eval_macro_f1"], "dev_loss": rec["eval_loss"]}, step=int(rec["epoch"]))

        # best epoch (restored by load_best_model_at_end): dev probabilities, then the D7 threshold
        logits = trainer.predict(ds_dev).predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        p_dev = softmax_pos(logits)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "dev_probs.npy", p_dev)
        at_half = compute_metrics(y_dev, (p_dev >= 0.5).astype(int))
        thr = pick_threshold(y_dev, p_dev, RECALL_FLOOR)
        mlflow.log_metrics({
            "dev_macro_f1_at_0.5": at_half["macro_f1"], "dev_threshold": thr["threshold"],
            "dev_macro_f1_tuned": thr["macro_f1"], "dev_recall_offensive_tuned": thr["recall_offensive"],
            "dev_recall_not_offensive_tuned": thr["recall_not_offensive"], "floor_met": float(thr["floor_met"]),
            "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if on_gpu else 0.0,
        })
        mlflow.log_artifact(str(out / "dev_probs.npy"))
        trainer.save_model(str(out / "best"))
        tok.save_pretrained(str(out / "best"))
        print(f"\n{name}: dev macro-F1 @0.5 = {at_half['macro_f1']:.4f} | D7 threshold {thr['threshold']:.2f} -> "
              f"macro-F1 {thr['macro_f1']:.4f}, R(off) {thr['recall_offensive']:.3f}, "
              f"R(not-off) {thr['recall_not_offensive']:.3f}, floor_met={thr['floor_met']}")


if __name__ == "__main__":
    main()
