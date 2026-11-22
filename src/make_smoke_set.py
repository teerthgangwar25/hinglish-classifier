"""D9: build the CI smoke set from DEV only (the sealed test sets are never touched).

300 comments, stratified by the binary label, fixed seed. The reference label is what the
registered ONNX model predicts today; CI later checks that a new commit still agrees with it.
Output stays in data/ (git-ignored: licensed, abusive text).
"""
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from preprocess import ROOT, load_frozen_split
from serve.app import Predictor
from serve.text import clean_text

N, SEED = 300, 1


def main() -> None:
    dev = load_frozen_split("dev")
    # stratify keeps the ~83% / 17% class split inside the sample
    _, smoke = train_test_split(dev, test_size=N, stratify=dev["label"], random_state=SEED)

    pred = Predictor(ROOT / "serving" / "model", threads=4, bucket=8)
    probs = pred.predict_proba([clean_text(t) for t in smoke["text"]])

    out = pd.DataFrame({
        "id": smoke["id"].to_numpy(),
        "text": smoke["text"].to_numpy(),
        "label": smoke["label"].to_numpy(),
        "ref_label": (probs >= pred.threshold).astype(int),
        "ref_prob": probs.round(6),
    })
    folder = ROOT / "data" / "smoke"
    folder.mkdir(parents=True, exist_ok=True)
    out.to_csv(folder / "smoke_set.csv", index=False, encoding="utf-8")

    print(f"rows {len(out)}, offensive share {out['label'].mean():.3f}, "
          f"not-offensive rows {(out['label'] == 0).sum()}")
    print(f"macro-F1 of the registered model on this sample: "
          f"{f1_score(out['label'], out['ref_label'], average='macro'):.4f}")


if __name__ == "__main__":
    main()


  