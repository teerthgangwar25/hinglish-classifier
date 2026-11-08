"""Decision D7: pick the decision threshold on DEV so that recall on the offensive class stays
at or above a floor, and macro-F1 is as high as possible under that constraint.

The threshold is chosen once on dev and then FROZEN. Test sets only ever see the frozen value.
(Dev scores at the tuned threshold are slightly optimistic, because dev picked the threshold.)
"""
import numpy as np
from sklearn.metrics import f1_score


def pick_threshold(y_true, p_offensive, recall_floor: float = 0.90) -> dict:
    y_true = np.asarray(y_true)
    p = np.asarray(p_offensive)
    n_pos = int((y_true == 1).sum())
    rows = []
    for t in np.round(np.arange(0.01, 1.00, 0.01), 2):
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        rows.append({
            "threshold": float(t),
            "macro_f1": float(f1_score(y_true, pred, average="macro")),
            "recall_offensive": tp / max(n_pos, 1),
            "precision_offensive": tp / max(tp + fp, 1),
            "recall_not_offensive": tn / max(int((y_true == 0).sum()), 1),
        })
    feasible = [r for r in rows if r["recall_offensive"] >= recall_floor]
    if feasible:
        # best macro-F1; on ties prefer the higher threshold (fewer comments flagged)
        best = max(feasible, key=lambda r: (round(r["macro_f1"], 6), r["threshold"]))
        return {**best, "floor_met": True, "recall_floor": recall_floor}
    best = max(rows, key=lambda r: (r["recall_offensive"], r["macro_f1"]))
    return {**best, "floor_met": False, "recall_floor": recall_floor}
