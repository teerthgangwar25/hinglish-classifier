import numpy as np

from thresholds import pick_threshold


def test_floor_forces_a_lower_threshold():
    # 8 offensive, 2 not. At t=0.5 only 6/8 offensive are caught (recall 0.75).
    y = np.array([1] * 8 + [0] * 2)
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.52, 0.4, 0.3, 0.2, 0.1])
    r = pick_threshold(y, p, recall_floor=0.90)
    assert r["floor_met"] and r["recall_offensive"] >= 0.90
    assert r["threshold"] < 0.5


def test_loose_floor_picks_best_macro_f1():
    y = np.array([1] * 6 + [0] * 4)
    p = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.3, 0.2, 0.15, 0.1])
    r = pick_threshold(y, p, recall_floor=0.5)
    assert r["macro_f1"] == 1.0


def test_unreachable_floor_is_reported_not_hidden():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.0, 0.0, 0.0, 0.0])   # model never scores the offensive class
    r = pick_threshold(y, p, recall_floor=0.90)
    assert r["floor_met"] is False
