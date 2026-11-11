"""
Sealed one-time test evaluation (TRAC-1 Hindi, binary: NAG=0, OAG+CAG=1).

Two stages, on purpose:
  score   -> runs every frozen model on the two test sets ONCE, saves the raw
             probabilities, writes reports/TEST_SEAL.json. Refuses to run again.
  report  -> turns the saved probabilities into tables, bootstrap CIs, McNemar.
             Safe to re-run any number of times: it never touches a model.

Nothing is tuned here. Each run's threshold is the one frozen on dev.

  python src/sealed_test.py score --dry-run     # full pipeline on DEV, no seal
  python src/sealed_test.py report --dry-run
  python src/sealed_test.py score --confirm     # the real, one-time run
  python src/sealed_test.py report
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")   # Windows console/files default to cp1252
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
SEAL = REPORTS / "TEST_SEAL.json"

# name -> (file, expected rows, expected aggressive share from the README)
TEST_SETS = {
    "fb": ("agr_hi_fb_gold.csv", 970, 0.799),   # Test 1, Facebook, in-domain
    "tw": ("agr_hi_tw_gold.csv", 1194, 0.704),  # Test 2, Twitter, out-of-domain
}
DRY_SETS = {"dev": ("agr_hi_dev.csv", 3001, 0.821)}

N_BOOT = 2000
BOOT_SEED = 20261111
ALPHA_BONF = 0.05 / 4   # 2 planned comparisons x 2 test sets


# ---------------------------------------------------------------- data
def load(fname):
    # These CSVs have NO header row: id, text, label3.
    df = pd.read_csv(RAW / fname, header=None, names=["id", "text", "label3"],
                     dtype=str, keep_default_na=False)
    df["y"] = (df["label3"] != "NAG").astype(int)
    return df


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_data(sets, run_verify):
    if run_verify:
        r = subprocess.run([sys.executable, str(ROOT / "src" / "verify_data.py")])
        if r.returncode != 0:
            sys.exit("ABORT: verify_data.py failed, the raw files are not what was frozen.")
    for name, (fname, n_exp, share_exp) in sets.items():
        df = load(fname)
        share = df["y"].mean()
        if len(df) != n_exp or abs(share - share_exp) > 0.002 or not df["id"].is_unique:
            sys.exit(f"ABORT: {fname} has {len(df)} rows, share {share:.3f}; "
                     f"expected {n_exp} rows, share {share_exp}. Header handling?")
        print(f"ok  {fname}: {len(df)} rows, aggressive share {share:.3f}")


# ---------------------------------------------------------------- stage 1
def stage_score(dry):
    sets = DRY_SETS if dry else TEST_SETS
    out_dir = REPORTS / ("dry_run" if dry else "test_scores")
    if not dry and SEAL.exists():
        sys.exit(f"REFUSED: {SEAL} exists. The test sets were already evaluated.\n"
                 "Use `report` to re-compute tables from the saved scores.")
    check_data(sets, run_verify=not dry)

    import test_adapter as ad
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {k: load(f) for k, (f, _, _) in sets.items()}
    scores = {k: {} for k in sets}
    thresholds = {}
    for model, runs in ad.RUNS.items():
        for run in runs:
            thr, predict = ad.load_run(run)
            thresholds[run] = float(thr)
            for k, df in frames.items():
                scores[k][run] = np.asarray(predict(df["text"].tolist()), dtype=float)
                assert scores[k][run].shape == (len(df),)
            print(f"scored {run} (threshold {thr:.3f})")

    for k, df in frames.items():
        np.savez_compressed(out_dir / f"{k}.npz", ids=df["id"].values, y=df["y"].values,
                            **{r: p for r, p in scores[k].items()})
    meta = {"time_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         capture_output=True, text=True).stdout.strip(),
            "files_sha256": {f: sha256(RAW / f) for f, _, _ in sets.values()},
            "thresholds": thresholds, "models": ad.RUNS}
    (out_dir / "seal.json" if dry else SEAL).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nScores saved.", "(dry run, nothing sealed)" if dry else "SEALED. Now run: report")


# ---------------------------------------------------------------- stats
def f1_pair(y, pred):
    """macro-F1, works on arrays of shape (..., n)."""
    fs = []
    for c in (0, 1):
        tp = ((y == c) & (pred == c)).sum(-1)
        fp = ((y != c) & (pred == c)).sum(-1)
        fn = ((y == c) & (pred != c)).sum(-1)
        d = 2 * tp + fp + fn
        fs.append(np.where(d > 0, 2 * tp / np.maximum(d, 1), 0.0))
    return (fs[0] + fs[1]) / 2


def recall(y, pred, c):
    m = (y == c)
    return ((pred == c) & m).sum(-1) / np.maximum(m.sum(-1), 1)


def mcnemar_exact(ok_a, ok_b):
    b = int((ok_a & ~ok_b).sum())   # A right, B wrong
    c = int((~ok_a & ok_b).sum())   # A wrong, B right
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return b, c, p


def stage_report(dry):
    sets = DRY_SETS if dry else TEST_SETS
    d = REPORTS / ("dry_run" if dry else "test_scores")
    meta = json.loads((d / "seal.json" if dry else SEAL).read_text(encoding="utf-8"))
    for f, h in meta["files_sha256"].items():     # files must be unchanged since scoring
        if sha256(RAW / f) != h:
            sys.exit(f"ABORT: {f} changed since the seal.")
    models = meta["models"]
    thr = meta["thresholds"]
    rng = np.random.default_rng(BOOT_SEED)
    lines = [("# DRY RUN on dev (not a result)" if dry else "# Test results (single sealed evaluation)"), "",
             f"Scored {meta['time_utc']} UTC at git commit `{meta['git_commit'][:8]}`. "
             "Thresholds were frozen on dev; nothing was tuned on test.", "",
             f"Bootstrap: {N_BOOT} resamples of the test set, percentile 95% intervals. "
             f"McNemar: exact, two-sided; 4 planned tests, so the Bonferroni level is {ALPHA_BONF:.4f}.", ""]
    result = {}

    for k in sets:
        z = np.load(d / f"{k}.npz", allow_pickle=True)
        y = z["y"]
        n = len(y)
        idx = rng.integers(0, n, size=(N_BOOT, n))   # same resamples for every model -> paired
        yb = y[idx]
        pred = {r: (z[r] >= thr[r]).astype(int) for m in models for r in models[m]}
        predb = {r: p[idx] for r, p in pred.items()}

        lines += [f"## {k.upper()} ({n} rows, aggressive share {y.mean():.3f})", "",
                  "| Model | Macro-F1 (mean ± sd over seeds) | 95% CI | Macro-F1 @0.5 | Recall aggr. | Recall not-aggr. | Recall floor 0.90 held? |",
                  "|:--|:--|:--|:--|:--|:--|:--|"]
        seedmean_boot, res = {}, {}
        for m, runs in models.items():
            f = np.array([f1_pair(y, pred[r]) for r in runs])
            f05 = np.array([f1_pair(y, (z[r] >= 0.5).astype(int)) for r in runs])
            ra = np.array([recall(y, pred[r], 1) for r in runs])
            rn = np.array([recall(y, pred[r], 0) for r in runs])
            bs = np.mean([f1_pair(yb, predb[r]) for r in runs], axis=0)   # seed-averaged, per resample
            seedmean_boot[m] = bs
            lo, hi = np.percentile(bs, [2.5, 97.5])
            sd = lambda a: f"{a.std(ddof=1):.3f}" if len(a) > 1 else "n/a"
            lines.append(f"| {m} | {f.mean():.3f} ± {sd(f)} | [{lo:.3f}, {hi:.3f}] | {f05.mean():.3f} | "
                         f"{ra.mean():.3f} | {rn.mean():.3f} | {'yes' if ra.mean() >= 0.90 else 'NO'} |")
            res[m] = {"macro_f1": f.mean(), "ci": [lo, hi], "per_seed": f.tolist()}
        lines.append("")

        lines += ["**Paired comparisons** (difference in seed-averaged macro-F1; McNemar per seed-matched pair)", "",
                  "| Comparison | Δ macro-F1 | 95% CI of Δ | CI excludes 0? | McNemar per seed: (b, c, p) |", "|:--|:--|:--|:--|:--|"]
        for a, b_ in [("hingbert", "muril"), ("hingbert", "tfidf")]:
            if a not in models or b_ not in models:
                continue
            delta = res[a]["macro_f1"] - res[b_]["macro_f1"]
            db = seedmean_boot[a] - seedmean_boot[b_]
            lo, hi = np.percentile(db, [2.5, 97.5])
            mc = []
            for i, ra_ in enumerate(models[a]):
                rb_ = models[b_][min(i, len(models[b_]) - 1)]   # baseline has a single run
                bb, cc, p = mcnemar_exact(pred[ra_] == y, pred[rb_] == y)
                mc.append(f"({bb}, {cc}, {p:.3f})")
            lines.append(f"| {a} − {b_} | {delta:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                         f"{'yes' if lo > 0 or hi < 0 else 'no'} | {'; '.join(mc)} |")
            res[f"{a}_minus_{b_}"] = {"delta": delta, "ci": [lo, hi]}
        lines += ["", "b = first model right and second wrong, c = the reverse. "
                  "The three seed pairs are not independent tests.", ""]
        result[k] = res

    out = REPORTS / ("dry_run_results.md" if dry else "test_results.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["score", "report"])
    ap.add_argument("--dry-run", action="store_true", help="use DEV data, write nothing sealed")
    ap.add_argument("--confirm", action="store_true", help="required for the real one-time score run")
    a = ap.parse_args()
    if a.stage == "score" and not a.dry_run and not a.confirm:
        sys.exit("The real run needs --confirm. Use --dry-run to test the pipeline on dev.")
    (stage_score if a.stage == "score" else stage_report)(a.dry_run)
