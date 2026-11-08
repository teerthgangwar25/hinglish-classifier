"""Small MLflow helpers shared by the baseline and the transformer runs."""
import hashlib
import subprocess

import mlflow

from preprocess import ROOT


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def data_hash() -> str:
    f = ROOT / "data_checksums.json"
    return hashlib.sha256(f.read_bytes()).hexdigest()[:16] if f.exists() else "missing"


def setup_mlflow(exp_name: str) -> None:
    """Local SQLite store at <repo>/mlflow.db; artifacts under <repo>/mlruns.
    as_uri() percent-encodes spaces, so folder names with spaces are safe."""
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    if mlflow.get_experiment_by_name(exp_name) is None:
        mlflow.create_experiment(exp_name, artifact_location=(ROOT / "mlruns").as_uri())
    mlflow.set_experiment(exp_name)
