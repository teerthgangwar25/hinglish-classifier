"""Phase 2 wrap-up: register the best run of one model config in the MLflow model registry.

Seed selection uses DEV only (highest dev macro-F1 at the D7 threshold). The test sets are not touched.
Stored with the model: the frozen D7 threshold (as a version tag) and the source run id.
Usage (repo root):  python src/register_best.py --model hingbert --lr 1e-05
View it: mlflow ui, then the 'Models' tab.   Needs ~1 GB of disk for the artifact copy.
"""
import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from preprocess import ROOT
from tracking import data_hash, git_commit, setup_mlflow

NAME = "hinglish-offensive-classifier"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hingbert")
    ap.add_argument("--lr", default="1e-05", help="exactly as shown in the summary table, e.g. 1e-05")
    a = ap.parse_args()

    setup_mlflow("transformers")
    runs = mlflow.search_runs(
        experiment_names=["transformers"],
        filter_string=f"attributes.status = 'FINISHED' and params.model_key = '{a.model}' and params.lr = '{a.lr}'",
    )
    if runs.empty:
        raise SystemExit(f"no finished runs for model={a.model} lr={a.lr}")
    runs = runs.sort_values("start_time").drop_duplicates("params.seed", keep="last")
    best = runs.sort_values("metrics.dev_macro_f1_tuned", ascending=False).iloc[0]
    run_name = best["tags.mlflow.runName"]
    model_dir = ROOT / "models" / run_name / "best"
    if not model_dir.exists():
        raise SystemExit(f"model folder not found: {model_dir}")
    print(f"best seed: {best['params.seed']} ({run_name}), dev tuned macro-F1 {best['metrics.dev_macro_f1_tuned']:.4f}, "
          f"threshold {best['metrics.dev_threshold']:.2f}")

    setup_mlflow("registry")
    with mlflow.start_run(run_name=f"register_{run_name}") as run:
        mlflow.log_params({"source_run_id": best["run_id"], "source_run_name": run_name, "model_id": best["params.model_id"],
                           "lr": a.lr, "seed": best["params.seed"]})
        mlflow.set_tags({"git_commit": git_commit(), "data_hash": data_hash()})
        mlflow.log_metrics({"dev_macro_f1_tuned": float(best["metrics.dev_macro_f1_tuned"]),
                            "dev_threshold": float(best["metrics.dev_threshold"])})
        mlflow.log_artifacts(str(model_dir), artifact_path="model")
        source = f"{run.info.artifact_uri}/model"
        run_id = run.info.run_id

    # MLflow 3 only accepts runs:/ URIs for models logged through a flavor, and this is a plain
    # Hugging Face folder, so we register the artifact location directly.
    client = MlflowClient()
    try:
        client.create_registered_model(NAME)
    except MlflowException:
        pass  # already exists: this becomes a new version
    mv = client.create_model_version(NAME, source=source, run_id=run_id)
    client.set_registered_model_alias(NAME, "champion", mv.version)
    client.set_model_version_tag(NAME, mv.version, "dev_threshold", f"{best['metrics.dev_threshold']:.2f}")
    client.set_model_version_tag(NAME, mv.version, "source_run", run_name)
    print(f"registered {NAME} version {mv.version} with alias 'champion'")


if __name__ == "__main__":
    main()
