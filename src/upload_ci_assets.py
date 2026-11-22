"""Upload the CI assets to the private HF repo. Run locally with a WRITE token in HF_TOKEN.

Uploads the ONNX bundle (serving/model) and the smoke set. Neither is ever committed to git.
"""
import os

from huggingface_hub import HfApi

from preprocess import ROOT

REPO = "teerth025/hinglish-classifier-ci"


def main() -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_folder(repo_id=REPO, repo_type="model",
                      folder_path=str(ROOT / "serving" / "model"), path_in_repo="model")
    api.upload_file(repo_id=REPO, repo_type="model",
                    path_or_fileobj=str(ROOT / "data" / "smoke" / "smoke_set.csv"),
                    path_in_repo="smoke/smoke_set.csv")
    print("files now in the repo:")
    for name in api.list_repo_files(REPO, repo_type="model"):
        print(" ", name)


if __name__ == "__main__":
    main()