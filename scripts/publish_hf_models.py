"""Publish the approved model artifacts to the dedicated Hugging Face repo."""

from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


REPO_ID = "dada8173/coffee-bean-classifier-models"
ROOT = Path(__file__).resolve().parents[1]

FILES = {
    "README.md": ROOT / "MODEL_CARD.md",
    "ethiopia_washed_custom_Noback_best_model.pth": ROOT
    / "models/ethiopia_washed_custom_Noback_best_model.pth",
    "ethiopia_washed_custom_Noback_best_model.json": ROOT
    / "models/ethiopia_washed_custom_Noback_best_model.json",
    "ethiopia_washed_metrics.json": ROOT / "models/ethiopia_washed_metrics.json",
    "ethiopia_washed_history.json": ROOT / "models/ethiopia_washed_history.json",
    "honduras_natural_custom_Noback_best_model.pth": ROOT
    / "models/honduras_natural_custom_Noback_best_model.pth",
    "honduras_natural_custom_Noback_best_model.json": ROOT
    / "models/honduras_natural_custom_Noback_best_model.json",
    "honduras_natural_metrics.json": ROOT / "models/honduras_natural_metrics.json",
    "honduras_natural_history.json": ROOT / "models/honduras_natural_history.json",
}


def main() -> None:
    missing = [str(path) for path in FILES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing approved artifacts: {missing}")

    api = HfApi()
    account = api.whoami().get("name")
    if account != "dada8173":
        raise RuntimeError(f"Refusing to publish as unexpected account: {account!r}")

    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="model",
        operations=[
            CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source)
            for destination, source in FILES.items()
        ],
        commit_message="Publish Ethiopia and Honduras model checkpoints",
    )
    print(commit.oid)


if __name__ == "__main__":
    main()
