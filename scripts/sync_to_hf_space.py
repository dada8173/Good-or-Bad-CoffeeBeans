"""Atomically publish the approved application files to Hugging Face Spaces."""

from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi


REPO_ID = "dada8173/coffee-bean-classifier"
ROOT = Path(__file__).resolve().parents[1]

EXACT_FILES = (
    ".dockerignore",
    "Dockerfile",
    "LICENSE",
    "requirements.txt",
    "app.py",
    "models/README.md",
)
DIRECTORIES = ("templates", "static", "samplePhoto")
MODEL_PATTERNS = (
    "*_best_model.json",
    "*_metrics.json",
    "*_history.json",
)
FORBIDDEN_PARTS = {
    ".env",
    ".git",
    "coffee_beans_data",
    "legacy",
    "scripts",
}
FORBIDDEN_SUFFIXES = {".ipynb", ".pth"}


def approved_files() -> dict[str, Path]:
    files = {path: ROOT / path for path in EXACT_FILES}
    files["README.md"] = ROOT / "SPACE_README.md"

    for directory in DIRECTORIES:
        for source in sorted((ROOT / directory).rglob("*")):
            if source.is_file():
                files[source.relative_to(ROOT).as_posix()] = source

    for pattern in MODEL_PATTERNS:
        for source in sorted((ROOT / "models").glob(pattern)):
            files[source.relative_to(ROOT).as_posix()] = source

    missing = [destination for destination, source in files.items() if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing approved Space files: {missing}")

    for destination in files:
        parts = set(Path(destination).parts)
        if parts & FORBIDDEN_PARTS or Path(destination).suffix.lower() in FORBIDDEN_SUFFIXES:
            raise RuntimeError(f"Refusing to upload forbidden path: {destination}")

    return files


def main() -> None:
    api = HfApi()
    account = api.whoami().get("name")
    if account != "dada8173":
        raise RuntimeError(f"Refusing to publish as unexpected account: {account!r}")

    files = approved_files()
    parent_commit = api.space_info(REPO_ID).sha
    commit = api.create_commit(
        repo_id=REPO_ID,
        repo_type="space",
        parent_commit=parent_commit,
        operations=[
            CommitOperationAdd(path_in_repo=destination, path_or_fileobj=source)
            for destination, source in files.items()
        ],
        commit_message="Sync application from GitHub",
    )
    print(commit.oid)


if __name__ == "__main__":
    main()
