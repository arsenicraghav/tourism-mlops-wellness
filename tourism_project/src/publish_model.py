# This script uploads trained model artifacts (e.g., model.joblib, meta.json, metrics.json)
# to a Hugging Face Model Hub repository. It generates a README.md
# model card with license, tags, and usage info.
# Suitable for GitHub Actions or other pipelines.

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# Minimal required files for a valid model repo
REQUIRED_FILES = ["model.joblib"]

# Template for auto-generated README (Hugging Face model card format)
README_TMPL = """---
library_name: scikit-learn
license: {license_id}
tags:
{tags_yaml}
pipeline_tag: tabular-classification
---

# {title}

This repository contains a scikit-learn pipeline for predicting purchase of the Wellness Tourism Package.

## Files
- `model.joblib`: Full pipeline (preprocessing + classifier)
- `meta.json`: Column metadata and training config
- `metrics.json`: Evaluation metrics on the held-out test set
"""

# Helper Functions

def ensure_artifacts(artifacts_dir: Path):
    """Check that required artifacts (like model.joblib) exist."""
    missing = [f for f in REQUIRED_FILES if not (artifacts_dir / f).exists()]
    if missing:
        raise SystemExit(f" Artifacts missing in {artifacts_dir}: {missing}")

def maybe_write_readme(tmp_dir: Path, title: str, license_id: str, tags_csv: str, force: bool) -> Path | None:
    """
    Create a README.md under tmp_dir.
    Returns the path if created, else None.
    """
    tags = [t.strip() for t in (tags_csv or "").split(",") if t.strip()]
    tags_yaml = "\n".join([f"- {t}" for t in tags]) if tags else "- tabular\n- tourism\n"
    readme_path = tmp_dir / "README.md"
    content = README_TMPL.format(title=title, license_id=license_id or "apache-2.0", tags_yaml=tags_yaml)
    if force or not readme_path.exists():
        readme_path.write_text(content, encoding="utf-8")
        return readme_path
    return None

# Main Execution

def main():
    p = argparse.ArgumentParser(
        description="Publish trained model artifacts to a Hugging Face model repo (CI-safe)."
    )
    p.add_argument("--model-repo", required=True, help="e.g. labhara/tourism-wellness-model")
    p.add_argument("--artifacts-dir", default="artifacts/model",
                   help="Directory with model.joblib (+ optional meta.json, metrics.json)")
    p.add_argument("--hf-token", required=True, help="Hugging Face access token with write access")
    p.add_argument("--private", action="store_true", help="Create the repo as private if it doesn't exist")

    # Optional extras
    p.add_argument("--title", default="Wellness Tourism Purchase Model")
    p.add_argument("--license-id", default="apache-2.0")
    p.add_argument("--tags", default="tourism,wellness,classification")
    p.add_argument("--write-readme", action="store_true",
                   help="Generate and upload a simple README.md (model card)")
    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise SystemExit(f" Artifacts dir not found: {artifacts_dir}")

    # Ensure artifacts exist
    ensure_artifacts(artifacts_dir)

    # Ensure model repo exists on Hugging Face Hub
    create_repo(
        repo_id=args.model_repo,
        token=args.hf_token,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api = HfApi(token=args.hf_token)

    # If --write-readme is passed, create a temp folder with README + artifacts
    upload_root = artifacts_dir
    tmp_root = None
    if args.write_readme:
        tmp_root = Path(".hf_model_publish_tmp")

        # Clean up any old temp directory
        if tmp_root.exists():
            for pth in tmp_root.rglob("*"):
                try:
                    pth.unlink()
                except IsADirectoryError:
                    pass
            try:
                tmp_root.rmdir()
            except Exception:
                pass

        tmp_root.mkdir(parents=True, exist_ok=True)

        # Copy over required artifacts
        for f in artifacts_dir.iterdir():
            if f.is_file() and f.suffix in {".joblib", ".json"}:
                (tmp_root / f.name).write_bytes(f.read_bytes())

        # Write README.md
        maybe_write_readme(tmp_root, args.title, args.license_id, args.tags, force=True)
        upload_root = tmp_root

    # Commit message includes CI SHA if available
    sha = os.getenv("GITHUB_SHA", "")[:7]
    commit_msg = f"Publish model artifacts from CI {f'({sha})' if sha else ''}: {artifacts_dir}"

    # Upload files to Hugging Face model repo
    api.upload_folder(
        folder_path=str(upload_root),
        repo_id=args.model_repo,
        repo_type="model",
        path_in_repo="/",
        commit_message=commit_msg,
        allow_patterns=["*.joblib", "*.json", "README.md"],  # keep repo tidy
    )

    # Cleanup temp folder if used
    if tmp_root:
        for pth in sorted(tmp_root.rglob("*"), reverse=True):
            try:
                pth.unlink()
            except IsADirectoryError:
                try:
                    pth.rmdir()
                except Exception:
                    pass
        try:
            tmp_root.rmdir()
        except Exception:
            pass

    print(f"Published artifacts from '{artifacts_dir}' to model repo '{args.model_repo}'")

if __name__ == "__main__":
    main()
