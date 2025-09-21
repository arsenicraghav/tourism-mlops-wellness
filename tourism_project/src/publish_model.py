import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

def main():
    p = argparse.ArgumentParser(description="Publish trained model artifacts to a Hugging Face model repo (CI-safe).")
    p.add_argument("--model-repo", required=True, help="e.g. labhara/tourism-wellness-model")
    p.add_argument("--artifacts-dir", default="artifacts/model", help="Directory with model.joblib, preprocessor.joblib, meta.joblib, metrics.json")
    p.add_argument("--hf-token", required=True, help="Hugging Face access token with write access")
    p.add_argument("--private", action="store_true", help="Create the repo as private if it doesn't exist")
    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.exists():
        raise SystemExit(f"Artifacts dir not found: {artifacts_dir}")

    # Ensure model repo exists
    create_repo(
        repo_id=args.model_repo,
        token=args.hf_token,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    api = HfApi(token=args.hf_token)
    api.upload_folder(
        folder_path=str(artifacts_dir),
        repo_id=args.model_repo,
        repo_type="model",
        path_in_repo="/",
        commit_message=f"Publish model artifacts from CI: {artifacts_dir}",
    )

    print(f"✅ Published artifacts from '{artifacts_dir}' to model repo '{args.model_repo}'")

if __name__ == "__main__":
    main()
