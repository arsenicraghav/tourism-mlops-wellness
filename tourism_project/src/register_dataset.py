import argparse
import os
from huggingface_hub import HfApi, create_repo

def main():
    parser = argparse.ArgumentParser(description="Upload local CSV to a Hugging Face dataset repo (CI-safe).")
    parser.add_argument("--dataset-repo", required=True, help="e.g. labhara/tourism-wellness-dataset")
    parser.add_argument("--local-path", default="tourism_project/data/tourism.csv", help="Local CSV path")
    parser.add_argument("--path-in-repo", default="data/tourism.csv", help="Destination path inside the HF dataset repo")
    parser.add_argument("--private", action="store_true", help="Create the dataset repo as private (if it doesn't exist)")
    parser.add_argument("--hf-token", required=True, help="Hugging Face access token")
    args = parser.parse_args()

    if not os.path.exists(args.local_path):
        raise SystemExit(f"Local dataset not found at: {args.local_path}")

    # Ensure dataset repo exists (or create it)
    create_repo(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    api = HfApi(token=args.hf_token)
    api.upload_file(
        path_or_fileobj=args.local_path,
        path_in_repo=args.path_in_repo,
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message=f"Upload {args.local_path}",
    )

    print(f"✅ Uploaded '{args.local_path}' → '{args.dataset_repo}/{args.path_in_repo}'")

if __name__ == "__main__":
    main()
