# This script uploads a local CSV dataset to a Hugging Face dataset repository.
# It can be called from the GitHub Actions pipelines.

import argparse
import os
from huggingface_hub import HfApi, create_repo

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Upload local CSV to a Hugging Face dataset repo (CI-safe)."
    )
    parser.add_argument("--dataset-repo", required=True,
                        help="Target HF dataset repo, e.g. labhara/tourism-wellness-dataset")
    parser.add_argument("--local-path", default="tourism_project/data/tourism.csv",
                        help="Path to the local CSV file")
    parser.add_argument("--path-in-repo", default="data/tourism.csv",
                        help="Destination path inside the HF dataset repo")
    parser.add_argument("--private", action="store_true",
                        help="Create the dataset repo as private (if it doesn't exist)")
    parser.add_argument("--hf-token", required=True,
                        help="Hugging Face access token")
    args = parser.parse_args()

    # Validate local dataset file exists
    if not os.path.exists(args.local_path):
        raise SystemExit(f"Local dataset not found at: {args.local_path}")

    # Ensure the dataset repository exists on Hugging Face (or create it if missing)
    create_repo(
        repo_id=args.dataset_repo,
        token=args.hf_token,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,   # don't fail if it already exists
    )

    # Initialize API client
    api = HfApi(token=args.hf_token)

    # Upload the dataset file to the specified path in the repo
    api.upload_file(
        path_or_fileobj=args.local_path,
        path_in_repo=args.path_in_repo,
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message=f"Upload {args.local_path}",
    )

    print(f"Uploaded '{args.local_path}' → '{args.dataset_repo}/{args.path_in_repo}'")

if __name__ == "__main__":
    main()
