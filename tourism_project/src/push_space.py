import argparse
from huggingface_hub import HfApi, create_repo

def main():
    p = argparse.ArgumentParser(description="Push a folder into a Hugging Face Space repository (CI-safe).")
    p.add_argument("--space-id", required=True, help="e.g. <hf-username>/tourism-wellness-app")
    p.add_argument("--folder", required=True, help="Local folder to upload as the Space root bundle")
    p.add_argument("--hf-token", required=True, help="Hugging Face access token")
    p.add_argument("--private", action="store_true", help="Create the Space as private if it doesn't exist")
    p.add_argument("--sdk", default="streamlit", choices=["streamlit", "gradio", "static", "docker"], help="Space SDK")
    args = p.parse_args()

    # Ensure Space exists (or create it)
    create_repo(
        repo_id=args.space_id,
        token=args.hf_token,
        repo_type="space",
        private=args.private,
        exist_ok=True,
        space_sdk=args.sdk,
    )
    create_repo(
        repo_id="labhara/tourism-wellness-app",
        token=hf_token,
        repo_type="space",
        space_sdk="streamlit",
        exist_ok=True
    )

    api = HfApi(token=args.hf_token)
    # Upload folder to Space root
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.space_id,
        repo_type="space",
        path_in_repo="/",
        commit_message=f"Update space from CI: upload {args.folder}",
        allow_patterns=None,  # upload all
    )

    print(f"✅ Uploaded folder '{args.folder}' to Space '{args.space_id}'")

if __name__ == "__main__":
    main()
