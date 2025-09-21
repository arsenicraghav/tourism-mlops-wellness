import argparse
from huggingface_hub import HfApi, create_repo, SpaceSdk
from huggingface_hub.utils._errors import HfHubHTTPError

def resolve_sdk(s: str) -> SpaceSdk:
    s = (s or "").strip().lower()
    mapping = {
        "streamlit": SpaceSdk.STREAMLIT,
        "gradio": SpaceSdk.GRADIO,
        "static": SpaceSdk.STATIC,
        "docker": SpaceSdk.DOCKER,
    }
    if s not in mapping:
        raise SystemExit(f"Invalid --sdk '{s}'. Choose from: streamlit, gradio, static, docker.")
    return mapping[s]

def repo_type_of(api: HfApi, repo_id: str):
    try:
        api.space_info(repo_id); return "space"
    except Exception:
        pass
    try:
        api.dataset_info(repo_id); return "dataset"
    except Exception:
        pass
    try:
        api.model_info(repo_id); return "model"
    except Exception:
        pass
    return None

def main():
    p = argparse.ArgumentParser(description="Push a folder to a Hugging Face Space (CI-safe).")
    p.add_argument("--space-id", required=True, help="e.g. labhara/tourism-wellness-app")
    p.add_argument("--folder", required=True, help="Local folder to upload as the Space root")
    p.add_argument("--hf-token", required=True, help="Hugging Face access token (with write permission)")
    p.add_argument("--private", action="store_true", help="Create the Space as private if it doesn't exist")
    p.add_argument("--sdk", default="streamlit", help="Space SDK: streamlit | gradio | static | docker")
    args = p.parse_args()

    api = HfApi(token=args.hf_token)
    sdk_enum = resolve_sdk(args.sdk)

    existing_type = repo_type_of(api, args.space_id)

    if existing_type is None:
        try:
            create_repo(
                repo_id=args.space_id,
                token=args.hf_token,
                repo_type="space",
                private=args.private,
                exist_ok=False,
                space_sdk=sdk_enum,
            )
            print(f"✅ Created Space '{args.space_id}' (sdk={sdk_enum.value}, private={args.private})")
        except HfHubHTTPError as e:
            raise SystemExit(
                f"Failed to create Space '{args.space_id}'. "
                f"Check --sdk value and token permissions. Original error: {e}"
            )
    elif existing_type != "space":
        raise SystemExit(
            f"❌ Repo '{args.space_id}' already exists as a {existing_type}. "
            f"Choose a different --space-id (or delete/rename the existing repo)."
        )
    else:
        print(f"ℹ️ Space '{args.space_id}' already exists. Skipping creation.")

    try:
        api.upload_folder(
            folder_path=args.folder,
            repo_id=args.space_id,
            repo_type="space",
            path_in_repo="/",
            commit_message=f"Update space from CI: upload {args.folder}",
        )
        print(f"✅ Uploaded folder '{args.folder}' to Space '{args.space_id}'")
    except HfHubHTTPError as e:
        raise SystemExit(
            f"Upload failed. Ensure the folder exists and token has write access. Original error: {e}"
        )

if __name__ == "__main__":
    main()
