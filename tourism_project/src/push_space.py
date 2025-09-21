import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils._errors import HfHubHTTPError

README_HEADER_TMPL = """---
title: Wellness Tourism App
sdk: {sdk}
app_file: app.py
---
"""

def ensure_space_readme(folder: Path, sdk: str) -> None:
    """Ensure README.md exists with a YAML front matter declaring the Space SDK."""
    folder.mkdir(parents=True, exist_ok=True)
    readme = folder / "README.md"
    header = README_HEADER_TMPL.format(sdk=sdk.strip().lower())
    if not readme.exists():
        readme.write_text(header, encoding="utf-8")
        return
    # If README exists but no front matter, prepend one.
    content = readme.read_text(encoding="utf-8")
    if not content.lstrip().startswith("---"):
        readme.write_text(header + "\n" + content, encoding="utf-8")
        return
    # If front matter exists, leave it as-is (assume it already declares sdk/app_file)

def repo_exists_as_space(api: HfApi, repo_id: str) -> bool:
    try:
        api.repo_info(repo_id, repo_type="space")
        return True
    except Exception:
        return False

def exists_as_other_type(api: HfApi, repo_id: str) -> str | None:
    for t in ("dataset", "model"):
        try:
            api.repo_info(repo_id, repo_type=t)
            return t
        except Exception:
            pass
    return None

def main():
    p = argparse.ArgumentParser(description="Push a folder to a Hugging Face Space (backward-compatible).")
    p.add_argument("--space-id", required=True, help="e.g. <user or org>/tourism-wellness-app")
    p.add_argument("--folder", required=True, help="Local folder to upload as the Space root")
    p.add_argument("--hf-token", required=True, help="Hugging Face access token with write permission")
    p.add_argument("--private", action="store_true", help="Create the Space as private if it doesn't exist")
    p.add_argument("--sdk", default="streamlit", help="Space SDK: streamlit | gradio | static | docker")
    args = p.parse_args()

    api = HfApi(token=args.hf_token)
    folder = Path(args.folder)
    sdk = args.sdk.strip().lower()

    # 0) Ensure README header so the hub knows the SDK even if we can't pass space_sdk
    ensure_space_readme(folder, sdk)

    # 1) Create the Space if needed (avoid SpaceSdk to keep compatibility)
    if not repo_exists_as_space(api, args.space_id):
        other = exists_as_other_type(api, args.space_id)
        if other:
            raise SystemExit(
                f"❌ '{args.space_id}' already exists as a {other}. "
                f"Choose a different --space-id or delete/rename the existing repo."
            )
        try:
            # Older hub versions may not support space_sdk kwarg; call without it.
            create_repo(
                repo_id=args.space_id,
                token=args.hf_token,
                repo_type="space",
                private=args.private,
                exist_ok=False,
            )
            print(f"✅ Created Space '{args.space_id}' (private={args.private})")
        except HfHubHTTPError as e:
            # If a 409 happens due to a race, continue; else surface error.
            if e.response is not None and e.response.status_code == 409:
                print(f"ℹ️ Space '{args.space_id}' already exists (race). Continuing…")
            else:
                raise

    # 2) Upload folder contents to the Space root
    try:
        api.upload_folder(
            folder_path=str(folder),
            repo_id=args.space_id,
            repo_type="space",
            path_in_repo="/",
            commit_message=f"Update space from CI: upload {folder}",
        )
        print(f"✅ Uploaded folder '{folder}' to Space '{args.space_id}'")
    except HfHubHTTPError as e:
        raise SystemExit(
            f"Upload failed. Ensure the folder exists and token has write access. Original error: {e}"
        )

if __name__ == "__main__":
    main()
