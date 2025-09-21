import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

README_FRONTMATTER = """---
title: Wellness Tourism App
sdk: {sdk}
app_file: app.py
---
"""

def ensure_readme_with_sdk(folder: Path, sdk: str) -> None:
    """Ensure README.md declares the Space SDK so creation works on older hub clients."""
    folder.mkdir(parents=True, exist_ok=True)
    readme = folder / "README.md"
    header = README_FRONTMATTER.format(sdk=sdk.strip().lower())
    if not readme.exists():
        readme.write_text(header, encoding="utf-8")
        return
    content = readme.read_text(encoding="utf-8")
    if not content.lstrip().startswith("---"):
        readme.write_text(header + "\n" + content, encoding="utf-8")

def repo_exists(api: HfApi, repo_id: str, repo_type: str) -> bool:
    try:
        api.repo_info(repo_id, repo_type=repo_type)
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
    p = argparse.ArgumentParser(description="Create/update a Hugging Face Space by uploading a local folder.")
    p.add_argument("--space-id", required=True, help="e.g. <user-or-org>/tourism-wellness-app")
    p.add_argument("--folder", required=True, help="Local folder to upload as the Space root")
    p.add_argument("--hf-token", required=True, help="HF token with write access")
    p.add_argument("--private", action="store_true", help="Create the Space as private if it doesn't exist")
    p.add_argument("--sdk", default="streamlit", help="Space SDK: streamlit | gradio | static | docker")
    args = p.parse_args()

    api = HfApi(token=args.hf_token)
    folder = Path(args.folder)
    sdk = args.sdk.strip().lower()

    # Make sure the bundle contains a README with SDK/front matter (works across hub versions)
    ensure_readme_with_sdk(folder, sdk)

    # Create Space if missing (avoid passing space_sdk for old clients)
    if not repo_exists(api, args.space_id, "space"):
        other = exists_as_other_type(api, args.space_id)
        if other:
            raise SystemExit(
                f"❌ '{args.space_id}' already exists as a {other}. Pick a different --space-id."
            )
        try:
            create_repo(
                repo_id=args.space_id,
                token=args.hf_token,
                repo_type="space",
                private=args.private,
                exist_ok=False,
            )
            print(f"✅ Created Space '{args.space_id}' (private={args.private})")
        except Exception as e:
            raise SystemExit(f"Failed to create Space '{args.space_id}': {e}")

    # Upload folder contents to the Space root
    try:
        api.upload_folder(
            folder_path=str(folder),
            repo_id=args.space_id,
            repo_type="space",
            path_in_repo="/",
            commit_message=f"Update space from CI: upload {folder}",
        )
        print(f"✅ Uploaded folder '{folder}' to Space '{args.space_id}'")
    except Exception as e:
        raise SystemExit(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
