

# Wellness Tourism MLOps
End-to-end MLOps pipeline using GitHub Actions, Hugging Face (dataset + Space), 
MLflow for tracking, and Streamlit UI.

## Live Links
- **GitHub Repo:** https://github.com/arsenicraghav/tourism-mlops-wellness
- **HF Dataset:** https://huggingface.co/datasets/labhara/tourism-wellness-dataset
- **HF Model:**   https://huggingface.co/labhara/tourism-wellness-model
- **HF Space:**   https://huggingface.co/spaces/labhara/tourism-wellness-app

## Screenshots
- Repo folder structure
- Successful GitHub Actions run (green check)
- Running Streamlit app on the Space

## How to Reproduce
1. Add `HF_TOKEN` as a GitHub Actions secret.
2. Put `tourism_project/data/tourism.csv` in the repo and push to `main`.
3. Pipeline runs automatically (register → prep → train/tune → publish → deploy).
4. Open the Space and test predictions.

