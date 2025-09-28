# ===== Data Preparation Script =====
# This script automates dataset preparation for model training:
# 1. Download dataset from Hugging Face Hub
# 2. Clean dataset (drop duplicates, handle NaNs, drop optional columns)
# 3. Train-test split (stratified)
# 4. Fit preprocessing pipelines (numeric + categorical)
# 5. Save local artifacts (splits + preprocessing objects)
# 6. Upload combined splits back to Hugging Face dataset repo
#
# Usage example:
# python data_prep.py \
#   --dataset-repo labhara/tourism-wellness-dataset \
#   --dataset-path-in-repo data/tourism.csv \
#   --hf-token $HF_TOKEN \
#   --artifacts-dir artifacts \
#   --drop-cols "CustomerID,Gender"

import argparse
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

# Target column for classification
TARGET = "ProdTaken"

# -------------------------------
# Helper Functions
# -------------------------------

def infer_cols(df: pd.DataFrame):
    """Infer numeric and categorical feature columns (excluding the target)."""
    num, cat = [], []
    for c in df.columns:
        if c == TARGET:
            continue
        (num if pd.api.types.is_numeric_dtype(df[c]) else cat).append(c)
    return sorted(num), sorted(cat)

def make_preprocessor(numeric_cols, categorical_cols):
    """Build preprocessing pipelines for numeric + categorical features."""
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

# -------------------------------
# Main Execution
# -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from HF, clean/split, save artifacts, and upload splits to HF."
    )
    parser.add_argument("--dataset-repo", required=True, help="e.g. labhara/tourism-wellness-dataset")
    parser.add_argument("--dataset-path-in-repo", default="data/tourism.csv", help="CSV path inside the dataset repo")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Local artifacts root")
    parser.add_argument("--hf-token", required=True, help="Hugging Face access token")
    parser.add_argument("--drop-cols", default="", help="Comma-separated columns to drop (e.g. 'CustomerID,Gender')")
    args = parser.parse_args()

    # 1) Download source CSV from HF
    local_csv = hf_hub_download(
        repo_id=args.dataset_repo,
        filename=args.dataset_path_in_repo,
        repo_type="dataset",
        token=args.hf_token,
    )
    df = pd.read_csv(local_csv)

    if TARGET not in df.columns:
        raise SystemExit(f"❌ Target column '{TARGET}' not found in dataset. Columns={list(df.columns)}")

    # 2) Cleaning: drop dups, enforce target type, optional drop columns
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[~df[TARGET].isna()].copy()
    df[TARGET] = df[TARGET].astype(int)

    drops = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    drops = [c for c in drops if c in df.columns and c != TARGET]
    if drops:
        df = df.drop(columns=drops)

    # 3) Split into train/test
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    num_cols, cat_cols = infer_cols(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Fit preprocessor (on train only)
    preproc = make_preprocessor(num_cols, cat_cols)
    preproc.fit(X_train)

    # 5) Save local artifacts
    art = Path(args.artifacts_dir)
    (art / "data").mkdir(parents=True, exist_ok=True)
    (art / "preprocess").mkdir(parents=True, exist_ok=True)
    (art / "splits").mkdir(parents=True, exist_ok=True)

    # Save splits separately
    X_train.to_csv(art / "data" / "X_train.csv", index=False)
    X_test.to_csv(art / "data" / "X_test.csv", index=False)
    y_train.to_csv(art / "data" / "y_train.csv", index=False)
    y_test.to_csv(art / "data" / "y_test.csv", index=False)

    # Save combined splits (features + target)
    train_full = pd.concat([X_train, y_train], axis=1)
    test_full  = pd.concat([X_test, y_test], axis=1)
    train_full_path = art / "splits" / "train.csv"
    test_full_path  = art / "splits" / "test.csv"
    train_full.to_csv(train_full_path, index=False)
    test_full.to_csv(test_full_path, index=False)

    # Save preprocessing metadata
    meta = {"numeric_cols": num_cols, "categorical_cols": cat_cols, "target": TARGET}
    joblib.dump(preproc, art / "preprocess" / "preprocessor.joblib")
    joblib.dump(meta,    art / "preprocess" / "meta.joblib")

    # 6) Upload combined splits back to HF dataset repo
    api = HfApi(token=args.hf_token)
    api.upload_file(
        path_or_fileobj=str(train_full_path),
        path_in_repo="splits/train.csv",
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message="Add train split",
    )
    api.upload_file(
        path_or_fileobj=str(test_full_path),
        path_in_repo="splits/test.csv",
        repo_id=args.dataset_repo,
        repo_type="dataset",
        commit_message="Add test split",
    )

    print("✅ Data prep complete.")
    print(f"   Dropped columns: {drops if drops else 'None'}")
    print(f"   Local artifacts: {art}")
    print(f"   Uploaded to: {args.dataset_repo}/splits/")

if __name__ == "__main__":
    main()
