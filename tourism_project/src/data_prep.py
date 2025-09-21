import argparse
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

TARGET = "ProdTaken"  # 0/1

def infer_cols(df: pd.DataFrame):
    numeric_cols, categorical_cols = [], []
    for c in df.columns:
        if c == TARGET:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)
    return sorted(numeric_cols), sorted(categorical_cols)

def make_preprocessor(numeric_cols, categorical_cols):
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

def main():
    parser = argparse.ArgumentParser(description="Download dataset from HF, clean/split, save artifacts (CI-safe).")
    parser.add_argument("--dataset-repo", required=True, help="e.g. labhara/tourism-wellness-dataset")
    parser.add_argument("--dataset-path-in-repo", default="data/tourism.csv", help="Path to CSV inside HF dataset repo")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Where to write split data & preprocessors")
    parser.add_argument("--hf-token", required=True, help="Hugging Face access token")
    args = parser.parse_args()

    # Download CSV from HF
    local_csv = hf_hub_download(
        repo_id=args.dataset_repo,
        filename=args.dataset_path_in_repo,
        repo_type="dataset",
        token=args.hf_token,
    )

    df = pd.read_csv(local_csv)
    if TARGET not in df.columns:
        raise SystemExit(f"Target column '{TARGET}' not found in dataset columns: {list(df.columns)}")

    # Basic cleaning
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[~df[TARGET].isna()].copy()
    df[TARGET] = df[TARGET].astype(int)

    # Split
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    numeric_cols, categorical_cols = infer_cols(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit preprocessing on train only
    preproc = make_preprocessor(numeric_cols, categorical_cols)
    preproc.fit(X_train)

    # Save artifacts
    art_dir = Path(args.artifacts_dir)
    (art_dir / "data").mkdir(parents=True, exist_ok=True)
    (art_dir / "preprocess").mkdir(parents=True, exist_ok=True)

    X_train.to_csv(art_dir / "data" / "X_train.csv", index=False)
    X_test.to_csv(art_dir / "data" / "X_test.csv", index=False)
    y_train.to_csv(art_dir / "data" / "y_train.csv", index=False)
    y_test.to_csv(art_dir / "data" / "y_test.csv", index=False)

    meta = {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "target": TARGET}
    joblib.dump(preproc, art_dir / "preprocess" / "preprocessor.joblib")
    joblib.dump(meta, art_dir / "preprocess" / "meta.joblib")

    print(f"✅ Data prep complete. Artifacts saved under '{art_dir}/'")

if __name__ == "__main__":
    main()
