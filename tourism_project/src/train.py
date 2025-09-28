# This script trains a RandomForest model with hyperparameter tuning,
# logs metrics/parameters to MLflow, saves artifacts locally,
# and optionally publishes the best model to Hugging Face Model Hub.

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from huggingface_hub import hf_hub_download, HfApi, create_repo
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Target column in the dataset
TARGET = "ProdTaken"

# Helper Functions

def infer_cols(df: pd.DataFrame):
    """Identify numeric and categorical features automatically (exclude target)."""
    num, cat = [], []
    for c in df.columns:
        if c == TARGET:
            continue
        (num if pd.api.types.is_numeric_dtype(df[c]) else cat).append(c)
    return sorted(num), sorted(cat)

def metrics_dict(y_true, y_pred, y_proba=None):
    """Compute evaluation metrics and return as a dictionary."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = float("nan")
    return out

def build_preprocessor(num_cols, cat_cols):
    """Build preprocessing pipeline with imputation + encoding."""
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

# Main Execution

def main():
    # CLI arguments
    p = argparse.ArgumentParser(
        description="Train RandomForest with hyperparam tuning, MLflow logging, and optional HF model publishing."
    )
    # Input dataset (splits stored on HF)
    p.add_argument("--dataset-repo", required=True, help="e.g. labhara/tourism-wellness-dataset")
    p.add_argument("--train-path-in-repo", default="splits/train.csv")
    p.add_argument("--test-path-in-repo",  default="splits/test.csv")
    p.add_argument("--hf-token", required=True)

    # Tracking + output dirs
    p.add_argument("--mlflow-uri", default="file:./mlruns")
    p.add_argument("--artifacts-dir", default="artifacts/model")

    # Search config
    p.add_argument("--cv", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)

    # Optional: publish model to HF hub
    p.add_argument("--publish-model-repo", default=None, help="e.g. labhara/tourism-wellness-model")
    p.add_argument("--publish-private", action="store_true", help="Make model repo private if publishing")

    args = p.parse_args()

    # ---- Step 1: Load train/test splits from HF dataset repo ----
    train_csv = hf_hub_download(
        repo_id=args.dataset_repo, filename=args.train_path_in_repo,
        repo_type="dataset", token=args.hf_token
    )
    test_csv = hf_hub_download(
        repo_id=args.dataset_repo, filename=args.test_path_in_repo,
        repo_type="dataset", token=args.hf_token
    )
    train_df, test_df = pd.read_csv(train_csv), pd.read_csv(test_csv)

    if TARGET not in train_df.columns or TARGET not in test_df.columns:
        raise SystemExit(f" Target '{TARGET}' missing from splits.")

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test,  y_test  = test_df.drop(columns=[TARGET]),  test_df[TARGET]

    # ---- Step 2: Build pipeline & hyperparam space ----
    num_cols, cat_cols = infer_cols(train_df)
    pre = build_preprocessor(num_cols, cat_cols)

    rf = RandomForestClassifier(
        random_state=args.random_state, n_jobs=-1, class_weight="balanced"
    )
    pipe = Pipeline([("pre", pre), ("clf", rf)])

    param_distributions = {
        "clf__n_estimators": [100, 150, 200, 300, 400, 500],
        "clf__max_depth": [None, 5, 10, 15, 20, 30, 50],
        "clf__min_samples_split": [2, 5, 10, 20],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", "log2", 0.5, 0.75, 1.0],
    }

    # ---- Step 3: Train + tune with MLflow tracking ----
    mlflow.set_tracking_uri(args.mlflow_uri)
    with mlflow.start_run(run_name="rf-randsearch"):
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            cv=args.cv,
            scoring="f1",
            n_jobs=-1,
            random_state=args.random_state,
            verbose=1,
        )
        search.fit(X_train, y_train)
        # --- log full tuning results + best params as artifacts ---
        out = Path(args.artifacts_dir); out.mkdir(parents=True, exist_ok=True)

        cv_df = pd.DataFrame(search.cv_results_)
        cv_csv = out / "cv_results.csv"
        cv_df.to_csv(cv_csv, index=False)
        mlflow.log_artifact(str(cv_csv), artifact_path="tuning")

        best_json = out / "best_params.json"
        best_json.write_text(json.dumps(search.best_params_, indent=2))
        mlflow.log_artifact(str(best_json), artifact_path="tuning")


        best = search.best_estimator_

        # ---- Step 4: Evaluate on test set ----
        yhat = best.predict(X_test)
        yproba = best.predict_proba(X_test)[:, 1] if hasattr(best, "predict_proba") else None
        m = metrics_dict(y_test, yhat, yproba)

        # Log hyperparams + metrics to MLflow
        mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
        mlflow.log_metrics(m)

        # ---- Step 5: Save artifacts locally ----
        out = Path(args.artifacts_dir)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(best, out / "model.joblib")

        meta = {
            "target": TARGET,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "cv": args.cv,
            "n_iter": args.n_iter,
            "random_state": args.random_state,
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        (out / "metrics.json").write_text(json.dumps(m, indent=2))

        print("Best params:", search.best_params_)
        print(" Test metrics:", m)
        print(f"Saved model and metadata under {out}")

    # ---- Step 6: Optionally publish model to HF Hub ----
    if args.publish_model_repo:
        create_repo(
            repo_id=args.publish_model_repo,
            token=args.hf_token,
            repo_type="model",
            private=args.publish_private,
            exist_ok=True,
        )
        api = HfApi(token=args.hf_token)
        api.upload_folder(
            folder_path=str(Path(args.artifacts_dir)),
            repo_id=args.publish_model_repo,
            repo_type="model",
            path_in_repo="/",
            commit_message=f"Publish best model from training job: {args.artifacts_dir}",
        )
        print(f" Published artifacts → model repo '{args.publish_model_repo}'")

if __name__ == "__main__":
    main()
