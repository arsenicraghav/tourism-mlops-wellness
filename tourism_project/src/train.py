import argparse
import json
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier

def load_artifacts(artifacts_dir: Path):
    X_train = pd.read_csv(artifacts_dir / "data" / "X_train.csv")
    X_test  = pd.read_csv(artifacts_dir / "data" / "X_test.csv")
    y_train = pd.read_csv(artifacts_dir / "data" / "y_train.csv").squeeze("columns")
    y_test  = pd.read_csv(artifacts_dir / "data" / "y_test.csv").squeeze("columns")

    preproc = joblib.load(artifacts_dir / "preprocess" / "preprocessor.joblib")
    meta    = joblib.load(artifacts_dir / "preprocess" / "meta.joblib")
    return X_train, X_test, y_train, y_test, preproc, meta

def evaluate_model(model, Xt, yt):
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xt)[:, 1]
    yhat = model.predict(Xt)
    metrics = {
        "accuracy": float(accuracy_score(yt, yhat)),
        "precision": float(precision_score(yt, yhat, zero_division=0)),
        "recall": float(recall_score(yt, yhat, zero_division=0)),
        "f1": float(f1_score(yt, yhat, zero_division=0)),
    }
    if proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(yt, proba))
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics

def main():
    p = argparse.ArgumentParser(description="Train baseline models and save artifacts (CI-safe).")
    p.add_argument("--artifacts-dir", default="artifacts", help="Where data/preprocess are located")
    p.add_argument("--model-out-dir", default="artifacts/model", help="Where to save model artifacts")
    p.add_argument("--mlflow-uri", default="file:./mlruns", help="MLflow tracking URI (local file by default)")
    p.add_argument("--run-name", default="baseline", help="MLflow run name")
    args = p.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    model_out_dir = Path(args.model_out_dir)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_test, y_train, y_test, preproc, meta = load_artifacts(artifacts_dir)

    # transform features
    Xt_train = preproc.transform(X_train)
    Xt_test  = preproc.transform(X_test)

    # Two simple candidates
    candidates = {
        "logreg": LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None),
        "rf": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    }

    mlflow.set_tracking_uri(args.mlflow_uri)

    best_name, best_model, best_metrics = None, None, {"f1": -1.0}
    for name, model in candidates.items():
        with mlflow.start_run(run_name=f"{args.run_name}-{name}"):
            model.fit(Xt_train, y_train)
            metrics = evaluate_model(model, Xt_test, y_test)
            # log params/metrics
            if hasattr(model, "get_params"):
                mlflow.log_params({f"{name}_{k}": v for k, v in model.get_params().items()})
            mlflow.log_metrics({f"{name}_{k}": v for k, v in metrics.items()})

            if metrics.get("f1", -1) > best_metrics.get("f1", -1):
                best_name, best_model, best_metrics = name, model, metrics

    # persist artifacts
    joblib.dump(best_model, model_out_dir / "model.joblib")
    joblib.dump(meta,       model_out_dir / "meta.joblib")  # convenience copy
    # (copy the preprocessor alongside model for inference)
    preproc_out = model_out_dir / "preprocessor.joblib"
    joblib.dump(joblib.load(artifacts_dir / "preprocess" / "preprocessor.joblib"), preproc_out)

    with open(model_out_dir / "metrics.json", "w") as f:
        json.dump({"model": best_name, **best_metrics}, f, indent=2)

    print(f"✅ Trained model: {best_name}")
    print(f"✅ Metrics: {best_metrics}")
    print(f"✅ Saved to: {model_out_dir}")

if __name__ == "__main__":
    main()
