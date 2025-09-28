# This app loads the trained scikit-learn pipeline from Hugging Face Model Hub,
# collects user inputs for numeric & categorical features,
# and predicts the probability of purchasing the wellness tourism package.

import os
import json
import joblib
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

# Page Configuration
st.set_page_config(page_title="Wellness Package Purchase Prediction", page_icon="🧘")
st.title(" Wellness Tourism — Purchase Prediction")

# Load Artifacts from Hugging Face Model Repo
MODEL_REPO = os.getenv("HF_MODEL_REPO", "labhara/tourism-wellness-model")  # default repo, override in Space
TOKEN = os.getenv("HF_TOKEN")  # required if repo is private

try:
    # Download model + metadata from HF Hub
    model_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="model.joblib", repo_type="model", token=TOKEN
    )
    meta_path = hf_hub_download(
        repo_id=MODEL_REPO, filename="meta.json", repo_type="model", token=TOKEN
    )
except Exception as e:
    st.error(
        "Could not download model artifacts from Hugging Face Hub. "
        "Check HF_MODEL_REPO / HF_TOKEN or confirm files exist (model.joblib, meta.json)."
    )
    st.exception(e)
    st.stop()

# Load pipeline (preprocessing + classifier) and metadata
pipe = joblib.load(model_path)
with open(meta_path, "r") as f:
    meta = json.load(f)

num_cols = meta.get("numeric_cols", [])
cat_cols = meta.get("categorical_cols", [])
all_cols = num_cols + cat_cols

if not all_cols:
    st.warning("⚠️ No input schema found in meta.json (numeric_cols/categorical_cols).")
    st.stop()

# Input Form for User Features
st.subheader("Enter customer features")
with st.form("predict_form"):
    inputs = {}

    # Numeric fields
    for c in num_cols:
        inputs[c] = st.number_input(
            c, value=0.0, step=1.0, format="%.4f"
        )

    # Categorical fields
    for c in cat_cols:
        inputs[c] = st.text_input(c, value="")

    submitted = st.form_submit_button("Predict")

# Prediction and Output
if submitted:
    # Create single-row DataFrame with same column order as training
    X = pd.DataFrame([inputs], columns=all_cols)
    proba = None
    try:
        if hasattr(pipe, "predict_proba"):
            proba = float(pipe.predict_proba(X)[0, 1])
        pred = int(pipe.predict(X)[0])
    except Exception as e:
        st.error(" Prediction failed. Please check input types/columns.")
        st.exception(e)
    else:
        st.success(f" Prediction: **{pred}** (1 = Will purchase, 0 = Will not)")
        if proba is not None:
            st.info(f" Confidence (probability of purchase): **{proba:.3f}**")
