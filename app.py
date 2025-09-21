import os
import joblib
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Wellness Package Purchase Prediction", page_icon="🧘")
st.title("🧘 Wellness Tourism — Purchase Prediction")

MODEL_REPO = os.getenv("HF_MODEL_REPO", "labhara/tourism-wellness-model")  # set via Space Variables if needed
TOKEN = os.getenv("HF_TOKEN")  # set via Space Secrets if the model repo is private

REQUIRED = ["preprocessor.joblib", "model.joblib", "meta.joblib"]
local_files = {}
for fname in REQUIRED:
    local_files[fname] = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=fname,
        repo_type="model",
        token=TOKEN,
    )

preproc = joblib.load(local_files["preprocessor.joblib"])
model = joblib.load(local_files["model.joblib"])
meta = joblib.load(local_files["meta.joblib"])

num_cols = meta.get("numeric_cols", [])
cat_cols = meta.get("categorical_cols", [])
all_cols = num_cols + cat_cols

st.subheader("Enter customer features")
with st.form("predict_form"):
    inputs = {}
    for c in num_cols:
        inputs[c] = st.number_input(c, value=0.0, step=1.0, format="%.4f")
    for c in cat_cols:
        inputs[c] = st.text_input(c, value="")
    submitted = st.form_submit_button("Predict")

if submitted:
    X = pd.DataFrame([inputs], columns=all_cols)
    Xt = preproc.transform(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(Xt)[0, 1])
    pred = int(model.predict(Xt)[0])
    st.success(f"Prediction: **{pred}** (1 = Will purchase, 0 = Will not)")
    if proba is not None:
        st.info(f"Confidence (probability of purchase): **{proba:.3f}**")
