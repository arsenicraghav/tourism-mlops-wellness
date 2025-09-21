import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Wellness Package Purchase Prediction", page_icon="🧘")

st.title("🧘 Wellness Tourism — Purchase Prediction")

# locate artifacts (as pushed by CI)
MODEL_DIR = Path("artifacts") / "model"
PREPROC_FILE = MODEL_DIR / "preprocessor.joblib"
MODEL_FILE = MODEL_DIR / "model.joblib"
META_FILE = MODEL_DIR / "meta.joblib"

if not (PREPROC_FILE.exists() and MODEL_FILE.exists() and META_FILE.exists()):
    st.warning("Artifacts not found. Please run the CI pipeline to generate and push them.")
    st.stop()

preproc = joblib.load(PREPROC_FILE)
model = joblib.load(MODEL_FILE)
meta = joblib.load(META_FILE)
num_cols = meta.get("numeric_cols", [])
cat_cols = meta.get("categorical_cols", [])
all_cols = num_cols + cat_cols

st.subheader("Enter customer features")
with st.form("predict_form"):
    inputs = {}
    # simple typed inputs; you can tailor these to your schema and add selectboxes for categoricals
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
