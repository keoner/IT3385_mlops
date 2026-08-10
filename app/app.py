# streamlit run app/app.py
import streamlit as st
import pandas as pd

from pycaret.classification import load_model as load_clf, predict_model as predict_clf
from pycaret.regression import load_model as load_reg, predict_model as predict_reg
from pycaret.anomaly import load_model as load_anom, predict_model as predict_anom

import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "models"
SENSOR_COLS = ["temperature", "vibration", "humidity", "pressure", "energy_consumption"]
RANGES = {
    "temperature": (20.0, 120.0, 65.0),
    "vibration": (0.0, 10.0, 2.0),
    "humidity": (10.0, 90.0, 45.0),
    "pressure": (0.0, 200.0, 100.0),
    "energy_consumption": (0.0, 500.0, 150.0),
}

st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")

HF_REPO_ID = "wasdwasdwasd123/MLOPS"
HF_REPO_TYPE = "model"
MODEL_FILES = [
    "classification_cleaned.pkl",
    "regression_engineered.pkl",
    "anomaly_cleaned.pkl",
]

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname in MODEL_FILES:
        local_path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(local_path):
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                filename=fname,
            )
            import shutil
            shutil.copy(downloaded, local_path)

@st.cache_resource
def load_models():
    download_models()
    return {
        "classification_cleaned": load_clf(f"{MODEL_DIR}/classification_cleaned"),
        "regression_engineered": load_reg(f"{MODEL_DIR}/regression_engineered"),
        "anomaly_cleaned": load_anom(f"{MODEL_DIR}/anomaly_cleaned"),
    }

models = load_models()

st.sidebar.header("Sensor Input")

input_mode = st.sidebar.radio("Data source", ["Manual input", "Upload CSV"])

uploaded_df = None
if input_mode == "Manual input":
    values = {c: st.sidebar.slider(c.replace("_", " ").title(), *RANGES[c][:2], RANGES[c][2]) for c in SENSOR_COLS}
    machine_id = st.sidebar.number_input("Machine ID", min_value=1, max_value=50, value=1)
    timestamp = st.sidebar.text_input("Timestamp", value="2025-03-10 12:00:00")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        uploaded_df = pd.read_csv(uploaded_file)
        missing_cols = [c for c in SENSOR_COLS if c not in uploaded_df.columns]
        if missing_cols:
            st.sidebar.error(f"CSV is missing required columns: {missing_cols}")
            uploaded_df = None
        else:
            st.sidebar.success(f"Loaded {len(uploaded_df)} rows")
            st.sidebar.dataframe(uploaded_df.head(), use_container_width=True)

TASK_SUFFIX = {
    "classification": "cleaned",
    "regression": "engineered",
    "anomaly": "cleaned",
}

def add_engineered_features(df):
    """Compute real rolling stats per machine if machine_id/timestamp exist,
    otherwise fall back to trivial values like the manual-input path does."""
    df = df.copy()
    has_grouping = "machine_id" in df.columns and "timestamp" in df.columns
    if has_grouping:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["machine_id", "timestamp"])
        for c in SENSOR_COLS:
            grouped = df.groupby("machine_id")[c]
            df[f"{c}_roll_mean"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
            df[f"{c}_roll_std"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).std().fillna(0.0))
            df[f"{c}_diff"] = grouped.transform(lambda s: s.diff().fillna(0.0))
    else:
        for c in SENSOR_COLS:
            df[f"{c}_roll_mean"], df[f"{c}_roll_std"], df[f"{c}_diff"] = df[c], 0.0, 0.0
    return df

def build_input_row(suffix):
    """Single-row DataFrame for manual input mode."""
    row = dict(values)
    row["machine_id"] = machine_id
    row["timestamp"] = pd.to_datetime(timestamp)
    if suffix == "engineered":
        for c in SENSOR_COLS:
            row[f"{c}_roll_mean"], row[f"{c}_roll_std"], row[f"{c}_diff"] = values[c], 0.0, 0.0
    return pd.DataFrame([row])

def build_batch_df(suffix):
    """Multi-row DataFrame for CSV upload mode."""
    if suffix == "engineered":
        return add_engineered_features(uploaded_df)
    return uploaded_df.copy()

st.title("Predictive Maintenance Dashboard")

tab_class, tab_reg, tab_anom, tab_compare = st.tabs(["Classification", "Regression", "Anomaly", "Comparison"])

with tab_class:
    suffix = TASK_SUFFIX["classification"]
    st.caption(f"Using the {suffix} model (only version currently deployed).")
    ready = uploaded_df is not None if input_mode == "Upload CSV" else True
    if st.button("Predict maintenance requirement", disabled=not ready):
        if input_mode == "Manual input":
            input_df = build_input_row(suffix)
            result = predict_clf(models[f"classification_{suffix}"], data=input_df)
            label = int(result["prediction_label"].iloc[0])
            score = float(result["prediction_score"].iloc[0])
            st.write("Maintenance required" if label == 1 else "Normal operation")
            st.write(f"Confidence: {score * 100:.1f}%")
        else:
            batch_df = build_batch_df(suffix)
            result = predict_clf(models[f"classification_{suffix}"], data=batch_df)
            display_df = batch_df.copy()
            display_df["prediction"] = result["prediction_label"].map({1: "Maintenance required", 0: "Normal"})
            display_df["confidence"] = (result["prediction_score"] * 100).round(1)
            st.dataframe(display_df, use_container_width=True)
            st.download_button("Download results", display_df.to_csv(index=False), "classification_results.csv")
        st.caption("Precision is high but recall is around 45%, so some real cases may be missed.")

with tab_reg:
    suffix = TASK_SUFFIX["regression"]
    st.caption(f"Using the {suffix} model (only version currently deployed).")
    ready = uploaded_df is not None if input_mode == "Upload CSV" else True
    if st.button("Predict time to failure", disabled=not ready):
        if input_mode == "Manual input":
            input_df = build_input_row(suffix)
            result = predict_reg(models[f"regression_{suffix}"], data=input_df)
            minutes = float(result["prediction_label"].iloc[0])
            st.write(f"Predicted time to failure: {minutes:.0f} minutes")
        else:
            batch_df = build_batch_df(suffix)
            result = predict_reg(models[f"regression_{suffix}"], data=batch_df)
            display_df = batch_df.copy()
            display_df["predicted_minutes_to_failure"] = result["prediction_label"].round(0)
            st.dataframe(display_df, use_container_width=True)
            st.download_button("Download results", display_df.to_csv(index=False), "regression_results.csv")
        st.caption("R² was around 0.05-0.06 on this dataset, so treat this as a rough estimate.")

with tab_anom:
    suffix = TASK_SUFFIX["anomaly"]
    st.caption(f"Using the {suffix} model (only version currently deployed).")
    ready = uploaded_df is not None if input_mode == "Upload CSV" else True
    if st.button("Check for anomaly", disabled=not ready):
        if input_mode == "Manual input":
            input_df = build_input_row(suffix)
            result = predict_anom(models[f"anomaly_{suffix}"], data=input_df)
            flagged = int(result["Anomaly"].iloc[0])
            score = float(result["Anomaly_Score"].iloc[0])
            st.write("Anomaly detected" if flagged == 1 else "Normal pattern")
            st.write(f"Anomaly score: {score:.3f}")
        else:
            batch_df = build_batch_df(suffix)
            result = predict_anom(models[f"anomaly_{suffix}"], data=batch_df)
            display_df = batch_df.copy()
            display_df["anomaly"] = result["Anomaly"].map({1: "Anomaly", 0: "Normal"})
            display_df["anomaly_score"] = result["Anomaly_Score"].round(3)
            st.dataframe(display_df, use_container_width=True)
            st.download_button("Download results", display_df.to_csv(index=False), "anomaly_results.csv")
        st.caption("Trained without failure labels, so it flags statistically unusual readings only.")

with tab_compare:
    st.table(pd.DataFrame({
        "Task": ["Classification", "Regression", "Anomaly"],
        "Deployed version": ["Cleaned", "Engineered", "Cleaned"],
        "Metric": ["F1: 0.623", "R2: 0.054", "F1: 0.511"],
    }))
    st.caption("Only one version per task is deployed; add rows here once more models are uploaded to the HF repo.")