<<<<<<< HEAD
import streamlit as st
import pandas as pd

MODEL_DIR = "model"

st.set_page_config(page_title="Employee Burnout Predictor", layout="wide")

NUMERIC_RANGES = {
    "work_hours_per_week": (20.0, 90.0, 47.0),
    "overtime_hours": (0.0, 40.0, 6.0),
    "job_satisfaction": (0.0, 10.0, 6.0),
    "manager_support": (0.0, 10.0, 6.0),
    "work_life_balance": (0.0, 10.0, 6.0),
    "sleep_hours": (0.0, 12.0, 6.5),
    "stress_level": (0.0, 10.0, 5.5),
    "anxiety_score": (0.0, 10.0, 5.0),
    "depression_score": (0.0, 10.0, 4.0),
}

REQUIRED_BATCH_COLS = list(NUMERIC_RANGES) + ["has_therapy", "seeks_professional_help"]


@st.cache_resource(max_entries=1)
def get_model(task):
    if task == "burnout_score":
        from pycaret.regression import load_model
        return load_model(f"{MODEL_DIR}/burnout_score_pipeline")
    elif task == "seeks_help":
        from pycaret.classification import load_model
        return load_model(f"{MODEL_DIR}/seeks_help_pipeline")


def burnout_band(score):
    if score <= 2:
        return "Low"
    if score <= 4:
        return "Moderate"
    return "High"


def yes_no(v):
    return 1 if v == "Yes" else 0


def to_binary(v):
    """Coerces a batch CSV cell to 0/1. Returns None if it's not recognizable,
    so the caller can flag exactly which rows are bad instead of guessing."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v in (0, 1):
        return int(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "y", "1", "true"):
            return 1
        if s in ("no", "n", "0", "false"):
            return 0
    return None


def validate_batch_df(df):
    """Checks columns exist, then checks every value is actually usable —
    catches things like a stray 'Twenty-Five' in a numeric column or a typo
    in a yes/no column before it ever reaches the model."""
    errors = []
    missing_cols = [c for c in REQUIRED_BATCH_COLS if c not in df.columns]
    if missing_cols:
        return None, [f"Missing required columns: {', '.join(missing_cols)}"]

    df = df.copy()

    for c in NUMERIC_RANGES:
        original = df[c]
        coerced = pd.to_numeric(original, errors="coerce")
        bad = coerced.isna() & original.notna()
        if bad.any():
            examples = original[bad].unique()[:5].tolist()
            errors.append(f"'{c}' has non-numeric values, e.g. {examples}")
        df[c] = coerced

    for c in ["has_therapy", "seeks_professional_help"]:
        df[c] = df[c].apply(to_binary)
        bad = df[c].isna()
        if bad.any():
            errors.append(f"'{c}' has values that aren't Yes/No/0/1 in {int(bad.sum())} row(s)")

    if not errors and df[REQUIRED_BATCH_COLS].isna().any().any():
        errors.append("Some required fields are blank — fill them in and re-upload.")

    return (None, errors) if errors else (df, [])


st.sidebar.header("Employee Signals")
predict_target = st.sidebar.radio("Predict", ["Burnout Score", "Seeks Professional Help"])
st.sidebar.divider()
input_mode = st.sidebar.radio("Data source", ["Manual input", "Upload CSV"])
st.sidebar.divider()

uploaded_df = None
values = {}
has_therapy = "No"
seeks_help_input = "No"
override_score, override_level = 2.0, "Low"

if input_mode == "Manual input":
    st.sidebar.subheader("Workload & Wellbeing")
    for c, (lo, hi, default) in NUMERIC_RANGES.items():
        values[c] = st.sidebar.slider(c.replace("_", " ").title(), lo, hi, default, step=0.5)

    has_therapy = st.sidebar.selectbox("Currently in therapy?", ["No", "Yes"])

    if predict_target == "Burnout Score":
        seeks_help_input = st.sidebar.selectbox("Currently seeking professional help?", ["No", "Yes"])
    else:
        if "burnout_score" in st.session_state:
            st.sidebar.success(
                f"Using burnout score {st.session_state['burnout_score']:.2f} "
                f"({st.session_state['burnout_level']}) from your last prediction."
            )
        else:
            st.sidebar.warning("Predict 'Burnout Score' first for a chained result.")
        with st.sidebar.expander("Override burnout score/level"):
            override_score = st.slider("Burnout score", 0.0, 10.0, st.session_state.get("burnout_score", 2.0))
            override_level = st.selectbox(
                "Burnout level", ["Low", "Moderate", "High"],
                index=["Low", "Moderate", "High"].index(st.session_state.get("burnout_level", "Low")),
            )
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Couldn't read that file: {e}")
            raw_df = None

        if raw_df is not None:
            cleaned_df, errors = validate_batch_df(raw_df)
            if errors:
                st.sidebar.error("Fix these before uploading again:\n\n" + "\n".join(f"- {e}" for e in errors))
            else:
                uploaded_df = cleaned_df
                st.sidebar.success(f"Loaded {len(uploaded_df)} valid rows")
                st.sidebar.dataframe(uploaded_df.head(), use_container_width=True)

st.title("Employee Burnout & Support Predictor")
st.caption(
    "Set an employee's signals in the sidebar, predict their burnout severity, "
    "then see how likely they are to already be seeking support."
)
st.divider()

if input_mode == "Manual input" and predict_target == "Burnout Score":
    st.caption("Predicts burnout_score from workload and wellbeing signals.")
    if st.button("Predict burnout score", type="primary"):
        try:
            from pycaret.regression import predict_model as predict_reg
            model = get_model("burnout_score")
            row = dict(values)
            row["has_therapy"] = yes_no(has_therapy)
            row["seeks_professional_help"] = yes_no(seeks_help_input)
            result = predict_reg(model, data=pd.DataFrame([row]))
            score = float(result["prediction_label"].iloc[0])
            band = burnout_band(score)
            st.session_state["burnout_score"] = score
            st.session_state["burnout_level"] = band

            with st.container(border=True):
                st.metric("Predicted burnout score", f"{score:.2f}")
                band_msg = f"Burnout level: **{band}**"
                {"Low": st.success, "Moderate": st.warning, "High": st.error}[band](band_msg)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Technical details"):
                st.exception(e)

elif input_mode == "Manual input" and predict_target == "Seeks Professional Help":
    st.caption("Predicts seeks_professional_help, using the burnout score/level from the sidebar.")
    if st.button("Predict help-seeking likelihood", type="primary"):
        try:
            from pycaret.classification import predict_model as predict_clf
            model = get_model("seeks_help")
            row = dict(values)
            row["has_therapy"] = yes_no(has_therapy)
            row["burnout_score"] = st.session_state.get("burnout_score", override_score)
            row["burnout_level"] = st.session_state.get("burnout_level", override_level)
            result = predict_clf(model, data=pd.DataFrame([row]), raw_score=True)
            label = int(result["prediction_label"].iloc[0])
            score_col = "prediction_score_1" if "prediction_score_1" in result.columns else "prediction_score"
            prob = float(result[score_col].iloc[0])

            with st.container(border=True):
                st.metric("Likely seeking help?", "Yes" if label == 1 else "No")
                st.progress(prob, text=f"{prob * 100:.1f}% confidence")
                (st.warning if label == 1 else st.success)(
                    "Flagged for follow-up" if label == 1 else "Not currently flagged"
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Technical details"):
                st.exception(e)

else:  # Upload CSV
    st.caption("Upload a CSV to predict burnout score and help-seeking likelihood for every row.")
    ready = uploaded_df is not None
    if st.button("Run batch prediction", type="primary", disabled=not ready):
        try:
            from pycaret.regression import predict_model as predict_reg
            from pycaret.classification import predict_model as predict_clf

            reg_model = get_model("burnout_score")
            clf_model = get_model("seeks_help")

            df = uploaded_df.copy()
            reg_result = predict_reg(reg_model, data=df[REQUIRED_BATCH_COLS])
            df["predicted_burnout_score"] = reg_result["prediction_label"].round(2)
            df["predicted_burnout_level"] = df["predicted_burnout_score"].apply(burnout_band)

            clf_input = df[list(NUMERIC_RANGES) + ["has_therapy"]].copy()
            clf_input["burnout_score"] = df["predicted_burnout_score"]
            clf_input["burnout_level"] = df["predicted_burnout_level"]
            clf_result = predict_clf(clf_model, data=clf_input, raw_score=True)
            score_col = "prediction_score_1" if "prediction_score_1" in clf_result.columns else "prediction_score"
            df["predicted_seeks_help"] = clf_result["prediction_label"]
            df["predicted_seeks_help_probability"] = clf_result[score_col].round(4)

            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows predicted", len(df))
                c2.metric("High burnout flagged", int((df["predicted_burnout_level"] == "High").sum()))
                c3.metric("Likely seeking help", int((df["predicted_seeks_help"] == 1).sum()))
                st.dataframe(df, use_container_width=True)
                st.download_button("Download results", df.to_csv(index=False), "burnout_predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            with st.expander("Technical details"):
                st.exception(e)
=======
import streamlit as st
import pandas as pd

MODEL_DIR = "model"

st.set_page_config(page_title="Employee Burnout Predictor", layout="wide")

NUMERIC_RANGES = {
    "work_hours_per_week": (20.0, 90.0, 47.0),
    "overtime_hours": (0.0, 40.0, 6.0),
    "job_satisfaction": (0.0, 10.0, 6.0),
    "manager_support": (0.0, 10.0, 6.0),
    "work_life_balance": (0.0, 10.0, 6.0),
    "sleep_hours": (0.0, 12.0, 6.5),
    "stress_level": (0.0, 10.0, 5.5),
    "anxiety_score": (0.0, 10.0, 5.0),
    "depression_score": (0.0, 10.0, 4.0),
}

REQUIRED_BATCH_COLS = list(NUMERIC_RANGES) + ["has_therapy", "seeks_professional_help"]


@st.cache_resource(max_entries=1)
def get_model(task):
    if task == "burnout_score":
        from pycaret.regression import load_model
        return load_model(f"{MODEL_DIR}/burnout_score_pipeline")
    elif task == "seeks_help":
        from pycaret.classification import load_model
        return load_model(f"{MODEL_DIR}/seeks_help_pipeline")


def burnout_band(score):
    if score <= 2:
        return "Low"
    if score <= 4:
        return "Moderate"
    return "High"


def yes_no(v):
    return 1 if v == "Yes" else 0


def to_binary(v):
    """Coerces a batch CSV cell to 0/1. Returns None if it's not recognizable,
    so the caller can flag exactly which rows are bad instead of guessing."""
    if isinstance(v, (int, float)) and not isinstance(v, bool) and v in (0, 1):
        return int(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("yes", "y", "1", "true"):
            return 1
        if s in ("no", "n", "0", "false"):
            return 0
    return None


def validate_batch_df(df):
    """Checks columns exist, then checks every value is actually usable —
    catches things like a stray 'Twenty-Five' in a numeric column or a typo
    in a yes/no column before it ever reaches the model."""
    errors = []
    missing_cols = [c for c in REQUIRED_BATCH_COLS if c not in df.columns]
    if missing_cols:
        return None, [f"Missing required columns: {', '.join(missing_cols)}"]

    df = df.copy()

    for c in NUMERIC_RANGES:
        original = df[c]
        coerced = pd.to_numeric(original, errors="coerce")
        bad = coerced.isna() & original.notna()
        if bad.any():
            examples = original[bad].unique()[:5].tolist()
            errors.append(f"'{c}' has non-numeric values, e.g. {examples}")
        df[c] = coerced

    for c in ["has_therapy", "seeks_professional_help"]:
        df[c] = df[c].apply(to_binary)
        bad = df[c].isna()
        if bad.any():
            errors.append(f"'{c}' has values that aren't Yes/No/0/1 in {int(bad.sum())} row(s)")

    if not errors and df[REQUIRED_BATCH_COLS].isna().any().any():
        errors.append("Some required fields are blank — fill them in and re-upload.")

    return (None, errors) if errors else (df, [])


st.sidebar.header("Employee Signals")
predict_target = st.sidebar.radio("Predict", ["Burnout Score", "Seeks Professional Help"])
st.sidebar.divider()
input_mode = st.sidebar.radio("Data source", ["Manual input", "Upload CSV"])
st.sidebar.divider()

uploaded_df = None
values = {}
has_therapy = "No"
seeks_help_input = "No"
override_score, override_level = 2.0, "Low"

if input_mode == "Manual input":
    st.sidebar.subheader("Workload & Wellbeing")
    for c, (lo, hi, default) in NUMERIC_RANGES.items():
        values[c] = st.sidebar.slider(c.replace("_", " ").title(), lo, hi, default, step=0.5)

    has_therapy = st.sidebar.selectbox("Currently in therapy?", ["No", "Yes"])

    if predict_target == "Burnout Score":
        seeks_help_input = st.sidebar.selectbox("Currently seeking professional help?", ["No", "Yes"])
    else:
        if "burnout_score" in st.session_state:
            st.sidebar.success(
                f"Using burnout score {st.session_state['burnout_score']:.2f} "
                f"({st.session_state['burnout_level']}) from your last prediction."
            )
        else:
            st.sidebar.warning("Predict 'Burnout Score' first for a chained result.")
        with st.sidebar.expander("Override burnout score/level"):
            override_score = st.slider("Burnout score", 0.0, 10.0, st.session_state.get("burnout_score", 2.0))
            override_level = st.selectbox(
                "Burnout level", ["Low", "Moderate", "High"],
                index=["Low", "Moderate", "High"].index(st.session_state.get("burnout_level", "Low")),
            )
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.sidebar.error(f"Couldn't read that file: {e}")
            raw_df = None

        if raw_df is not None:
            cleaned_df, errors = validate_batch_df(raw_df)
            if errors:
                st.sidebar.error("Fix these before uploading again:\n\n" + "\n".join(f"- {e}" for e in errors))
            else:
                uploaded_df = cleaned_df
                st.sidebar.success(f"Loaded {len(uploaded_df)} valid rows")
                st.sidebar.dataframe(uploaded_df.head(), use_container_width=True)

st.title("Employee Burnout & Support Predictor")
st.caption(
    "Set an employee's signals in the sidebar, predict their burnout severity, "
    "then see how likely they are to already be seeking support."
)
st.divider()

if input_mode == "Manual input" and predict_target == "Burnout Score":
    st.caption("Predicts burnout_score from workload and wellbeing signals.")
    if st.button("Predict burnout score", type="primary"):
        try:
            from pycaret.regression import predict_model as predict_reg
            model = get_model("burnout_score")
            row = dict(values)
            row["has_therapy"] = yes_no(has_therapy)
            row["seeks_professional_help"] = yes_no(seeks_help_input)
            result = predict_reg(model, data=pd.DataFrame([row]))
            score = float(result["prediction_label"].iloc[0])
            band = burnout_band(score)
            st.session_state["burnout_score"] = score
            st.session_state["burnout_level"] = band

            with st.container(border=True):
                st.metric("Predicted burnout score", f"{score:.2f}")
                band_msg = f"Burnout level: **{band}**"
                {"Low": st.success, "Moderate": st.warning, "High": st.error}[band](band_msg)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Technical details"):
                st.exception(e)

elif input_mode == "Manual input" and predict_target == "Seeks Professional Help":
    st.caption("Predicts seeks_professional_help, using the burnout score/level from the sidebar.")
    if st.button("Predict help-seeking likelihood", type="primary"):
        try:
            from pycaret.classification import predict_model as predict_clf
            model = get_model("seeks_help")
            row = dict(values)
            row["has_therapy"] = yes_no(has_therapy)
            row["burnout_score"] = st.session_state.get("burnout_score", override_score)
            row["burnout_level"] = st.session_state.get("burnout_level", override_level)
            result = predict_clf(model, data=pd.DataFrame([row]), raw_score=True)
            label = int(result["prediction_label"].iloc[0])
            score_col = "prediction_score_1" if "prediction_score_1" in result.columns else "prediction_score"
            prob = float(result[score_col].iloc[0])

            with st.container(border=True):
                st.metric("Likely seeking help?", "Yes" if label == 1 else "No")
                st.progress(prob, text=f"{prob * 100:.1f}% confidence")
                (st.warning if label == 1 else st.success)(
                    "Flagged for follow-up" if label == 1 else "Not currently flagged"
                )
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("Technical details"):
                st.exception(e)

else:  # Upload CSV
    st.caption("Upload a CSV to predict burnout score and help-seeking likelihood for every row.")
    ready = uploaded_df is not None
    if st.button("Run batch prediction", type="primary", disabled=not ready):
        try:
            from pycaret.regression import predict_model as predict_reg
            from pycaret.classification import predict_model as predict_clf

            reg_model = get_model("burnout_score")
            clf_model = get_model("seeks_help")

            df = uploaded_df.copy()
            reg_result = predict_reg(reg_model, data=df[REQUIRED_BATCH_COLS])
            df["predicted_burnout_score"] = reg_result["prediction_label"].round(2)
            df["predicted_burnout_level"] = df["predicted_burnout_score"].apply(burnout_band)

            clf_input = df[list(NUMERIC_RANGES) + ["has_therapy"]].copy()
            clf_input["burnout_score"] = df["predicted_burnout_score"]
            clf_input["burnout_level"] = df["predicted_burnout_level"]
            clf_result = predict_clf(clf_model, data=clf_input, raw_score=True)
            score_col = "prediction_score_1" if "prediction_score_1" in clf_result.columns else "prediction_score"
            df["predicted_seeks_help"] = clf_result["prediction_label"]
            df["predicted_seeks_help_probability"] = clf_result[score_col].round(4)

            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Rows predicted", len(df))
                c2.metric("High burnout flagged", int((df["predicted_burnout_level"] == "High").sum()))
                c3.metric("Likely seeking help", int((df["predicted_seeks_help"] == 1).sum()))
                st.dataframe(df, use_container_width=True)
                st.download_button("Download results", df.to_csv(index=False), "burnout_predictions.csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
            with st.expander("Technical details"):
                st.exception(e)
>>>>>>> main
