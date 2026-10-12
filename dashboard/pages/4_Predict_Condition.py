
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
MODEL_PATH = RESULTS / "models" / "independent_bearing_model.joblib"
DATA_PATH = RESULTS / "independent_test_data.csv"

FEATURES = [
    "mean", "std", "rms", "peak",
    "peak_to_peak", "kurtosis", "crest_factor",
]

st.set_page_config(page_title="Predict Condition",
                   page_icon="🔍", layout="wide")
st.title("🔍 Predict Bearing Condition")
st.caption(
    "Run the saved model on a selected row from the independent test dataset.")

st.warning(
    "Experimental prediction only. This model was developed using Paderborn "
    "bearing data and has not been validated for reliable blender fault diagnosis."
)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not DATA_PATH.exists():
    st.error(f"Test dataset not found: {DATA_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        st.error(f"Test dataset is missing required features: {missing}")
        st.write("Available columns:", list(df.columns))
        st.stop()

    if df.empty:
        st.error("The test dataset contains no rows.")
        st.stop()

    st.metric("Available test samples", f"{len(df):,}")

    row_number = st.number_input(
        "Test sample row number (starting at 1)",
        min_value=1,
        max_value=len(df),
        value=1,
        step=1,
    )

    row = df.iloc[[int(row_number) - 1]]
    sample = row[FEATURES].apply(pd.to_numeric, errors="coerce")

    st.subheader("Selected sample features")
    st.dataframe(sample, use_container_width=True, hide_index=True)

    if sample.isna().any().any():
        st.error("This sample contains missing or non-numeric feature values.")
        st.stop()

    if st.button("Predict bearing condition", type="primary"):
        prediction = str(model.predict(sample)[0])
        st.subheader("Prediction result")
        st.write(f"**Predicted class:** `{prediction}`")

        normalized = prediction.lower().replace(" ", "_")
        if normalized == "healthy":
            st.success("The model predicts the healthy class.")
        elif "fault" in normalized:
            st.error("The model predicts a fault class.")
        else:
            st.info("The model returned a class that needs interpretation.")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(sample)[0]
            classes = model.classes_

            result = pd.DataFrame({
                "Class": classes,
                "Model probability (%)": probabilities * 100,
            }).sort_values("Model probability (%)", ascending=False)

            st.subheader("Model output probabilities")
            st.bar_chart(result.set_index("Class"))
            st.dataframe(
                result.style.format({"Model probability (%)": "{:.2f}%"}),
                use_container_width=True,
                hide_index=True,
            )

        st.caption(
            "Probabilities are model outputs, not verified real-world "
            "health confidence. The selected row's true label is not "
            "used as an input to the prediction."
        )

except Exception as exc:
    st.error(f"Prediction could not be completed: {exc}")
