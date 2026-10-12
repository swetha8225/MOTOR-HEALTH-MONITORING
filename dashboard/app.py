
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ==================================================
# PATHS AND CONFIGURATION
# ==================================================
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MODEL_PATH = RESULTS / "models" / "independent_bearing_model.joblib"
RECORDING_PATH = RESULTS / "latest_motor_recording.csv"
FEATURES_PATH = RESULTS / "latest_motor_features.csv"
EVALUATION_PATH = RESULTS / "independent_evaluation_report.txt"

FEATURE_NAMES = [
    "mean",
    "std",
    "rms",
    "peak",
    "peak_to_peak",
    "kurtosis",
    "crest_factor",
]

st.set_page_config(
    page_title="Motor Health Monitoring",
    page_icon="⚙️",
    layout="wide",
)

# ==================================================
# HEADER
# ==================================================
st.title("⚙️ Motor Health Monitoring Dashboard")
st.caption(
    "Low-Cost Embedded Multi-Sensor Health Monitoring | "
    "ESP32 • MPU6050 • ACS712 • Experimental ML"
)

# ==================================================
# SIDEBAR INFORMATION
# ==================================================
with st.sidebar:
    st.header("Project Information")
    st.write("**Application:** Domestic motor / blender")
    st.write("**Controller:** ESP32")
    st.write("**Vibration sensor:** MPU6050")
    st.write("**Current sensor:** ACS712")
    st.write("**ML dataset:** Paderborn bearing dataset")

    st.divider()
    st.caption(
        "Use the navigation menu above to access Dataset Explorer, "
        "Predict Condition, Model Evaluation, Live Monitoring, "
        "Prediction Comparison, and Motor Comparison."
    )

# ==================================================
# LOAD MODEL
# ==================================================


@st.cache_resource
def load_model(path, modified_time):
    return joblib.load(path)


model = None

if MODEL_PATH.exists():
    try:
        model = load_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime,
        )
    except Exception as exc:
        st.error(f"Could not load the trained model: {exc}")
else:
    st.warning(f"Model file not found: {MODEL_PATH}")

# ==================================================
# OVERVIEW
# ==================================================
st.header("Project Overview")

st.markdown(
    """
    This project investigates low-cost motor condition monitoring by
    collecting vibration and current-sensor readings, extracting
    statistical features, and exploring machine-learning predictions.

    **Main objectives**
    - Collect sensor readings using an ESP32.
    - Monitor vibration using the MPU6050.
    - Record raw ACS712 ADC readings.
    - Extract statistical features from saved recordings.
    - Explore Paderborn bearing data and evaluate a trained model.
    - Compare motor OFF and motor ON recordings.
    """
)

# ==================================================
# SYSTEM STATUS
# ==================================================
st.subheader("System Status")

status_cols = st.columns(4)

status_cols[0].metric(
    "ML Model",
    "Loaded" if model is not None else "Unavailable",
)

status_cols[1].metric(
    "Latest Recording",
    "Available" if RECORDING_PATH.exists() else "Not found",
)

status_cols[2].metric(
    "Latest Features",
    "Available" if FEATURES_PATH.exists() else "Not found",
)

status_cols[3].metric(
    "Evaluation Report",
    "Available" if EVALUATION_PATH.exists() else "Not found",
)

st.divider()

# ==================================================
# LATEST SENSOR RECORDING
# ==================================================
st.header("1. Latest Sensor Recording")

if RECORDING_PATH.exists():
    try:
        recording = pd.read_csv(RECORDING_PATH)

        if recording.empty:
            st.warning("The latest sensor recording contains no rows.")
        else:
            st.metric("Recorded samples", f"{len(recording):,}")

            st.subheader("Recent sensor readings")
            st.dataframe(
                recording.tail(20),
                use_container_width=True,
                hide_index=True,
            )

            if "time_ms" in recording.columns:
                st.caption(
                    "The time axis is shown in milliseconds when "
                    "available in the saved recording."
                )

            if "total_accel" in recording.columns:
                acceleration = pd.to_numeric(
                    recording["total_accel"],
                    errors="coerce",
                ).dropna()

                if not acceleration.empty:
                    st.subheader("Total Acceleration")
                    st.line_chart(acceleration.reset_index(drop=True))

            if "acs_adc" in recording.columns:
                current_readings = pd.to_numeric(
                    recording["acs_adc"],
                    errors="coerce",
                ).dropna()

                if not current_readings.empty:
                    st.subheader("ACS712 Sensor Output")
                    st.line_chart(current_readings.reset_index(drop=True))
                    st.caption(
                        "ACS712 values are raw ADC counts, not calibrated "
                        "current in amperes."
                    )

    except Exception as exc:
        st.error(f"Could not read the sensor recording: {exc}")
else:
    st.info(
        "No latest recording was found. Use your existing ESP32 "
        "acquisition workflow to collect sensor data."
    )

st.divider()

# ==================================================
# LATEST FEATURE VALUES AND MODEL PREDICTION
# ==================================================
st.header("2. Latest Model Prediction")

st.warning(
    "Experimental model only: the trained model uses Paderborn bearing "
    "data. It has not been validated for reliable fault diagnosis on "
    "your actual blender motor."
)

if model is None:
    st.info("Prediction is unavailable because the model could not be loaded.")

elif not FEATURES_PATH.exists():
    st.info(
        "No latest feature file was found. Run the feature extraction "
        "workflow before requesting a prediction."
    )

else:
    try:
        features_df = pd.read_csv(FEATURES_PATH)

        missing = [
            name for name in FEATURE_NAMES
            if name not in features_df.columns
        ]

        if missing:
            st.error(
                "The feature file is missing required columns: "
                + ", ".join(missing)
            )

        elif features_df.empty:
            st.warning("The feature file contains no rows.")

        else:
            sample = features_df[FEATURE_NAMES].tail(1)
            sample = sample.apply(pd.to_numeric, errors="coerce")

            st.subheader("Latest Extracted Features")
            st.dataframe(
                sample.T.rename(columns={sample.index[0]: "Value"}),
                use_container_width=True,
            )

            if sample.isna().any().any():
                st.warning(
                    "The latest feature row contains missing or "
                    "non-numeric values. Prediction was not performed."
                )

            else:
                if st.button("Predict Latest Feature Row", type="primary"):
                    prediction = str(model.predict(sample)[0]).strip()
                    normalized = prediction.lower().replace(" ", "_")

                    st.subheader("Prediction Result")

                    if normalized == "healthy":
                        st.success(
                            f"Predicted class: {prediction}"
                        )
                    elif "fault" in normalized:
                        st.error(
                            f"Predicted class: {prediction}"
                        )
                    else:
                        st.warning(
                            f"Predicted class: {prediction}"
                        )

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(sample)[0]
                        classes = model.classes_

                        probability_df = pd.DataFrame({
                            "Class": classes,
                            "Model probability (%)": probabilities * 100,
                        }).sort_values(
                            "Model probability (%)",
                            ascending=False,
                        )

                        st.subheader("Model Output Probabilities")
                        st.bar_chart(
                            probability_df.set_index("Class")
                            ["Model probability (%)"]
                        )

                        st.dataframe(
                            probability_df.style.format(
                                {"Model probability (%)": "{:.2f}%"}
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.caption(
                            "Probabilities are model outputs, not verified "
                            "real-world health confidence."
                        )

                    st.info(
                        "This prediction uses the latest saved feature row. "
                        "It is not necessarily synchronized with the latest "
                        "raw sensor reading."
                    )

    except Exception as exc:
        st.error(f"Prediction could not be completed: {exc}")

st.divider()

# ==================================================
# DATASET AND EVALUATION FILES
# ==================================================
st.header("3. Project Resources")

resource_cols = st.columns(2)

with resource_cols[0]:
    st.subheader("Available Dataset Files")

    dataset_files = [
        RESULTS / "paderborn_features.csv",
        RESULTS / "paderborn_labeled_features.csv",
        RESULTS / "independent_train_data.csv",
        RESULTS / "independent_test_data.csv",
    ]

    for path in dataset_files:
        if path.exists():
            st.write(f"✅ `{path.name}`")
        else:
            st.write(f"— `{path.name}` not found")

with resource_cols[1]:
    st.subheader("Model Evaluation")

    if EVALUATION_PATH.exists():
        if st.button("View Evaluation Report"):
            try:
                report = EVALUATION_PATH.read_text(encoding="utf-8")
                st.text_area(
                    "Saved evaluation report",
                    value=report,
                    height=300,
                    disabled=True,
                )
            except Exception as exc:
                st.error(f"Could not read the report: {exc}")
    else:
        st.info("No saved evaluation report was found.")

# ==================================================
# INTERPRETATION AND LIMITATIONS
# ==================================================
st.divider()
st.header("4. Interpretation and Limitations")

st.markdown(
    """
    - Sensor readings describe the recorded signals, not a confirmed fault.
    - Statistical features summarize the characteristics of a signal window.
    - The ML prediction reflects the class learned from its training data.
    - Paderborn bearing-dataset performance does not establish accuracy
      on a domestic blender motor.
    - Raw ACS712 ADC counts cannot be interpreted as amperes without
      appropriate calibration.
    - Validate the system with representative, labelled motor recordings
      before relying on it for maintenance or safety decisions.
    """
)

st.caption(
    "Project dashboard | Experimental prototype | "
    "Predictions are for evaluation, not safety-critical decisions."
)
