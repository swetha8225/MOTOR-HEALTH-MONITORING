
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

REPORT_PATH = RESULTS / "independent_evaluation_report.txt"
COMPARISON_PATH = RESULTS / "models" / "model_comparison.csv"
TEST_PATH = RESULTS / "independent_test_data.csv"
TRAIN_PATH = RESULTS / "independent_train_data.csv"

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Model Evaluation")
st.caption("Review the saved bearing-model evaluation and comparison results.")

# Dataset summary
st.subheader("Dataset summary")
col1, col2 = st.columns(2)

for col, label, path in [
    (col1, "Independent training data", TRAIN_PATH),
    (col2, "Independent test data", TEST_PATH),
]:
    with col:
        if path.exists():
            try:
                df = pd.read_csv(path)
                st.metric(f"{label} rows", f"{len(df):,}")
                st.caption(f"Columns: {len(df.columns)}")
            except Exception as exc:
                st.error(f"Could not read {path.name}: {exc}")
        else:
            st.warning(f"File not found: {path.name}")

# Evaluation report
st.divider()
st.subheader("Saved evaluation report")

if REPORT_PATH.exists():
    try:
        report_text = REPORT_PATH.read_text(encoding="utf-8")
        st.text_area(
            "Evaluation details",
            value=report_text,
            height=300,
            disabled=True,
        )
        st.download_button(
            "Download evaluation report",
            data=report_text,
            file_name=REPORT_PATH.name,
            mime="text/plain",
        )
    except Exception as exc:
        st.error(f"Could not read evaluation report: {exc}")
else:
    st.info("No independent evaluation report was found.")

# Model comparison
st.divider()
st.subheader("Model comparison")

if COMPARISON_PATH.exists():
    try:
        comparison = pd.read_csv(COMPARISON_PATH)
        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        numeric = comparison.select_dtypes(include="number")
        if not numeric.empty:
            metric_options = list(numeric.columns)
            metric = st.selectbox(
                "Choose a numeric metric to compare",
                metric_options,
            )

            chart_data = comparison.copy()
            chart_data[metric] = pd.to_numeric(
                chart_data[metric], errors="coerce"
            )
            chart_data = chart_data.dropna(subset=[metric])

            if not chart_data.empty:
                label_candidates = [
                    c for c in comparison.columns
                    if c not in numeric.columns
                ]
                if label_candidates:
                    label_col = label_candidates[0]
                    chart_data = chart_data.set_index(label_col)
                    st.bar_chart(chart_data[metric])
                else:
                    st.bar_chart(chart_data[metric])
    except Exception as exc:
        st.error(f"Could not read model comparison: {exc}")
else:
    st.info("No model comparison CSV was found.")

st.divider()
st.subheader("How to interpret these results")
st.markdown(
    """
    - Training and test row counts describe the saved datasets; they are not
      measures of model accuracy.
    - Use the evaluation report for metrics actually calculated during testing.
    - Model comparisons are meaningful only when the models were evaluated
      on comparable data using the same metrics.
    - These bearing-dataset results do not establish reliable fault-detection
      accuracy for a domestic blender.
    """
)
