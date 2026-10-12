
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Motor Comparison",
    page_icon="📊",
    layout="wide"
)

st.title("Motor OFF vs ON Comparison")
st.write(
    "Compare vibration features extracted from the two recorded "
    "operating conditions."
)

BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"

comparison_file = RESULTS_DIR / "motor_condition_comparison.csv"
report_file = RESULTS_DIR / "motor_analysis_report.txt"

if not comparison_file.exists():
    st.error("Comparison data not found. Run the comparison script first.")
    st.stop()

try:
    df = pd.read_csv(comparison_file, index_col=0)
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not read comparison data: {error}")
    st.stop()

required_columns = {"Motor OFF", "Motor ON", "Difference (ON - OFF)"}
missing_columns = required_columns - set(df.columns)
if missing_columns:
    st.error(
        "Comparison data is missing required columns: "
        + ", ".join(sorted(missing_columns))
    )
    st.stop()

st.subheader("Feature comparison")
st.dataframe(df.round(4), width="stretch")
try:
    comparison_csv = comparison_file.read_bytes()
except OSError as error:
    st.warning(f"Could not prepare the comparison CSV download: {error}")
else:
    st.download_button(
        label="Download comparison data CSV",
        data=comparison_csv,
        file_name=comparison_file.name,
        mime="text/csv",
    )

st.subheader("Motor OFF vs Motor ON")
chart_data = df[["Motor OFF", "Motor ON"]]
st.bar_chart(chart_data)

st.subheader("Change in each feature")
st.bar_chart(df[["Difference (ON - OFF)"]])

st.info(
    "A difference between motor OFF and ON measurements does not "
    "confirm a mechanical fault. Sensor orientation, mounting, noise, "
    "and operating conditions can affect the readings."
)

if report_file.exists():
    try:
        report = report_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        st.warning(f"Could not read the motor analysis report: {error}")
    else:
        with st.expander("View complete analysis report"):
            st.text(report)

        st.download_button(
            label="Download analysis report",
            data=report,
            file_name="motor_analysis_report.txt",
            mime="text/plain"
        )
else:
    st.info("Motor analysis report was not found; the download is unavailable.")
