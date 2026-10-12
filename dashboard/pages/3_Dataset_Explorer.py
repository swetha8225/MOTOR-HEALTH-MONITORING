
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

st.set_page_config(page_title="Dataset Explorer", page_icon="📂", layout="wide")
st.title("📂 Paderborn Dataset Explorer")
st.caption(
    "Explore the saved feature datasets used in the bearing-condition project.")

options = {
    "Paderborn features": RESULTS / "paderborn_features.csv",
    "Labelled Paderborn features": RESULTS / "paderborn_labeled_features.csv",
    "Independent training data": RESULTS / "independent_train_data.csv",
    "Independent test data": RESULTS / "independent_test_data.csv",
}

choice = st.selectbox("Choose a dataset", list(options))
path = options[choice]

if not path.exists():
    st.error(f"Dataset file not found: {path}")
    st.stop()

try:
    df = pd.read_csv(path)
    st.subheader(choice)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

    st.write("**Columns and data types**")
    schema = pd.DataFrame({
        "Column": df.columns,
        "Data type": df.dtypes.astype(str).values,
    })
    st.dataframe(schema, use_container_width=True, hide_index=True)

    st.write("**Dataset preview**")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    if not df.empty:
        st.download_button(
            "Download selected dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=path.name,
            mime="text/csv",
        )

    numeric = df.select_dtypes(include="number")
    if not numeric.empty:
        st.subheader("Feature distributions")
        selected = st.selectbox(
            "Choose a numeric column",
            list(numeric.columns),
        )
        st.bar_chart(numeric[selected].value_counts().sort_index().head(50))
        st.subheader("Descriptive statistics")
        st.dataframe(numeric.describe().T, use_container_width=True)

    for col in df.columns:
        if any(word in col.lower() for word in ["label", "condition", "class", "bearing_id"]):
            st.subheader(f"Distribution: {col}")
            st.bar_chart(df[col].fillna("Missing").astype(str).value_counts())

except Exception as exc:
    st.error(f"Could not load the selected dataset: {exc}")
