
from pathlib import Path
import pandas as pd

from bearing_labels import LABEL_MAP

input_file = Path("results/paderborn_features.csv")
output_file = Path("results/paderborn_labeled_features.csv")

if not input_file.exists():
    raise FileNotFoundError(
        f"Feature file not found: {input_file}"
    )

df = pd.read_csv(input_file)

if "bearing_id" not in df.columns:
    raise ValueError("The feature CSV has no bearing_id column.")

df["condition"] = df["bearing_id"].map(LABEL_MAP)

unmapped = df[df["condition"].isna()]["bearing_id"].unique()

if len(unmapped) > 0:
    raise ValueError(
        f"These bearing IDs have no labels: {list(unmapped)}"
    )

df.to_csv(output_file, index=False)

print("=" * 55)
print("LABELED FEATURE DATASET CREATED")
print("=" * 55)
print("Input file:", input_file)
print("Output file:", output_file)
print("Total feature rows:", len(df))
print("Unique bearings:", df["bearing_id"].nunique())
print("Missing labels:", df["condition"].isna().sum())

print("\nRows per condition:")
print(df.groupby("condition").size())

print("\nPreview:")
print(df[["bearing_id", "window", "rms", "kurtosis", "condition"]].head(10))
